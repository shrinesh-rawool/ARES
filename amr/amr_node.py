"""
amr/amr_node.py
Autonomous edge node with Dynamic Aisle Obstruction Discovery & P2P Costmap Sharing.
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Tuple, Set
import zmq
from core.planner import SpaceTimeAStar


class AMRNode:
    def __init__(
        self,
        robot_id: str,
        init_x: int,
        init_y: int,
        goal_x: int,
        goal_y: int,
        warehouse_addr: str,
        pub_port: int,
        peer_addrs: list,
    ):
        self.robot_id = robot_id
        self.pos = (init_x, init_y)
        self.goal = (goal_x, goal_y)

        self.context = zmq.Context()

        # 1. Warehouse Sim Socket
        self.warehouse_sock = self.context.socket(zmq.REQ)
        self.warehouse_sock.connect(warehouse_addr)

        # 2. P2P Mesh Publisher Socket
        self.pub_sock = self.context.socket(zmq.PUB)
        self.pub_sock.bind(f"tcp://0.0.0.0:{pub_port}")

        # 3. P2P Mesh Subscriber Socket
        self.sub_sock = self.context.socket(zmq.SUB)
        for peer in peer_addrs:
            self.sub_sock.connect(peer)
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self.peer_states: Dict[str, dict] = {}
        self.planned_path: List[Tuple[int, int]] = []
        self.planner: Optional[SpaceTimeAStar] = None
        self.static_obstacles: frozenset = frozenset()

        # Track discovered dynamic obstacles (Scenario C Costmap Delta)
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()

        self._register_with_warehouse(init_x, init_y)

        self.has_payload: bool = True  # Starts carrying a package to its destination

    def _register_with_warehouse(self, x: int, y: int):
        req = {"type": "REGISTER", "robot_id": self.robot_id, "x": x, "y": y}
        self.warehouse_sock.send_json(req)
        res = self.warehouse_sock.recv_json()

        self.static_obstacles = frozenset(tuple(p) for p in res.get("static_map", []))
        self.rebuild_planner()
        print(f"[{self.robot_id}] Registered at {self.pos}. Static map loaded ({len(self.static_obstacles)} obstacles).")

    def rebuild_planner(self):
        """Re-instantiates Space-Time A* combining static layout and discovered obstacles."""
        combined_obs = self.static_obstacles | frozenset(self.dynamic_obstacles)
        self.planner = SpaceTimeAStar(30, 30, static_obstacles=combined_obs)

    def broadcast_telemetry(self):
        """Broadcast state, path, and discovered dynamic obstacles at 10 Hz."""
        payload = {
            "robot_id": self.robot_id,
            "x": self.pos[0],
            "y": self.pos[1],
            "goal": self.goal,
            "planned_path": self.planned_path,
            "dist_to_goal": abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1]),
            "dynamic_obstacles": list(self.dynamic_obstacles),
            "timestamp": time.time(),
        }
        self.pub_sock.send_json(payload)

    def update_peers(self):
        """Non-blocking ingestion of peer telemetry and P2P costmap updates."""
        costmap_updated = False
        while True:
            try:
                msg = self.sub_sock.recv_json(flags=zmq.NOBLOCK)
                p_id = msg["robot_id"]
                self.peer_states[p_id] = {
                    "pos": (msg["x"], msg["y"]),
                    "goal": msg.get("goal"),
                    "planned_path": msg.get("planned_path", []),
                    "dist_to_goal": msg.get("dist_to_goal", 999),
                    "last_seen": msg["timestamp"],
                }

                # Anomaly Broadcast Ingestion (Scenario C)
                peer_obs = msg.get("dynamic_obstacles", [])
                for obs in peer_obs:
                    t_obs = tuple(obs)
                    if t_obs not in self.dynamic_obstacles:
                        self.dynamic_obstacles.add(t_obs)
                        costmap_updated = True
            except zmq.Again:
                break

        if costmap_updated:
            print(f"[{self.robot_id}] Received P2P costmap update from peer. Triggering sub-local replanning...")
            self.rebuild_planner()
            reservations = self.get_peer_reservations()
            self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []

    def get_peer_reservations(self) -> Dict[int, Tuple[int, int]]:
        reservations: Dict[int, Tuple[int, int]] = {}
        for p_id, p_info in self.peer_states.items():
            p_path = p_info.get("planned_path", [])
            for t, coord in enumerate(p_path):
                reservations[t] = tuple(coord)
            cur = p_info.get("pos")
            if cur:
                for t in range(4):
                    reservations[t] = cur
        return reservations

    def has_higher_priority(self, peer_id: str, peer_dist: int) -> bool:
        my_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        if my_dist < peer_dist:
            return True
        elif my_dist == peer_dist:
            return self.robot_id < peer_id
        return False

    def find_sidestep_cell(self) -> Optional[Tuple[int, int]]:
        cx, cy = self.pos
        candidates = [(cx, cy + 1), (cx, cy - 1), (cx - 1, cy), (cx + 1, cy)]
        all_blocked = self.static_obstacles | self.dynamic_obstacles
        peer_occupied = {p_info["pos"] for p_info in self.peer_states.values() if "pos" in p_info}

        for nx, ny in candidates:
            if not (0 <= nx < 30 and 0 <= ny < 30):
                continue
            if (nx, ny) in all_blocked or (nx, ny) in peer_occupied:
                continue
            return (nx, ny)
        return None

    def step(self, next_x: int, next_y: int) -> bool:
        req = {
            "type": "STEP",
            "robot_id": self.robot_id,
            "next_x": next_x,
            "next_y": next_y,
            "goal": self.goal,
            "has_payload": self.has_payload,
        }
        self.warehouse_sock.send_json(req)
        res = self.warehouse_sock.recv_json()

        if not res.get("collision", False):
            self.pos = (next_x, next_y)
            return True
        return False

    def run(self):
        print(f"[{self.robot_id}] Autonomous loop started. Target goal: {self.goal}")

        reservations = self.get_peer_reservations()
        self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []

        stuck_ticks = 0

        while True:
            self.broadcast_telemetry()
            self.update_peers()

            # Continuous back-and-forth transit
            if self.pos == self.goal:
                action_str = "Delivered package!" if self.has_payload else "Loaded package!"
                print(f"[{self.robot_id}] Reached {self.goal}. {action_str} Swapping in 2s...")

                # Toggle payload state (delivery completed -> return empty, or pick up new package)
                self.has_payload = not self.has_payload

                time.sleep(2.0)
                if self.robot_id == "amr_1":
                    self.goal = (1, 14) if self.pos == (28, 14) else (28, 14)
                elif self.robot_id == "amr_2":
                    self.goal = (14, 1) if self.pos == (14, 28) else (14, 28)
                elif self.robot_id == "amr_3":
                    self.goal = (28, 14) if self.pos == (1, 14) else (1, 14)

                reservations = self.get_peer_reservations()
                self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []
                continue

            # Drop current pos
            while self.planned_path and self.planned_path[0] == self.pos:
                self.planned_path.pop(0)

            if not self.planned_path:
                reservations = self.get_peer_reservations()
                self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []
                if not self.planned_path:
                    time.sleep(0.3)
                    continue

            next_coord = self.planned_path[0]

            # Peer priority arbitration
            yield_needed = False
            for p_id, p_info in self.peer_states.items():
                p_pos = p_info.get("pos")
                if not p_pos:
                    continue

                dist_to_peer = abs(self.pos[0] - p_pos[0]) + abs(self.pos[1] - p_pos[1])
                if dist_to_peer <= 2:
                    peer_dist = p_info.get("dist_to_goal", 999)
                    if not self.has_higher_priority(p_id, peer_dist):
                        yield_needed = True
                        break

            if yield_needed:
                stuck_ticks += 1
                if stuck_ticks >= 2:
                    sidestep_coord = self.find_sidestep_cell()
                    if sidestep_coord:
                        print(f"[{self.robot_id}] Sidestepping to {sidestep_coord} to clear corridor...")
                        self.step(sidestep_coord[0], sidestep_coord[1])
                        time.sleep(0.4)
                        reservations = self.get_peer_reservations()
                        self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []
                        stuck_ticks = 0
                        continue
                time.sleep(0.3)
                continue

            # Execute step
            success = self.step(next_coord[0], next_coord[1])

            if success:
                self.planned_path.pop(0)
                stuck_ticks = 0
            else:
                stuck_ticks += 1
                # Check if cell is occupied by a peer robot
                peer_in_cell = any(p_info.get("pos") == next_coord for p_info in self.peer_states.values())

                if not peer_in_cell:
                    # Physical blockage discovered (Scenario C: Fallen box / dynamic obstruction)
                    print(f"[{self.robot_id}] Detected physical obstacle at {next_coord}! Broadcasting costmap delta...")
                    self.dynamic_obstacles.add(next_coord)
                    self.rebuild_planner()
                    reservations = self.get_peer_reservations()
                    self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []
                else:
                    # Temporary peer conflict, wait or replan
                    time.sleep(0.2)
                    if stuck_ticks > 3:
                        reservations = self.get_peer_reservations()
                        self.planned_path = self.planner.plan(self.pos, self.goal, reservations) or []
                        stuck_ticks = 0

            time.sleep(0.4)


if __name__ == "__main__":
    r_id = sys.argv[1]
    ix, iy = int(sys.argv[2]), int(sys.argv[3])
    gx, gy = int(sys.argv[4]), int(sys.argv[5])
    port = int(sys.argv[6])
    peers = sys.argv[7].split(",") if len(sys.argv) > 7 and sys.argv[7] else []

    server_addr = os.getenv("WAREHOUSE_ADDR", "tcp://warehouse_sim:5555")

    node = AMRNode(
        robot_id=r_id,
        init_x=ix,
        init_y=iy,
        goal_x=gx,
        goal_y=gy,
        warehouse_addr=server_addr,
        pub_port=port,
        peer_addrs=peers,
    )
    node.run()
