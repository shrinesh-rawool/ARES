"""
amr/amr_node.py
Autonomous edge node with Dynamic Aisle Obstruction Discovery & P2P Costmap Sharing.
"""

import os
import sys

# Ensure repository root is on sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import zmq
from core.planner import SpaceTimeAStar

try:
    from marl.policy import MAPPOPolicy
    HAS_MARL = True
except ImportError:
    HAS_MARL = False
    MAPPOPolicy = None


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
        use_marl: bool = True,
    ):
        self.robot_id = robot_id
        self.init_pos = (init_x, init_y)
        self.target_goal = (goal_x, goal_y)
        self.pos = (init_x, init_y)
        self.goal = (goal_x, goal_y)

        self.context = zmq.Context()

        # 1. Warehouse Sim Socket with timeout
        self.warehouse_sock = self.context.socket(zmq.REQ)
        self.warehouse_sock.setsockopt(zmq.RCVTIMEO, 3000)
        self.warehouse_sock.setsockopt(zmq.SNDTIMEO, 3000)
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
        self.blocked_cell_attempts: Dict[Tuple[int, int], int] = {}
        self.yield_hold_ticks: int = 0

        # MAPPO policy initialization
        self.policy = None
        if use_marl and HAS_MARL:
            model_candidate = "models/mappo_actor.pt" if os.path.exists("models/mappo_actor.pt") else "mappo_actor.pt"
            self.policy = MAPPOPolicy(model_path=model_candidate)
            print(f"[{self.robot_id}] MAPPO MARL policy loaded (has_weights={self.policy.has_weights})")

        self._register_with_warehouse(init_x, init_y)

        self.has_payload: bool = True  # Starts carrying a package to its destination

    def _register_with_warehouse(self, x: int, y: int):
        req = {
            "type": "REGISTER",
            "robot_id": self.robot_id,
            "x": x,
            "y": y,
            "goal": self.goal,
            "has_payload": True,
        }
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
        """Broadcast state, path, and discovered dynamic obstacles."""
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
            self.replan_path()

    def get_peer_reservations(self) -> Tuple[Dict[int, Set[Tuple[int, int]]], Set[Tuple[int, Tuple[int, int], Tuple[int, int]]]]:
        """Collect vertex and edge reservations for all active peers without overwriting."""
        vertex_res: Dict[int, Set[Tuple[int, int]]] = {}
        edge_res: Set[Tuple[int, Tuple[int, int], Tuple[int, int]]] = set()

        for p_id, p_info in self.peer_states.items():
            p_path = p_info.get("planned_path", [])
            for t, coord in enumerate(p_path):
                pt = tuple(coord)
                vertex_res.setdefault(t, set()).add(pt)
                if t > 0:
                    prev = tuple(p_path[t - 1])
                    edge_res.add((t - 1, prev, pt))

            cur = p_info.get("pos")
            if cur:
                cur_tup = tuple(cur)
                for t in range(4):
                    vertex_res.setdefault(t, set()).add(cur_tup)

        return vertex_res, edge_res

    def replan_path(self):
        """Helper to invoke Space-Time A* with full peer reservations."""
        if self.planner is None:
            self.rebuild_planner()
        v_res, e_res = self.get_peer_reservations()
        self.planned_path = self.planner.plan(
            self.pos,
            self.goal,
            dynamic_reservations=v_res,
            edge_reservations=e_res,
        ) or []

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

        # Avoid immediate path steps of any peer to prevent stepping into their path
        peer_path_cells = set()
        for p_info in self.peer_states.values():
            for step_cell in p_info.get("planned_path", [])[:3]:
                peer_path_cells.add(tuple(step_cell))

        for nx, ny in candidates:
            if not (0 <= nx < 30 and 0 <= ny < 30):
                continue
            if (nx, ny) in all_blocked or (nx, ny) in peer_occupied or (nx, ny) in peer_path_cells:
                continue
            return (nx, ny)
        return None

    def get_local_observation(self) -> np.ndarray:
        """Construct 37-dim local observation vector for MARL policy inference."""
        cx, cy = self.pos
        gx, gy = self.goal

        # 1. 5x5 occupancy grid (25 dims)
        patch = np.zeros((5, 5), dtype=np.float32)
        peer_occupied = {p_info["pos"] for p_info in self.peer_states.values() if "pos" in p_info}

        for dy_idx, dy in enumerate(range(-2, 3)):
            for dx_idx, dx in enumerate(range(-2, 3)):
                nx = cx + dx
                ny = cy + dy
                coord = (nx, ny)
                if not (0 <= nx < 30 and 0 <= ny < 30):
                    patch[dy_idx, dx_idx] = 1.0
                elif coord in self.static_obstacles:
                    patch[dy_idx, dx_idx] = 1.0
                elif coord in self.dynamic_obstacles:
                    patch[dy_idx, dx_idx] = 2.0 / 3.0
                elif coord in peer_occupied:
                    patch[dy_idx, dx_idx] = 1.0

        # 2. Self state (4 dims)
        dx_goal = (gx - cx) / 30.0
        dy_goal = (gy - cy) / 30.0
        dist_goal = (abs(gx - cx) + abs(gy - cy)) / 60.0
        payload = 1.0 if self.has_payload else 0.0
        self_state = np.array([dx_goal, dy_goal, dist_goal, payload], dtype=np.float32)

        # 3. Peer features (16 dims: up to 4 peers x 4 dims)
        peer_features = []
        my_dist = abs(gx - cx) + abs(gy - cy)
        sorted_peers = sorted(
            [p for p in self.peer_states.values() if "pos" in p],
            key=lambda p: abs(p["pos"][0] - cx) + abs(p["pos"][1] - cy)
        )

        for p_info in sorted_peers[:4]:
            px, py = p_info["pos"]
            p_dist_to_goal = p_info.get("dist_to_goal", 999)
            p_dx = (px - cx) / 30.0
            p_dy = (py - cy) / 30.0
            p_norm_dist = p_dist_to_goal / 60.0
            has_prio = 1.0 if self.has_higher_priority(p_info.get("robot_id", "peer"), p_dist_to_goal) else 0.0
            peer_features.extend([p_dx, p_dy, p_norm_dist, has_prio])

        while len(peer_features) < 16:
            peer_features.extend([0.0, 0.0, 1.0, 0.0])

        return np.concatenate([patch.flatten(), self_state, np.array(peer_features, dtype=np.float32)])

    def step(self, next_x: int, next_y: int) -> bool:
        req = {
            "type": "STEP",
            "robot_id": self.robot_id,
            "next_x": next_x,
            "next_y": next_y,
            "goal": self.goal,
            "has_payload": self.has_payload,
        }
        try:
            self.warehouse_sock.send_json(req)
            res = self.warehouse_sock.recv_json()
        except zmq.ZMQError as e:
            print(f"[{self.robot_id}] Warehouse socket error: {e}")
            return False

        if not res.get("collision", False):
            self.pos = (next_x, next_y)
            return True
        return False

    def run(self):
        print(f"[{self.robot_id}] Autonomous loop started. Target goal: {self.goal}")

        # Peer discovery synchronization phase to avoid "slow joiner" collision at tick 0
        print(f"[{self.robot_id}] Synchronizing with peer mesh...")
        for _ in range(5):
            self.broadcast_telemetry()
            self.update_peers()
            time.sleep(0.1)

        self.replan_path()
        stuck_ticks = 0

        while True:
            self.broadcast_telemetry()
            self.update_peers()

            # Continuous back-and-forth transit (generic ping-pong)
            if self.pos == self.goal:
                action_str = "Delivered package!" if self.has_payload else "Loaded package!"
                print(f"[{self.robot_id}] Reached {self.goal}. {action_str} Swapping in 2s...")

                self.has_payload = not self.has_payload
                time.sleep(2.0)

                # Toggle goal between initial location and target destination
                self.goal = self.init_pos if self.pos == self.target_goal else self.target_goal
                self.replan_path()
                continue

            # Drop current pos from planned path
            while self.planned_path and self.planned_path[0] == self.pos:
                self.planned_path.pop(0)

            if not self.planned_path:
                self.replan_path()
                if not self.planned_path:
                    time.sleep(0.3)
                    continue

            next_coord = self.planned_path[0]

            # Peer priority arbitration (MARL policy with heuristic fallback)
            yield_needed = False
            peer_conflict = any(
                abs(self.pos[0] - p_info["pos"][0]) + abs(self.pos[1] - p_info["pos"][1]) <= 2
                for p_info in self.peer_states.values() if "pos" in p_info
            )

            if peer_conflict and self.policy is not None:
                obs = self.get_local_observation()
                marl_action = self.policy.get_action(obs, deterministic=True)
                if marl_action == MAPPOPolicy.ACTION_YIELD or marl_action == MAPPOPolicy.ACTION_HALT:
                    yield_needed = True
                elif marl_action == MAPPOPolicy.ACTION_SIDESTEP:
                    sidestep_coord = self.find_sidestep_cell()
                    if sidestep_coord and self.step(sidestep_coord[0], sidestep_coord[1]):
                        print(f"[{self.robot_id}] MARL MAPPO: Sidestepped to {sidestep_coord} to clear passage.")
                        self.yield_hold_ticks = 3
                        self.replan_path()
                        time.sleep(0.4)
                        continue
                    else:
                        yield_needed = True
            elif peer_conflict:
                # Rule-based priority fallback
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
                if self.yield_hold_ticks > 0:
                    self.yield_hold_ticks -= 1
                    time.sleep(0.3)
                    continue

                stuck_ticks += 1
                if stuck_ticks >= 2:
                    sidestep_coord = self.find_sidestep_cell()
                    if sidestep_coord:
                        print(f"[{self.robot_id}] Sidestepping to {sidestep_coord} to clear corridor...")
                        if self.step(sidestep_coord[0], sidestep_coord[1]):
                            self.yield_hold_ticks = 3  # Hold position for 3 ticks to let higher-priority peer pass
                            stuck_ticks = 0
                            self.replan_path()
                            time.sleep(0.4)
                            continue
                time.sleep(0.3)
                continue
            else:
                self.yield_hold_ticks = 0

            # Execute step
            success = self.step(next_coord[0], next_coord[1])

            if success:
                self.planned_path.pop(0)
                stuck_ticks = 0
                self.blocked_cell_attempts.pop(next_coord, None)
            else:
                stuck_ticks += 1
                peer_in_cell = any(
                    p_info.get("pos") == next_coord or next_coord in [tuple(c) for c in p_info.get("planned_path", [])[:2]]
                    for p_info in self.peer_states.values()
                )

                if not peer_in_cell:
                    self.blocked_cell_attempts[next_coord] = self.blocked_cell_attempts.get(next_coord, 0) + 1
                    # Only confirm physical obstacle after multiple persistent failures to avoid false positives
                    if self.blocked_cell_attempts[next_coord] >= 3:
                        print(f"[{self.robot_id}] Detected physical obstacle at {next_coord} after 3 attempts! Broadcasting costmap delta...")
                        self.dynamic_obstacles.add(next_coord)
                        self.rebuild_planner()
                        self.replan_path()
                        self.blocked_cell_attempts.pop(next_coord, None)
                    else:
                        time.sleep(0.2)
                else:
                    self.blocked_cell_attempts.pop(next_coord, None)
                    time.sleep(0.2)
                    if stuck_ticks > 2:
                        self.replan_path()
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
