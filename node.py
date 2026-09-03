"""
node.py — Autonomous Mobile Robot (AMR) edge node.

Each node is an independent OS process. Three run concurrently.

Architecture
------------
* Uses SpaceTimePlanner (Space-Time A*) for collision-free path planning.
* Publishes a ZeroMQ telemetry packet every TICK_INTERVAL seconds.
* Subscribes to peer telemetry to maintain a live peer_reservations table.
* Subscribes to the visualizer's control channel for dynamic events.
* Implements Contract Net Protocol (CNP) self-healing when a peer dies.

3-Layer Collision Guarantee
----------------------------
Layer 1 — Priority-ranked planning
    Each robot only routes around peers with STRICTLY HIGHER priority.
    Higher-priority robots keep their optimal path; lower-priority ones
    plan around them. Priority = urgency×50 + 100/(dist+1), tiebroken
    by robot ID (lower ID wins). This prevents oscillation where both
    robots keep replanning around each other indefinitely.

Layer 2 — Continuous path monitoring
    Every tick, the robot checks whether any upcoming waypoint conflicts
    with a higher-priority peer's current reservations. If blocked, it
    replans immediately using the updated peer tables.

Layer 3 — Per-step execution safety gate
    Immediately before physically moving to the next cell, the robot
    checks ALL peers (regardless of priority) for vertex, position,
    and swap conflicts. If any conflict is detected the robot WAITS in
    place for one tick and replans — guaranteeing no two robots ever
    occupy the same cell simultaneously.

Addressing
----------
  AMR i → binds tcp://*:{5550+i} as PUB
  All peers → SUBscribe to each other's ports
  Control  → SUBscribes to tcp://localhost:5560 (visualizer PUB)
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import zmq

from core.grid import WarehouseGrid
from core.planner import SpaceTimePlanner
from core.rl_agent import DQNAgent
from core.rl_env import ACTIONS as RL_ACTIONS, LOCAL_RADIUS, OBS_DIM
from core.protocol import (
    HEARTBEAT_TIMEOUT,
    RESERVATION_HORIZON,
    TICK_INTERVAL,
    build_telemetry,
    calc_priority,
    cnp_winner,
    detect_failure,
    global_tick,
    negotiate_choke_point,
    telemetry_port,
    CONTROL_PORT,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_IDS: List[str] = ["1", "2", "3", "4"]


# ─────────────────────────────────────────────────────────────────────
# AMRNode
# ─────────────────────────────────────────────────────────────────────

class AMRNode:
    """Fully autonomous robot node."""

    def __init__(
        self,
        robot_id: str,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        urgency: int = 1,
        policy: str = "rl",
    ) -> None:
        self.robot_id = str(robot_id)
        self.current_pos = start
        self.urgency = urgency
        self.status = "WAITING"   # becomes "ACTIVE" after START signal
        self.killed = False
        self._running = True
        self.started  = False     # holds position until visualizer broadcasts START
        self.policy   = policy

        # Grid + planner
        self.grid = WarehouseGrid()
        self.planner = SpaceTimePlanner(self.grid)

        # RL Policy Agent
        self.rl_agent: Optional[DQNAgent] = None
        if self.policy == "rl":
            self.rl_agent = DQNAgent(obs_dim=OBS_DIM, num_actions=5, device="cpu")
            model_file = os.path.join(ROOT, "models", "amr_rl_policy.pt")
            if self.rl_agent.load(model_file):
                print(f"[AMR {self.robot_id}] RL Policy loaded from {model_file}")
            else:
                print(f"[AMR {self.robot_id}] Model {model_file} not found, falling back to planner")
                self.policy = "planner"

        # Task queue: robots can adopt additional goals via CNP
        self.task_queue: List[Tuple[int, int]] = [goal]
        self.task_idx: int = 0

        # Peer state tables
        self.peer_reservations: Dict[Tuple[int, int, int], str] = {}
        self.peer_heartbeats:   Dict[str, float] = {}
        self.peer_positions:    Dict[str, Tuple[int, int]] = {}
        self.peer_goals:        Dict[str, Tuple[int, int]] = {}
        self.peer_priorities:   Dict[str, float] = {}
        self.dead_peers:        Set[str] = set()
        self.cnp_adopted:       Set[str] = set()   # peers whose goal I've adopted

        # Path
        self.planned_path: List[Tuple[int, int]] = []
        self.step_idx: int = 0
        if self.policy == "rl" and self.rl_agent:
            self.planned_path = self._rl_rollout(steps=8)
        else:
            self._replan()

        # ZMQ
        self._setup_zmq()

        # Signals
        signal.signal(signal.SIGTERM, self._on_shutdown)
        signal.signal(signal.SIGINT,  self._on_shutdown)

    # ── ZMQ ─────────────────────────────────────────────────────────

    def _setup_zmq(self) -> None:
        self.ctx = zmq.Context()

        # PUB — telemetry
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 10)
        self.pub.bind(f"tcp://*:{telemetry_port(self.robot_id)}")

        # SUB — peer telemetry
        self.peer_sub = self.ctx.socket(zmq.SUB)
        self.peer_sub.setsockopt(zmq.RCVHWM, 30)
        for pid in ALL_IDS:
            if pid != self.robot_id:
                self.peer_sub.connect(f"tcp://localhost:{telemetry_port(pid)}")
        self.peer_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # SUB — control (from visualizer)
        self.ctrl_sub = self.ctx.socket(zmq.SUB)
        self.ctrl_sub.connect(f"tcp://localhost:{CONTROL_PORT}")
        self.ctrl_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    def _close_zmq(self) -> None:
        for sock in (self.pub, self.peer_sub, self.ctrl_sub):
            sock.setsockopt(zmq.LINGER, 0)
            sock.close()
        self.ctx.term()

    # ── Shutdown ─────────────────────────────────────────────────────

    def _on_shutdown(self, signum=None, frame=None) -> None:
        self._running = False

    @property
    def current_goal(self) -> Tuple[int, int]:
        return self.task_queue[self.task_idx]

    # ── RL Policy Helpers ────────────────────────────────────────────

    def _get_rl_observation(self, at_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Construct the 28-dim observation vector from current perception."""
        x, y = at_pos if at_pos is not None else self.current_pos
        gx, gy = self.current_goal

        dx = (gx - x) / 30.0
        dy = (gy - y) / 30.0
        manhattan_dist = (abs(gx - x) + abs(gy - y)) / 60.0

        local_sensor = []
        peer_positions = set(self.peer_positions.values())

        for oy in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            for ox in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                if ox == 0 and oy == 0:
                    local_sensor.append(0.0)
                    continue
                cx, cy = x + ox, y + oy
                if not self.grid.is_valid(cx, cy) or (cx, cy) in peer_positions:
                    local_sensor.append(1.0)
                else:
                    local_sensor.append(0.0)

        return np.array([dx, dy, manhattan_dist] + local_sensor, dtype=np.float32)

    def _rl_rollout(self, steps: int = 8) -> List[Tuple[int, int]]:
        """Simulate a short rollout of the RL policy for telemetry path broadcast."""
        sim_pos = self.current_pos
        rollout: List[Tuple[int, int]] = []
        if not self.rl_agent:
            return rollout
        for _ in range(steps):
            if sim_pos == self.current_goal:
                break
            sim_obs = self._get_rl_observation(at_pos=sim_pos)
            act = self.rl_agent.select_action(sim_obs, eval_mode=True)
            dx, dy = RL_ACTIONS[act]
            if (dx, dy) == (0, 0):
                break
            sim_pos = (sim_pos[0] + dx, sim_pos[1] + dy)
            rollout.append(sim_pos)
        return rollout

    # ── Layer 1: Priority-ranked planning ────────────────────────────

    def _my_priority(self) -> float:
        """
        Compute this robot's current priority score.
        Tiebreaker: lower robot_id wins (is treated as higher priority).
        """
        dist = (abs(self.current_pos[0] - self.current_goal[0]) +
                abs(self.current_pos[1] - self.current_goal[1]))
        base = calc_priority(self.urgency, dist)
        # Subtract a tiny fraction of robot ID so lower IDs win ties
        return base - int(self.robot_id) * 0.001

    def _blocking_reservations(self) -> Dict[Tuple[int, int, int], str]:
        """
        Return the reservations this robot MUST avoid during path planning:
        1. Any active peer's current physical position is blocked at next tick (t_now + 1),
           because no robot can phase through another physical robot.
           If the peer has reached its goal, its position is blocked across the full horizon.
        2. All future reservations of peers with STRICTLY HIGHER priority.
        """
        my_pri = self._my_priority()
        t_now = global_tick()
        out: Dict[Tuple[int, int, int], str] = {}

        # 1. Immediate physical presence of active peers
        for pid, pos in self.peer_positions.items():
            if pid != self.robot_id and pid not in self.dead_peers:
                out[(pos[0], pos[1], t_now + 1)] = pid
                peer_goal = self.peer_goals.get(pid)
                if peer_goal and pos == peer_goal:
                    for offset in range(1, RESERVATION_HORIZON + 1):
                        out[(pos[0], pos[1], t_now + offset)] = pid

        # 2. Future reservations of strictly higher-priority peers
        for (x, y, t), pid in self.peer_reservations.items():
            if pid != self.robot_id and pid not in self.dead_peers:
                peer_pri = self.peer_priorities.get(pid, float("inf"))
                if peer_pri > my_pri:
                    out[(x, y, t)] = pid

        return out

    # ── Layer 1+2: Replan using filtered reservations ────────────────

    def _replan(self) -> bool:
        """
        Attempt to replan from current_pos to current_goal.
        Only avoids higher-priority peers and immediate physical peer positions.
        Returns True if a new path was found.
        """
        if self.current_pos == self.current_goal:
            self.planned_path = []
            self.step_idx = 0
            return False

        path = self.planner.plan(
            start=self.current_pos,
            goal=self.current_goal,
            peer_reservations=self._blocking_reservations(),
            start_t=global_tick(),
        )
        if path and len(path) > 1:
            # path[0] is current_pos; path[1:] are remaining waypoints
            self.planned_path = path[1:]
            self.step_idx = 0
            return True
        elif path and len(path) == 1:
            self.planned_path = []
            self.step_idx = 0
            return True

        self.planned_path = []
        self.step_idx = 0
        return False

    def _path_is_blocked(self) -> bool:
        """
        Layer 2: return True if any upcoming waypoint in our planned path
        conflicts with a higher-priority peer's latest reservations, an occupied
        cell, or a grid obstacle. Triggers an immediate replan.
        """
        if not self.planned_path:
            return False

        t0 = global_tick()
        blocking = self._blocking_reservations()
        for offset, (rx, ry) in enumerate(
            self.planned_path[:RESERVATION_HORIZON],
            start=1,
        ):
            if not self.grid.is_valid(rx, ry):
                return True
            if (rx, ry, t0 + offset) in blocking:
                return True
        return False

    # ── Layer 3: Per-step execution safety gate ──────────────────────

    def _next_step_is_safe(self) -> bool:
        """
        Hard safety check executed immediately before every physical move.
        Checks ALL peers (regardless of priority) for three conflict types:

        1. Position conflict: is any active peer physically at next_cell right now?
        2. Vertex conflict: did any higher-priority peer reserve next_cell at next_t?
        3. Swap conflict: is any peer moving next_cell -> curr_cell while we move curr_cell -> next_cell?

        If ANY check fails → robot waits in place this tick to let the other robot pass.
        """
        if not self.planned_path:
            return True

        next_cell = self.planned_path[0]
        curr_cell = self.current_pos
        curr_t = global_tick()
        next_t = curr_t + 1

        # 1. Position: any active peer physically at next_cell right now
        for pid, pos in self.peer_positions.items():
            if pid != self.robot_id and pid not in self.dead_peers:
                if pos == next_cell:
                    return False

        # 2. Vertex: higher-priority peer reserved next_cell at next_t
        peer_res = self.peer_reservations.get((next_cell[0], next_cell[1], next_t))
        if peer_res is not None and peer_res != self.robot_id:
            peer_pri = self.peer_priorities.get(peer_res, float("inf"))
            if peer_pri >= self._my_priority():
                return False

        # 3. Swap: peer moving next_cell -> curr_cell
        peer_at_nc_now = self.peer_reservations.get((next_cell[0], next_cell[1], curr_t))
        peer_at_cc_next = self.peer_reservations.get((curr_cell[0], curr_cell[1], next_t))
        if (peer_at_nc_now is not None
                and peer_at_nc_now == peer_at_cc_next
                and peer_at_nc_now != self.robot_id):
            return False

        return True



    def _drain_control(self) -> bool:
        """Read all control messages. Returns True if replanning needed."""
        needs_replan = False
        try:
            while True:
                raw = self.ctrl_sub.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(raw)
                ctype = data.get("type")

                if ctype == "START":
                    if not self.started:
                        self.started = True
                        self.status  = "ACTIVE"
                        print(f"[AMR {self.robot_id}] START received — moving!")

                elif ctype == "OBSTACLE":
                    x, y = data.get("x"), data.get("y")
                    if x is not None and y is not None:
                        self.grid.add_obstacle(int(x), int(y))
                        needs_replan = True
                        print(f"[AMR {self.robot_id}] Dynamic obstacle added at ({x},{y}). Replanning…")

                elif ctype == "CLEAR_OBSTACLES":
                    self.grid.clear_dynamic()
                    needs_replan = True
                    print(f"[AMR {self.robot_id}] All dynamic obstacles cleared. Replanning…")

                elif ctype == "KILL":
                    if str(data.get("target_id")) == self.robot_id:
                        self.killed = True
                        self.status = "DEAD"
                        print(f"[AMR {self.robot_id}] Kill signal received.")

        except zmq.Again:
            pass
        return needs_replan

    def _drain_peers(self) -> bool:
        """Read all peer telemetry. Returns True if peer table changed."""
        changed = False
        try:
            while True:
                raw = self.peer_sub.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(raw)
                pid = str(data.get("id"))
                if pid == self.robot_id:
                    continue

                # Update heartbeat + position metadata
                self.peer_heartbeats[pid]  = data.get("timestamp", time.time())
                self.peer_positions[pid]   = (int(data.get("x", 0)), int(data.get("y", 0)))
                g = data.get("goal", [0, 0])
                self.peer_goals[pid]       = (int(g[0]), int(g[1]))
                self.peer_priorities[pid]  = float(data.get("priority", 0))

                # Refresh peer reservations (remove stale, insert fresh)
                for key in [k for k, v in self.peer_reservations.items() if v == pid]:
                    del self.peer_reservations[key]
                for entry in data.get("reservations", []):
                    self.peer_reservations[(int(entry[0]), int(entry[1]), int(entry[2]))] = pid

                changed = True
        except zmq.Again:
            pass
        return changed

    # ── CNP self-healing ────────────────────────────────────────────

    def _check_peer_failures(self) -> bool:
        """
        Detect silent peers, mark them dead, and trigger CNP task adoption
        for the best-placed alive robot.
        Returns True if a new task was adopted (needs replanning).
        """
        adopted = False
        for pid in list(self.peer_heartbeats.keys()):
            if pid in self.dead_peers or pid in self.cnp_adopted:
                continue
            if not detect_failure(self.peer_heartbeats[pid]):
                continue

            print(f"[AMR {self.robot_id}] Peer {pid} declared DEAD. Running CNP auction…")
            self.dead_peers.add(pid)

            # Mark dead robot's last known position as a static obstacle
            if pid in self.peer_positions:
                px, py = self.peer_positions[pid]
                self.grid.add_obstacle(px, py)

            # Build CNP candidate list: (my_id, distance_to_peer_goal)
            peer_goal = self.peer_goals.get(pid)
            if peer_goal is None:
                continue

            my_dist = abs(self.current_pos[0] - peer_goal[0]) + \
                      abs(self.current_pos[1] - peer_goal[1])
            candidates = [(self.robot_id, float(my_dist))]

            for other_id in ALL_IDS:
                if (other_id == self.robot_id
                        or other_id == pid
                        or other_id in self.dead_peers):
                    continue
                if other_id in self.peer_positions:
                    ox, oy = self.peer_positions[other_id]
                    d = abs(ox - peer_goal[0]) + abs(oy - peer_goal[1])
                    candidates.append((other_id, float(d)))

            winner = cnp_winner(candidates)
            if winner == self.robot_id:
                print(f"[AMR {self.robot_id}] CNP winner — adopting AMR {pid}'s goal {peer_goal}.")
                self.task_queue.append(peer_goal)
                self.cnp_adopted.add(pid)
                adopted = True
            else:
                print(f"[AMR {self.robot_id}] CNP: AMR {winner} is closer, they will adopt.")

        return adopted

    # ── Broadcast ───────────────────────────────────────────────────

    def _broadcast(self) -> None:
        packet = build_telemetry(
            robot_id=self.robot_id,
            x=self.current_pos[0],
            y=self.current_pos[1],
            goal=self.current_goal,
            status=self.status,
            planned_path=self.planned_path,
            step_idx=self.step_idx,
            urgency=self.urgency,
        )
        try:
            self.pub.send_string(packet, flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        print(f"[AMR {self.robot_id}] Running. start={self.current_pos}, "
              f"goal={self.current_goal}, urgency={self.urgency}")

        while self._running:
            tick_start = time.time()

            # ── 1. Control channel ───────────────────────────────────
            ctrl_replan = self._drain_control()

            if self.killed:
                self.status = "DEAD"
                self._broadcast()
                time.sleep(TICK_INTERVAL)
                continue

            # ── 2. Peer telemetry ───────────────────────────────────
            self._drain_peers()

            # ── 3. CNP failure detection ────────────────────────────
            cnp_replan = self._check_peer_failures()

            # ── 4. Advance to next task if current one is done ───────
            if self.current_pos == self.current_goal:
                if self.task_idx + 1 < len(self.task_queue):
                    self.task_idx += 1
                    self.status = "ACTIVE"
                    print(f"[AMR {self.robot_id}] Task complete → next goal={self.current_goal}")
                    self._replan()
                else:
                    self.status = "GOAL_REACHED"

            # ── 5. Replan if needed (planner mode only) ─────────────
            if self.policy == "planner":
                if ctrl_replan or cnp_replan or self._path_is_blocked():
                    self._replan()

            # ── 6. Step forward (only after START signal) ────────────
            if self.started and self.status == "ACTIVE":
                if self.policy == "rl" and self.rl_agent is not None:
                    # ── RL Mode: Neural network policy selects action from sensor ──
                    obs = self._get_rl_observation()
                    action = self.rl_agent.select_action(obs, eval_mode=True)
                    dx, dy = RL_ACTIONS[action]
                    nx, ny = self.current_pos[0] + dx, self.current_pos[1] + dy

                    # Physical validity check: don't phase through boundaries/shelves/peers
                    peer_positions = set(self.peer_positions.values())
                    if self.grid.is_valid(nx, ny) and (nx, ny) not in peer_positions:
                        self.current_pos = (nx, ny)

                    # Update planned_path for visualizer and peer telemetry rollout
                    self.planned_path = self._rl_rollout(steps=8)
                else:
                    if self.planned_path:
                        if self._next_step_is_safe():
                            self.current_pos = self.planned_path.pop(0)
                            self.step_idx = 0
                        else:
                            # Wait in place this tick: yield to peer and replan
                            self._replan()

            # ── 7. Broadcast telemetry ───────────────────────────────
            self._broadcast()

            # ── Tick pacing ──────────────────────────────────────────
            elapsed = time.time() - tick_start
            time.sleep(max(0.0, TICK_INTERVAL - elapsed))

        # Graceful shutdown
        self._broadcast()   # final packet so peers can detect departure
        self._close_zmq()
        print(f"[AMR {self.robot_id}] Shutdown complete.")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AMR Edge Node")
    parser.add_argument("--id",      required=True,       help="Robot ID (1, 2, 3)")
    parser.add_argument("--start",   required=True,       help="Start position: x,y")
    parser.add_argument("--goal",    required=True,       help="Goal position: x,y")
    parser.add_argument("--urgency", type=int, default=1, help="Task urgency 1–3")
    parser.add_argument("--policy",  default="rl", choices=["rl", "planner"], help="Policy: rl or planner")
    args = parser.parse_args()

    start = tuple(map(int, args.start.split(",")))
    goal  = tuple(map(int, args.goal.split(",")))

    node = AMRNode(str(args.id), start, goal, args.urgency, policy=args.policy) # type: ignore
    time.sleep(0.5)   # Let ZMQ socket mesh settle
    node.run()


if __name__ == "__main__":
    main()