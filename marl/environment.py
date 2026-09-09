"""
marl/environment.py
Multi-Agent Warehouse Simulation Environment for MARL / MAPPO.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from core.grid import WarehouseGrid
from core.planner import SpaceTimeAStar


class MultiAgentWarehouseEnv:
    """
    Multi-Agent Warehouse Environment supporting CTDE.
    Action Space per agent:
        0: ADVANCE  - Advance along Space-Time A* planned path
        1: YIELD    - Wait in place for 1 timestep
        2: SIDESTEP - Step sideways into an open adjacent cell to clear corridor
        3: HALT     - Stop moving
        4: CREEP    - Cautious advance: only step if next cell is free and no peer within 1 cell
    """

    ACTION_ADVANCE = 0
    ACTION_YIELD = 1
    ACTION_SIDESTEP = 2
    ACTION_HALT = 3
    ACTION_CREEP = 4

    ACTION_NAMES = ["ADVANCE", "YIELD", "SIDESTEP", "HALT", "CREEP"]

    def __init__(
        self,
        num_agents: int = 3,
        max_steps: int = 150,
        grid_width: int = 30,
        grid_height: int = 30,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.grid = WarehouseGrid()
        self.planner = SpaceTimeAStar(
            width=grid_width,
            height=grid_height,
            static_obstacles=self.grid.static_obstacles,
        )

        self.agent_ids = [f"amr_{i+1}" for i in range(num_agents)]
        self.default_spawns = [
            ((1, 14), (28, 14)),    # amr_1
            ((14, 1), (14, 28)),    # amr_2
            ((28, 14), (1, 14)),    # amr_3
        ]

        self.positions: Dict[str, Tuple[int, int]] = {}
        self.goals: Dict[str, Tuple[int, int]] = {}
        self.planned_paths: Dict[str, List[Tuple[int, int]]] = {}
        self.has_payload: Dict[str, bool] = {}
        self.step_count = 0

        # Dimension calculations
        # Local window 5x5 = 25, self state = 4, peer states (max 2 peers) = 8 -> 37
        self.obs_dim = 25 + 4 + (min(2, self.num_agents - 1) * 4)
        self.action_dim = 5
        self.state_dim = self.obs_dim * self.num_agents

    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        self.step_count = 0
        self.positions.clear()
        self.goals.clear()
        self.planned_paths.clear()
        self.has_payload.clear()

        for idx, agent_id in enumerate(self.agent_ids):
            spawn_idx = idx % len(self.default_spawns)
            start, goal = self.default_spawns[spawn_idx]
            self.positions[agent_id] = start
            self.goals[agent_id] = goal
            self.has_payload[agent_id] = True
            path = self.planner.plan(start, goal) or [start]
            # Remove start if it is the first element
            if path and path[0] == start:
                path = path[1:]
            self.planned_paths[agent_id] = path

        obs = self._get_all_observations()
        state = self._get_global_state(obs)
        return obs, state

    def _get_local_grid_patch(self, agent_id: str, patch_radius: int = 2) -> np.ndarray:
        cx, cy = self.positions[agent_id]
        patch = np.zeros((2 * patch_radius + 1, 2 * patch_radius + 1), dtype=np.float32)

        other_robot_positions = {pos for r_id, pos in self.positions.items() if r_id != agent_id}

        for dy_idx, dy in enumerate(range(-patch_radius, patch_radius + 1)):
            for dx_idx, dx in enumerate(range(-patch_radius, patch_radius + 1)):
                gx = cx + dx
                gy = cy + dy
                coord = (gx, gy)
                if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
                    patch[dy_idx, dx_idx] = 1.0  # Out of bounds / wall
                elif coord in self.grid.static_obstacles:
                    patch[dy_idx, dx_idx] = 1.0  # Static wall
                elif coord in self.grid.all_obstacles:
                    patch[dy_idx, dx_idx] = 2.0 / 3.0  # Dynamic obstacle
                elif coord in other_robot_positions:
                    patch[dy_idx, dx_idx] = 1.0  # Peer robot occupied
                else:
                    patch[dy_idx, dx_idx] = 0.0  # Open floor

        return patch.flatten()

    def _get_observation(self, agent_id: str) -> np.ndarray:
        cx, cy = self.positions[agent_id]
        gx, gy = self.goals[agent_id]

        # 1. Local 5x5 occupancy grid (25 dims)
        patch = self._get_local_grid_patch(agent_id, patch_radius=2)

        # 2. Self state (4 dims)
        dx_goal = (gx - cx) / float(self.grid_width)
        dy_goal = (gy - cy) / float(self.grid_height)
        dist_goal = (abs(gx - cx) + abs(gy - cy)) / float(self.grid_width + self.grid_height)
        payload = 1.0 if self.has_payload[agent_id] else 0.0
        self_state = np.array([dx_goal, dy_goal, dist_goal, payload], dtype=np.float32)

        # 3. Peer states (up to 2 nearest peers, 4 dims each = 8 dims)
        peer_features = []
        my_dist = abs(gx - cx) + abs(gy - cy)
        other_agents = [p_id for p_id in self.agent_ids if p_id != agent_id]

        # Sort peers by Manhattan distance to self
        other_agents.sort(
            key=lambda p: abs(self.positions[p][0] - cx) + abs(self.positions[p][1] - cy)
        )

        for peer_id in other_agents[:2]:
            px, py = self.positions[peer_id]
            pgx, pgy = self.goals[peer_id]
            p_dist_to_goal = abs(pgx - px) + abs(pgy - py)

            p_dx = (px - cx) / float(self.grid_width)
            p_dy = (py - cy) / float(self.grid_height)
            p_norm_dist = p_dist_to_goal / float(self.grid_width + self.grid_height)
            has_priority = 1.0 if (my_dist < p_dist_to_goal or (my_dist == p_dist_to_goal and agent_id < peer_id)) else 0.0
            peer_features.extend([p_dx, p_dy, p_norm_dist, has_priority])

        while len(peer_features) < (min(2, self.num_agents - 1) * 4):
            peer_features.extend([0.0, 0.0, 1.0, 0.0])

        return np.concatenate([patch, self_state, np.array(peer_features, dtype=np.float32)])

    def _get_all_observations(self) -> Dict[str, np.ndarray]:
        return {agent_id: self._get_observation(agent_id) for agent_id in self.agent_ids}

    def _get_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([observations[agent_id] for agent_id in self.agent_ids])

    def _find_sidestep_cell(self, agent_id: str) -> Optional[Tuple[int, int]]:
        cx, cy = self.positions[agent_id]
        candidates = [(cx, cy + 1), (cx, cy - 1), (cx - 1, cy), (cx + 1, cy)]
        occupied = set(self.positions.values())
        blocked = self.grid.all_obstacles

        for nx, ny in candidates:
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                if (nx, ny) not in blocked and (nx, ny) not in occupied:
                    return (nx, ny)
        return None

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        self.step_count += 1
        rewards = {agent_id: -0.1 for agent_id in self.agent_ids}  # Step penalty
        intended_positions: Dict[str, Tuple[int, int]] = {}

        for agent_id, action in actions.items():
            cx, cy = self.positions[agent_id]
            path = self.planned_paths.get(agent_id, [])

            if action == self.ACTION_ADVANCE:
                if path:
                    intended_positions[agent_id] = path[0]
                else:
                    intended_positions[agent_id] = (cx, cy)

            elif action == self.ACTION_YIELD or action == self.ACTION_HALT:
                intended_positions[agent_id] = (cx, cy)

            elif action == self.ACTION_SIDESTEP:
                side_cell = self._find_sidestep_cell(agent_id)
                intended_positions[agent_id] = side_cell if side_cell else (cx, cy)

            elif action == self.ACTION_CREEP:
                if path:
                    target = path[0]
                    # Only creep if cell free of walls
                    if self.grid.is_valid(target[0], target[1]):
                        intended_positions[agent_id] = target
                    else:
                        intended_positions[agent_id] = (cx, cy)
                else:
                    intended_positions[agent_id] = (cx, cy)
            else:
                intended_positions[agent_id] = (cx, cy)

        # Collision detection and resolution
        collisions = set()
        occupied_counts: Dict[Tuple[int, int], List[str]] = {}
        for agent_id, target in intended_positions.items():
            occupied_counts.setdefault(target, []).append(agent_id)

        # 1. Vertex collisions (two or more moving into same cell)
        for target, agents in occupied_counts.items():
            if len(agents) > 1:
                for a in agents:
                    collisions.add(a)

        # 2. Edge-swap collisions
        for i, a1 in enumerate(self.agent_ids):
            for a2 in self.agent_ids[i+1:]:
                if (
                    intended_positions[a1] == self.positions[a2]
                    and intended_positions[a2] == self.positions[a1]
                    and self.positions[a1] != self.positions[a2]
                ):
                    collisions.add(a1)
                    collisions.add(a2)

        # 3. Wall collisions
        for agent_id, target in intended_positions.items():
            if not self.grid.is_valid(target[0], target[1]):
                collisions.add(agent_id)

        # Update positions
        for agent_id in self.agent_ids:
            old_pos = self.positions[agent_id]
            target = intended_positions[agent_id]
            goal = self.goals[agent_id]

            old_dist = abs(goal[0] - old_pos[0]) + abs(goal[1] - old_pos[1])

            if agent_id in collisions:
                # Collision penalty
                rewards[agent_id] -= 30.0
                # Robot remains at previous position
            else:
                self.positions[agent_id] = target
                if self.planned_paths.get(agent_id) and self.planned_paths[agent_id][0] == target:
                    self.planned_paths[agent_id].pop(0)

                new_dist = abs(goal[0] - target[0]) + abs(goal[1] - target[1])
                # Progress reward
                rewards[agent_id] += float(old_dist - new_dist) * 1.5

                # Goal reached reward
                if target == goal:
                    rewards[agent_id] += 50.0
                    # Replanning back to spawn for continuous loop
                    spawn_idx = self.agent_ids.index(agent_id) % len(self.default_spawns)
                    start, orig_goal = self.default_spawns[spawn_idx]
                    self.goals[agent_id] = start if target == orig_goal else orig_goal
                    new_path = self.planner.plan(target, self.goals[agent_id]) or []
                    if new_path and new_path[0] == target:
                        new_path.pop(0)
                    self.planned_paths[agent_id] = new_path

        dones = {agent_id: (self.step_count >= self.max_steps) for agent_id in self.agent_ids}
        dones["__all__"] = (self.step_count >= self.max_steps)

        obs = self._get_all_observations()
        state = self._get_global_state(obs)
        infos = {
            agent_id: {
                "collision": agent_id in collisions,
                "pos": self.positions[agent_id],
                "goal": self.goals[agent_id],
            }
            for agent_id in self.agent_ids
        }

        return obs, state, rewards, dones, infos
