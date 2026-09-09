"""
marl/environment.py
Multi-Agent Warehouse Simulation Environment for MARL / MAPPO.
Supports Domain Randomization, Dynamic Obstacle Injections, and Curriculum Fleet Sizing (2 to 5 AMRs).
"""

import random
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from core.grid import WarehouseGrid
from core.planner import SpaceTimeAStar


class MultiAgentWarehouseEnv:
    """
    Multi-Agent Warehouse Environment supporting CTDE and Domain Randomization.
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
    MAX_FLEET_SIZE = 5

    def __init__(
        self,
        num_agents: int = 3,
        max_steps: int = 150,
        grid_width: int = 30,
        grid_height: int = 30,
        randomize_tasks: bool = True,
        dynamic_obstacle_prob: float = 0.08,
        max_dynamic_obstacles: int = 3,
    ):
        self.num_agents = min(max(2, num_agents), self.MAX_FLEET_SIZE)
        self.max_steps = max_steps
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.randomize_tasks = randomize_tasks
        self.dynamic_obstacle_prob = dynamic_obstacle_prob
        self.max_dynamic_obstacles = max_dynamic_obstacles

        self.grid = WarehouseGrid()
        self.planner = SpaceTimeAStar(
            width=grid_width,
            height=grid_height,
            static_obstacles=self.grid.static_obstacles,
        )

        self.agent_ids = [f"amr_{i+1}" for i in range(self.num_agents)]
        self.default_spawns = [
            ((1, 14), (28, 14)),    # amr_1
            ((14, 1), (14, 28)),    # amr_2
            ((28, 14), (1, 14)),    # amr_3
            ((1, 1), (28, 28)),     # amr_4
            ((28, 1), (1, 28)),     # amr_5
        ]

        self.open_cells: List[Tuple[int, int]] = [
            (x, y)
            for x in range(1, self.grid_width - 1)
            for y in range(1, self.grid_height - 1)
            if self.grid.is_valid(x, y)
        ]

        self.positions: Dict[str, Tuple[int, int]] = {}
        self.goals: Dict[str, Tuple[int, int]] = {}
        self.planned_paths: Dict[str, List[Tuple[int, int]]] = {}
        self.has_payload: Dict[str, bool] = {}
        self.active_dynamic_obstacles: Dict[Tuple[int, int], int] = {}  # cell -> TTL
        self.step_count = 0

        # Fixed dimensions across curriculum: 5x5 window (25) + self (4) + top-4 peers (16) = 45
        self.obs_dim = 25 + 4 + (4 * 4)  # 45 dims
        self.action_dim = 5
        # Global state padded up to MAX_FLEET_SIZE for centralized critic CTDE
        self.state_dim = self.obs_dim * self.MAX_FLEET_SIZE  # 225 dims

    def set_num_agents(self, num_agents: int):
        """Allows dynamically resizing the active fleet for curriculum learning."""
        self.num_agents = min(max(2, num_agents), self.MAX_FLEET_SIZE)
        self.agent_ids = [f"amr_{i+1}" for i in range(self.num_agents)]

    def _sample_random_task(self, min_dist: int = 12) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        occupied = set(self.positions.values())
        valid_starts = [c for c in self.open_cells if c not in occupied]
        start = random.choice(valid_starts) if valid_starts else random.choice(self.open_cells)

        candidates = [c for c in self.open_cells if (abs(c[0] - start[0]) + abs(c[1] - start[1])) >= min_dist]
        goal = random.choice(candidates) if candidates else random.choice(self.open_cells)
        return start, goal

    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        self.step_count = 0
        self.positions.clear()
        self.goals.clear()
        self.planned_paths.clear()
        self.has_payload.clear()

        # Clear dynamic obstacles
        for obs_cell in list(self.active_dynamic_obstacles.keys()):
            self.grid.remove_obstacle(obs_cell[0], obs_cell[1])
        self.active_dynamic_obstacles.clear()

        for idx, agent_id in enumerate(self.agent_ids):
            if self.randomize_tasks:
                start, goal = self._sample_random_task(min_dist=10)
            else:
                spawn_idx = idx % len(self.default_spawns)
                start, goal = self.default_spawns[spawn_idx]

            self.positions[agent_id] = start
            self.goals[agent_id] = goal
            self.has_payload[agent_id] = True
            path = self.planner.plan(start, goal) or [start]
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
                    patch[dy_idx, dx_idx] = 1.0  # Wall/out of bounds
                elif coord in self.grid.static_obstacles:
                    patch[dy_idx, dx_idx] = 1.0  # Static shelf/wall
                elif coord in self.grid.all_obstacles:
                    patch[dy_idx, dx_idx] = 2.0 / 3.0  # Dynamic obstacle
                elif coord in other_robot_positions:
                    patch[dy_idx, dx_idx] = 1.0  # Peer robot occupied
                else:
                    patch[dy_idx, dx_idx] = 0.0  # Free floor

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

        # 3. Peer states: top-4 nearest peers (4 dims each = 16 dims)
        peer_features = []
        my_dist = abs(gx - cx) + abs(gy - cy)
        other_agents = [p_id for p_id in self.agent_ids if p_id != agent_id]

        # Sort peers by distance to self
        other_agents.sort(
            key=lambda p: abs(self.positions[p][0] - cx) + abs(self.positions[p][1] - cy)
        )

        for peer_id in other_agents[:4]:
            px, py = self.positions[peer_id]
            pgx, pgy = self.goals[peer_id]
            p_dist_to_goal = abs(pgx - px) + abs(pgy - py)

            p_dx = (px - cx) / float(self.grid_width)
            p_dy = (py - cy) / float(self.grid_height)
            p_norm_dist = p_dist_to_goal / float(self.grid_width + self.grid_height)
            has_priority = 1.0 if (my_dist < p_dist_to_goal or (my_dist == p_dist_to_goal and agent_id < peer_id)) else 0.0
            peer_features.extend([p_dx, p_dy, p_norm_dist, has_priority])

        # Zero-pad if fewer than 4 peers exist
        while len(peer_features) < 16:
            peer_features.extend([0.0, 0.0, 1.0, 0.0])

        return np.concatenate([patch, self_state, np.array(peer_features, dtype=np.float32)])

    def _get_all_observations(self) -> Dict[str, np.ndarray]:
        return {agent_id: self._get_observation(agent_id) for agent_id in self.agent_ids}

    def _get_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        # Concatenate active agent observations and zero-pad up to MAX_FLEET_SIZE (225 dims)
        active_obs = [observations[agent_id] for agent_id in self.agent_ids]
        while len(active_obs) < self.MAX_FLEET_SIZE:
            active_obs.append(np.zeros(self.obs_dim, dtype=np.float32))
        return np.concatenate(active_obs)

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

        # 1. Dynamic Obstacle Injection & TTL Management
        if self.dynamic_obstacle_prob > 0:
            expired = [cell for cell, ttl in self.active_dynamic_obstacles.items() if ttl <= 1]
            for cell in expired:
                del self.active_dynamic_obstacles[cell]
                self.grid.remove_obstacle(cell[0], cell[1])
            for cell in self.active_dynamic_obstacles:
                self.active_dynamic_obstacles[cell] -= 1

            if len(self.active_dynamic_obstacles) < self.max_dynamic_obstacles and random.random() < self.dynamic_obstacle_prob:
                occupied_now = set(self.positions.values())
                free_cells = [c for c in self.open_cells if c not in occupied_now and c not in self.active_dynamic_obstacles]
                if free_cells:
                    new_cell = random.choice(free_cells)
                    ttl = random.randint(15, 30)
                    self.active_dynamic_obstacles[new_cell] = ttl
                    self.grid.add_obstacle(new_cell[0], new_cell[1])

                    # Trigger dynamic replanning for any AMR whose path is blocked
                    for a_id in self.agent_ids:
                        if new_cell in self.planned_paths.get(a_id, []):
                            re_path = self.planner.plan(self.positions[a_id], self.goals[a_id]) or []
                            if re_path and re_path[0] == self.positions[a_id]:
                                re_path.pop(0)
                            self.planned_paths[a_id] = re_path

        # 2. Process Actions
        for agent_id, action in actions.items():
            cx, cy = self.positions[agent_id]
            path = self.planned_paths.get(agent_id, [])

            if action == self.ACTION_ADVANCE:
                intended_positions[agent_id] = path[0] if path else (cx, cy)
            elif action == self.ACTION_YIELD or action == self.ACTION_HALT:
                intended_positions[agent_id] = (cx, cy)
            elif action == self.ACTION_SIDESTEP:
                side = self._find_sidestep_cell(agent_id)
                intended_positions[agent_id] = side if side else (cx, cy)
            elif action == self.ACTION_CREEP:
                if path and self.grid.is_valid(path[0][0], path[0][1]):
                    intended_positions[agent_id] = path[0]
                else:
                    intended_positions[agent_id] = (cx, cy)
            else:
                intended_positions[agent_id] = (cx, cy)

        # 3. Collision Detection
        collisions = set()
        occupied_counts: Dict[Tuple[int, int], List[str]] = {}
        for agent_id, target in intended_positions.items():
            occupied_counts.setdefault(target, []).append(agent_id)

        # Vertex collisions
        for target, agents in occupied_counts.items():
            if len(agents) > 1:
                for a in agents:
                    collisions.add(a)

        # Edge-swap collisions
        for i, a1 in enumerate(self.agent_ids):
            for a2 in self.agent_ids[i+1:]:
                if (
                    intended_positions[a1] == self.positions[a2]
                    and intended_positions[a2] == self.positions[a1]
                    and self.positions[a1] != self.positions[a2]
                ):
                    collisions.add(a1)
                    collisions.add(a2)

        # Wall collisions
        for agent_id, target in intended_positions.items():
            if not self.grid.is_valid(target[0], target[1]):
                collisions.add(agent_id)

        # 4. Resolve Movements & Rewards
        for agent_id in self.agent_ids:
            old_pos = self.positions[agent_id]
            target = intended_positions[agent_id]
            goal = self.goals[agent_id]

            old_dist = abs(goal[0] - old_pos[0]) + abs(goal[1] - old_pos[1])

            if agent_id in collisions:
                rewards[agent_id] -= 30.0
            else:
                self.positions[agent_id] = target
                if self.planned_paths.get(agent_id) and self.planned_paths[agent_id][0] == target:
                    self.planned_paths[agent_id].pop(0)

                new_dist = abs(goal[0] - target[0]) + abs(goal[1] - target[1])
                rewards[agent_id] += float(old_dist - new_dist) * 1.5

                # Goal Reached: Package Delivered
                if target == goal:
                    rewards[agent_id] += 50.0
                    self.has_payload[agent_id] = not self.has_payload[agent_id]

                    if self.randomize_tasks:
                        _, new_goal = self._sample_random_task(min_dist=10)
                        self.goals[agent_id] = new_goal
                    else:
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
