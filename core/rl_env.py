"""
core/rl_env.py — Multi-Agent Warehouse Reinforcement Learning Environment.

Provides a fast simulation environment for training 1 to 4 autonomous AMRs
to navigate simultaneously around shelves, dynamic obstacles, and each other
without hardcoded collision conditions.
"""

import math
import random
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from core.grid import WarehouseGrid
from core.layouts import LAYOUT_NAMES, get_layout, get_random_layout

# Action mapping: (dx, dy)
ACTIONS: List[Tuple[int, int]] = [
    (0, -1),  # 0: North
    (0, 1),   # 1: South
    (-1, 0),  # 2: West
    (1, 0),   # 3: East
    (0, 0),   # 4: Wait
]

LOCAL_RADIUS = 2  # 5x5 local perception window: [-2, -1, 0, 1, 2]
OBS_DIM = 2 + 1 + (2 * LOCAL_RADIUS + 1) ** 2  # (dx, dy, dist, 25 grid cells) = 28

DEFAULT_4_ROUTES = [
    ((2, 14), (27, 14)),  # AMR 1: West -> East along horizontal corridor
    ((14, 2), (14, 27)),  # AMR 2: North -> South along vertical corridor
    ((2, 27), (27, 2)),   # AMR 3: South-West -> North-East diagonal
    ((2, 2), (27, 27)),   # AMR 4: North-West -> South-East diagonal
]


class AMRWarehouseEnv:
    """
    Multi-Agent Warehouse RL Environment (1–4 AMRs).
    Observation per robot (size 28):
      - [0:2]  : Normalized relative goal vector (gx - x)/30, (gy - y)/30
      - [2]    : Normalized Manhattan distance to goal
      - [3:28] : Flattened 5x5 local sensor grid (1.0 = blocked/peer, 0.0 = free)
    """

    def __init__(
        self,
        grid: Optional[WarehouseGrid] = None,
        max_steps: int = 150,
        num_agents: int = 4,
    ) -> None:
        self.grid = grid if grid is not None else WarehouseGrid()
        self.max_steps = max_steps
        self.num_agents = max(1, min(4, num_agents))
        self.agent_ids = [str(i + 1) for i in range(self.num_agents)]
        self.step_count = 0
        self.layout_name = "Standard Dual-Corridor"

        # Multi-agent state tables
        self.positions: Dict[str, Tuple[int, int]] = {}
        self.goals: Dict[str, Tuple[int, int]] = {}
        self.dones: Dict[str, bool] = {}

        # Initialize default routes
        for i, aid in enumerate(self.agent_ids):
            s, g = DEFAULT_4_ROUTES[i % len(DEFAULT_4_ROUTES)]
            self.positions[aid] = s
            self.goals[aid] = g
            self.dones[aid] = False

    # Backwards compatibility properties for single-agent access
    @property
    def pos(self) -> Tuple[int, int]:
        return self.positions.get("1", (2, 14))

    @pos.setter
    def pos(self, val: Tuple[int, int]) -> None:
        self.positions["1"] = val

    @property
    def goal(self) -> Tuple[int, int]:
        return self.goals.get("1", (27, 14))

    @goal.setter
    def goal(self, val: Tuple[int, int]) -> None:
        self.goals["1"] = val

    def set_layout(self, layout_name_or_index) -> str:
        """Switch active layout topology."""
        name, obstacles = get_layout(layout_name_or_index)
        self.layout_name = name
        self.grid.load_layout(obstacles)
        return name

    def get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Construct the 28-dimensional state vector for a specific agent."""
        x, y = self.positions[agent_id]
        gx, gy = self.goals[agent_id]

        dx = (gx - x) / 30.0
        dy = (gy - y) / 30.0
        manhattan_dist = (abs(gx - x) + abs(gy - y)) / 60.0

        # Peer positions: all other active agents
        peer_positions = {p for aid, p in self.positions.items() if aid != agent_id}

        local_sensor = []
        for oy in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            for ox in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                if ox == 0 and oy == 0:
                    local_sensor.append(0.0)  # Self position
                    continue
                cx, cy = x + ox, y + oy
                # Blocked if out of bounds, static shelf, dynamic wall, or peer AMR
                if not self.grid.is_valid(cx, cy) or (cx, cy) in peer_positions:
                    local_sensor.append(1.0)
                else:
                    local_sensor.append(0.0)

        return np.array([dx, dy, manhattan_dist] + local_sensor, dtype=np.float32)

    def get_observation(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> np.ndarray:
        """Generic observation constructor from position and goal."""
        x, y = pos
        gx, gy = goal
        dx = (gx - x) / 30.0
        dy = (gy - y) / 30.0
        manhattan_dist = (abs(gx - x) + abs(gy - y)) / 60.0

        local_sensor = []
        for oy in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            for ox in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                if ox == 0 and oy == 0:
                    local_sensor.append(0.0)
                    continue
                cx, cy = x + ox, y + oy
                if not self.grid.is_valid(cx, cy):
                    local_sensor.append(1.0)
                else:
                    local_sensor.append(0.0)

        return np.array([dx, dy, manhattan_dist] + local_sensor, dtype=np.float32)

    def reset(
        self,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        starts: Optional[Dict[str, Tuple[int, int]]] = None,
        goals: Optional[Dict[str, Tuple[int, int]]] = None,
        randomize_obstacles: bool = True,
        randomize_routes: bool = True,
        layout: Optional[Union[str, int]] = None,
        randomize_layout: bool = False,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Reset environment state for a new multi-agent training episode."""
        self.step_count = 0

        # Select layout
        if layout is not None:
            self.set_layout(layout)
        elif randomize_layout:
            name, obstacles = get_random_layout()
            self.layout_name = name
            self.grid.load_layout(obstacles)

        self.grid.clear_dynamic()

        # Randomize dynamic obstacle walls
        if randomize_obstacles:
            self._spawn_random_training_obstacles()

        # Setup positions and goals for all agents
        used_starts: Set[Tuple[int, int]] = set()
        used_goals: Set[Tuple[int, int]] = set()

        for i, aid in enumerate(self.agent_ids):
            self.dones[aid] = False

            # 1. Determine Starting Position
            if starts and aid in starts and self.grid.is_valid(*starts[aid]):
                s = starts[aid]
            elif start and aid == "1" and self.grid.is_valid(*start):
                s = start
            elif randomize_routes:
                s = self._sample_free_cell(used_starts)
            else:
                def_s, _ = DEFAULT_4_ROUTES[i % len(DEFAULT_4_ROUTES)]
                if self.grid.is_valid(*def_s) and def_s not in used_starts:
                    s = def_s
                else:
                    s = self._sample_free_cell(used_starts)

            used_starts.add(s)
            self.positions[aid] = s

            # 2. Determine Goal Position (separated by at least 8 cells for realistic traversal)
            if goals and aid in goals and self.grid.is_valid(*goals[aid]):
                g = goals[aid]
            elif goal and aid == "1" and self.grid.is_valid(*goal):
                g = goal
            elif randomize_routes:
                g = self._sample_free_goal(start=s, exclude=used_goals | {s}, min_dist=8)
            else:
                _, def_g = DEFAULT_4_ROUTES[i % len(DEFAULT_4_ROUTES)]
                if self.grid.is_valid(*def_g) and def_g not in used_goals and def_g != s:
                    g = def_g
                else:
                    g = self._sample_free_goal(start=s, exclude=used_goals | {s}, min_dist=8)

            used_goals.add(g)
            self.goals[aid] = g

        obs_dict = {aid: self.get_agent_observation(aid) for aid in self.agent_ids}
        if self.num_agents == 1 and start is not None:
            return obs_dict["1"]
        return obs_dict

    def _spawn_random_training_obstacles(self) -> None:
        """Inject 1-2 random 3-cell obstacle walls across open lanes."""
        num_obstacles = random.randint(1, 2)
        for _ in range(num_obstacles):
            is_horizontal = random.choice([True, False])
            ox = random.randint(6, 23)
            oy = random.randint(6, 23)
            for d in range(3):
                cx = ox + (d if is_horizontal else 0)
                cy = oy + (0 if is_horizontal else d)
                if (cx, cy) not in self.grid.static_obstacles:
                    self.grid.add_obstacle(cx, cy)

    def _sample_free_cell(self, exclude: Optional[Set[Tuple[int, int]]] = None) -> Tuple[int, int]:
        exclude_set = exclude if exclude is not None else set()
        for _ in range(800):
            x = random.randint(1, 28)
            y = random.randint(1, 28)
            if self.grid.is_valid(x, y) and (x, y) not in exclude_set:
                return (x, y)
        return (1, 1)

    def _sample_free_goal(
        self,
        start: Tuple[int, int],
        exclude: Optional[Set[Tuple[int, int]]] = None,
        min_dist: int = 8,
    ) -> Tuple[int, int]:
        """Sample a valid free goal cell with a minimum distance from start."""
        exclude_set = exclude if exclude is not None else set()
        for _ in range(800):
            x = random.randint(1, 28)
            y = random.randint(1, 28)
            if self.grid.is_valid(x, y) and (x, y) not in exclude_set:
                dist = abs(x - start[0]) + abs(y - start[1])
                if dist >= min_dist:
                    return (x, y)
        return self._sample_free_cell(exclude_set | {start})

    def step(
        self, actions: Union[Dict[str, int], int]
    ) -> Union[
        Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]],
        Tuple[np.ndarray, float, bool, dict],
    ]:
        """
        Execute simultaneous actions for all active agents.
        Handles static obstacle collisions, vertex conflicts, and swap conflicts.
        """
        self.step_count += 1
        is_single_call = isinstance(actions, (int, np.integer))

        if is_single_call:
            actions_dict = {"1": int(actions)}
        else:
            actions_dict = actions

        prev_positions = dict(self.positions)
        proposed_positions: Dict[str, Tuple[int, int]] = {}

        # 1. Propose movements
        for aid in self.agent_ids:
            if self.dones.get(aid, False):
                proposed_positions[aid] = self.positions[aid]
                continue
            act = actions_dict.get(aid, 4)  # default wait
            dx, dy = ACTIONS[act]
            proposed_positions[aid] = (self.positions[aid][0] + dx, self.positions[aid][1] + dy)

        # 2. Collision arbitration
        collisions: Dict[str, bool] = {aid: False for aid in self.agent_ids}
        peer_collisions: Dict[str, bool] = {aid: False for aid in self.agent_ids}

        # Check static / dynamic grid obstacles
        for aid in self.agent_ids:
            if not self.dones[aid]:
                nx, ny = proposed_positions[aid]
                if not self.grid.is_valid(nx, ny):
                    collisions[aid] = True

        # Check Vertex Collisions (two agents attempting to enter the exact same cell)
        dest_counts: Dict[Tuple[int, int], List[str]] = {}
        for aid, dest in proposed_positions.items():
            dest_counts.setdefault(dest, []).append(aid)

        for dest, claimants in dest_counts.items():
            if len(claimants) > 1:
                # Conflict: all colliding claimants fail to move
                for aid in claimants:
                    if not self.dones[aid]:
                        collisions[aid] = True
                        peer_collisions[aid] = True

        # Check Swap Collisions (agent A moves to B while agent B moves to A)
        for i, aid1 in enumerate(self.agent_ids):
            for aid2 in self.agent_ids[i + 1:]:
                if not self.dones[aid1] and not self.dones[aid2]:
                    if (
                        proposed_positions[aid1] == prev_positions[aid2]
                        and proposed_positions[aid2] == prev_positions[aid1]
                    ):
                        collisions[aid1] = True
                        collisions[aid2] = True
                        peer_collisions[aid1] = True
                        peer_collisions[aid2] = True

        # 3. Apply moves & calculate rewards
        rewards: Dict[str, float] = {}
        for aid in self.agent_ids:
            if self.dones[aid]:
                rewards[aid] = 0.0
                continue

            prev_pos = prev_positions[aid]
            goal = self.goals[aid]
            prev_dist = abs(prev_pos[0] - goal[0]) + abs(prev_pos[1] - goal[1])

            if collisions[aid]:
                # Failed move: agent stays in previous cell
                self.positions[aid] = prev_pos
                penalty = -30.0 if peer_collisions[aid] else -25.0
                rewards[aid] = penalty
            else:
                new_pos = proposed_positions[aid]
                self.positions[aid] = new_pos
                curr_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
                reward = -0.4  # Step cost
                reward += 2.5 * (prev_dist - curr_dist)
                rewards[aid] = reward

            # Goal check
            if self.positions[aid] == self.goals[aid]:
                rewards[aid] += 100.0
                self.dones[aid] = True

        # Horizon timeout
        timed_out = self.step_count >= self.max_steps
        if timed_out:
            for aid in self.agent_ids:
                self.dones[aid] = True

        obs_dict = {aid: self.get_agent_observation(aid) for aid in self.agent_ids}
        infos_dict = {
            aid: {
                "collision": collisions[aid],
                "peer_collision": peer_collisions[aid],
                "success": self.positions[aid] == self.goals[aid],
                "step": self.step_count,
            }
            for aid in self.agent_ids
        }

        if is_single_call:
            return obs_dict["1"], rewards["1"], self.dones["1"], infos_dict["1"]

        return obs_dict, rewards, self.dones, infos_dict
