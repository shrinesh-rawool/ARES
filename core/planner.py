"""
core/planner.py — Space-Time A* Path Planner.
Plans paths across (x, y, time) avoiding static walls and dynamic peer reservations.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple, Union


class SpaceTimeAStar:

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        static_obstacles: frozenset = frozenset(),
    ):
        self.width = width
        self.height = height
        self.static_obstacles = static_obstacles

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        dynamic_reservations: Optional[Union[Dict[int, Set[Tuple[int, int]]], Dict[int, Tuple[int, int]]]] = None,
        edge_reservations: Optional[Set[Tuple[int, Tuple[int, int], Tuple[int, int]]]] = None,
        max_time: int = 120,
        hold_at_goal: int = 2,
    ) -> Optional[List[Tuple[int, int]]]:
        vertex_reservations: Dict[int, Set[Tuple[int, int]]] = {}
        if dynamic_reservations:
            for t, val in dynamic_reservations.items():
                if isinstance(val, (set, frozenset, list)):
                    vertex_reservations[t] = set(val)
                elif isinstance(val, tuple):
                    vertex_reservations[t] = {val}

        edge_res: Set[Tuple[int, Tuple[int, int], Tuple[int, int]]] = edge_reservations if edge_reservations else set()

        open_set: List[Tuple[int, int, Tuple[int, int, int]]] = []
        start_state = (start[0], start[1], 0)
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start_state))

        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], int] = {start_state: 0}
        closed_set: Set[Tuple[int, int, int]] = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            cx, cy, ct = current

            if (cx, cy) == goal:
                # Check if holding at goal is safe from peer reservations
                goal_conflict = False
                for ht in range(1, hold_at_goal + 1):
                    if goal in vertex_reservations.get(ct + ht, set()):
                        goal_conflict = True
                        break

                if not goal_conflict:
                    path = []
                    curr = current
                    while curr in came_from:
                        path.append((curr[0], curr[1]))
                        curr = came_from[curr]
                    path.append(start)
                    path.reverse()
                    return path

            if ct >= max_time:
                continue

            closed_set.add(current)

            neighbors = [
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1),
                (cx, cy),  # Wait in place
            ]

            nt = ct + 1
            for nx, ny in neighbors:
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in self.static_obstacles:
                    continue

                # Dynamic vertex collision (any peer at (nx, ny) at nt)
                if (nx, ny) in vertex_reservations.get(nt, set()):
                    continue

                # Dynamic edge swap collision (peer moving from (nx, ny) to (cx, cy) at time ct)
                if (ct, (nx, ny), (cx, cy)) in edge_res:
                    continue

                next_state = (nx, ny, nt)
                if next_state in closed_set:
                    continue

                tentative_g = current_g + 1
                if tentative_g < g_score.get(next_state, float("inf")):
                    came_from[next_state] = current
                    g_score[next_state] = tentative_g
                    f_score = tentative_g + self._heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, next_state))

        return None
