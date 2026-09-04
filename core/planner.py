"""
core/planner.py — Space-Time A* Path Planner.
Plans paths across (x, y, time) avoiding static walls and dynamic peer reservations.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple


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
        dynamic_reservations: Optional[Dict[int, Tuple[int, int]]] = None,
        max_time: int = 120,
    ) -> Optional[List[Tuple[int, int]]]:
        if dynamic_reservations is None:
            dynamic_reservations = {}

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

                # Dynamic vertex collision
                if dynamic_reservations.get(nt) == (nx, ny):
                    continue

                # Dynamic edge swap collision
                if dynamic_reservations.get(ct) == (nx, ny) and dynamic_reservations.get(nt) == (cx, cy):
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
