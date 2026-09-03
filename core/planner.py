"""
core/planner.py — Space-Time A* path planner.

Plans collision-free trajectories through (x, y, t) state-space,
respecting static/dynamic grid obstacles AND peer robot reservations.

Collision types handled
-----------------------
1. Static/dynamic obstacles   : (nx, ny) blocked on the grid
2. Vertex collision            : peer reserved (nx, ny) at time nt
3. Edge / swap collision       : peer moving (nx,ny)→(x,y) at the same
                                 timestep I'm moving (x,y)→(nx,ny)

Time encoding
-------------
All times `t` are *global ticks* — `int(time.time() * TICK_RATE)` —
so reservations from different robots share the same clock reference.
`start_t` is the caller's current global tick.
"""

from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple

from core.grid import WarehouseGrid

# (dx, dy) actions — N, S, E, W, Wait
ACTIONS: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

MOVE_COST = 1.0
WAIT_COST = 1.2   # slight penalty keeps robots moving forward
MAX_HORIZON = 150  # max timesteps before giving up


class SpaceTimePlanner:
    def __init__(self, grid: WarehouseGrid) -> None:
        self.grid = grid

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        peer_reservations: Dict[Tuple[int, int, int], str],
        start_t: int = 0,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find a minimum-cost path from `start` to `goal` beginning at
        global tick `start_t`.

        Parameters
        ----------
        start             : (x, y) current robot position
        goal              : (x, y) target position
        peer_reservations : {(x, y, global_tick): robot_id_str}
        start_t           : global tick at which the robot occupies `start`

        Returns
        -------
        List of (x, y) waypoints from start (inclusive) to goal (inclusive),
        or None if no path found within MAX_HORIZON steps.
        """
        if start == goal:
            return [start]

        # open_set entries: (f_cost, t, x, y)
        sx, sy = start
        h0 = abs(sx - goal[0]) + abs(sy - goal[1])
        open_set: List[Tuple[float, int, int, int]] = []
        heappush(open_set, (h0, start_t, sx, sy))

        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {(sx, sy, start_t): 0.0}

        while open_set:
            f, t, x, y = heappop(open_set)

            # Goal check
            if (x, y) == goal:
                return self._reconstruct(came_from, x, y, t, start)

            # Time horizon guard
            if t >= start_t + MAX_HORIZON:
                continue

            current_g = g_score.get((x, y, t), float("inf"))

            for dx, dy in ACTIONS:
                nx, ny = x + dx, y + dy
                nt = t + 1

                # ── 1. Grid bounds + static/dynamic obstacle ──────────────
                if not self.grid.is_valid(nx, ny):
                    continue

                # ── 2. Vertex collision ────────────────────────────────────
                # Another robot has reserved (nx, ny) at nt.
                if (nx, ny, nt) in peer_reservations:
                    continue

                # ── 3. Edge / swap collision ───────────────────────────────
                # A peer is moving from (nx, ny) at t → (x, y) at nt.
                # This is a swap: we want (x,y)→(nx,ny) while peer does (nx,ny)→(x,y).
                # Both (nx,ny,t) and (x,y,nt) are reserved by the SAME peer.
                peer_at_nx_t  = peer_reservations.get((nx, ny, t))
                peer_at_x_nt  = peer_reservations.get((x,  y, nt))
                if (peer_at_nx_t is not None
                        and peer_at_nx_t == peer_at_x_nt):
                    continue

                # ── Cost update ────────────────────────────────────────────
                step_cost = MOVE_COST if (dx, dy) != (0, 0) else WAIT_COST
                tentative_g = current_g + step_cost

                if tentative_g < g_score.get((nx, ny, nt), float("inf")):
                    g_score[(nx, ny, nt)] = tentative_g
                    h = abs(nx - goal[0]) + abs(ny - goal[1])
                    came_from[(nx, ny, nt)] = (x, y, t)
                    heappush(open_set, (tentative_g + h, nt, nx, ny))

        return None  # No path found within horizon

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct(
        came_from: Dict,
        x: int,
        y: int,
        t: int,
        start: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        curr = (x, y, t)
        while curr in came_from:
            path.append((curr[0], curr[1]))
            curr = came_from[curr]
        path.append(start)
        path.reverse()
        return path

