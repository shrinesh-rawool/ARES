"""
core/grid.py — Warehouse grid model.

30 × 30 warehouse map with two rows of shelves (6-cell blocks each side),
leaving clear horizontal and vertical corridors at y=14 and x=14
for the AMR demo crossing scenario.
"""


from typing import Optional


class WarehouseGrid:
    WIDTH  = 30
    HEIGHT = 30

    # Two shelf rows — scaled up for the 30×30 map.
    # Top row  at y=8 : left cluster x=4..9, right cluster x=18..23
    # Bottom row at y=21: same x layout
    STATIC_OBSTACLES: frozenset = frozenset({
        # ── Top shelf row (y = 8) ──────────────────────────────────
        (4, 8),  (5, 8),  (6, 8),  (7, 8),  (8, 8),  (9, 8),
        (18, 8), (19, 8), (20, 8), (21, 8), (22, 8), (23, 8),
        # ── Bottom shelf row (y = 21) ─────────────────────────────
        (4, 21), (5, 21), (6, 21), (7, 21), (8, 21), (9, 21),
        (18, 21),(19, 21),(20, 21),(21, 21),(22, 21),(23, 21),
    })

    def __init__(self, static_obstacles: Optional[frozenset] = None):
        self._static: frozenset = static_obstacles if static_obstacles is not None else self.STATIC_OBSTACLES
        self._dynamic: set = set()

    def load_layout(self, obstacles: frozenset) -> None:
        """Switch active layout static obstacles."""
        self._static = obstacles
        self._dynamic.clear()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_obstacle(self, x: int, y: int) -> None:
        """Add a dynamic (runtime) obstacle at (x, y)."""
        self._dynamic.add((x, y))

    def remove_obstacle(self, x: int, y: int) -> None:
        """Remove a dynamic obstacle (no-op if not present)."""
        self._dynamic.discard((x, y))

    def clear_dynamic(self) -> None:
        """Remove all dynamic obstacles."""
        self._dynamic.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def static_obstacles(self) -> frozenset:
        return self._static

    @property
    def all_obstacles(self) -> frozenset:
        return self._static | frozenset(self._dynamic)

    def is_valid(self, x: int, y: int) -> bool:
        """Return True iff (x, y) is inside the grid and free of obstacles."""
        if not (0 <= x < self.WIDTH and 0 <= y < self.HEIGHT):
            return False
        if (x, y) in self._static:
            return False
        if (x, y) in self._dynamic:
            return False
        return True
