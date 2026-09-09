"""
core/grid.py — Warehouse grid model.
"""
from typing import Optional, Set, Tuple


class WarehouseGrid:
    WIDTH  = 30
    HEIGHT = 30

    @staticmethod
    def _generate_default_layout() -> frozenset:
        obs: Set[Tuple[int, int]] = set()

        # 1. Outer perimeter walls
        for i in range(30):
            obs.update({(i, 0), (i, 29), (0, i), (29, i)})

        # 2. Shelves leaving cross corridors at x=14, y=14
        for y in range(4, 12):
            for x in range(3, 7): obs.add((x, y))
            for x in range(8, 12): obs.add((x, y))
            for x in range(17, 21): obs.add((x, y))
            for x in range(22, 26): obs.add((x, y))

        for y in range(17, 25):
            for x in range(3, 7): obs.add((x, y))
            for x in range(8, 12): obs.add((x, y))
            for x in range(17, 21): obs.add((x, y))
            for x in range(22, 26): obs.add((x, y))

        # 3. 1-cell choke point boundary at (14, 14)
        for offset in range(1, 4):
            obs.update({
                (14 - offset, 14 - offset),
                (14 - offset, 14 + offset),
                (14 + offset, 14 - offset),
                (14 + offset, 14 + offset),
            })
        return frozenset(obs)

    STATIC_OBSTACLES: frozenset = _generate_default_layout()

    def __init__(self, static_obstacles: Optional[frozenset] = None):
        self._static: frozenset = static_obstacles if static_obstacles is not None else self.STATIC_OBSTACLES
        self._dynamic: Set[Tuple[int, int]] = set()

    def load_layout(self, obstacles: frozenset) -> None:
        self._static = obstacles
        self._dynamic.clear()

    def add_obstacle(self, x: int, y: int) -> None:
        self._dynamic.add((x, y))

    def remove_obstacle(self, x: int, y: int) -> None:
        self._dynamic.discard((x, y))

    def clear_dynamic(self) -> None:
        self._dynamic.clear()

    @property
    def static_obstacles(self) -> frozenset:
        return self._static

    @property
    def all_obstacles(self) -> frozenset:
        return self._static | frozenset(self._dynamic)

    def is_valid(self, x: int, y: int) -> bool:
        if not (0 <= x < self.WIDTH and 0 <= y < self.HEIGHT):
            return False
        return (x, y) not in self._static and (x, y) not in self._dynamic
