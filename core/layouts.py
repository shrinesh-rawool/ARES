"""
core/layouts.py — Multi-Scenario Warehouse Layout Generator.

Provides 6 distinct warehouse topologies to train autonomous AMRs
on diverse spatial challenges (narrow aisles, chokepoints, pillars,
clutter, and asymmetric storage) for robust generalization.
"""

import random
from typing import Dict, List, Set, Tuple

WIDTH = 30
HEIGHT = 30


def _bounds_check(cells: Set[Tuple[int, int]]) -> frozenset:
    """Filter out any cells outside the 30x30 playable area."""
    return frozenset((x, y) for (x, y) in cells if 0 <= x < WIDTH and 0 <= y < HEIGHT)


# ── Scenario 1: Standard Dual-Corridor ──────────────────────────────
def layout_standard() -> frozenset:
    """Two shelf rows leaving open crossing corridors at x=14 and y=14."""
    obs: Set[Tuple[int, int]] = set()
    # Row y=8
    for x in list(range(4, 10)) + list(range(18, 24)):
        obs.add((x, 8))
    # Row y=21
    for x in list(range(4, 10)) + list(range(18, 24)):
        obs.add((x, 21))
    return _bounds_check(obs)


# ── Scenario 2: Dense Storage Aisles ────────────────────────────────
def layout_dense_aisles() -> frozenset:
    """Narrow vertical storage racks with 3 horizontal cross-aisles."""
    obs: Set[Tuple[int, int]] = set()
    racks_x = [6, 11, 18, 23]
    cross_aisles_y = {7, 15, 22}

    for rx in racks_x:
        for y in range(3, 27):
            if y not in cross_aisles_y:
                obs.add((rx, y))
                obs.add((rx + 1, y))  # double-depth rack
    return _bounds_check(obs)


# ── Scenario 3: Central Chokepoints / Tunnels ──────────────────────
def layout_chokepoints() -> frozenset:
    """Dividing central wall at x=14 with two narrow doorways at y=6 and y=23."""
    obs: Set[Tuple[int, int]] = set()
    doorways = {5, 6, 7, 22, 23, 24}
    for y in range(2, 28):
        if y not in doorways:
            obs.add((14, y))
            obs.add((15, y))

    # Small storage pods on east and west wings
    for px, py in [(6, 10), (6, 19), (23, 10), (23, 19)]:
        for dx in range(3):
            for dy in range(3):
                obs.add((px + dx, py + dy))
    return _bounds_check(obs)


# ── Scenario 4: Pillar / Column Field ───────────────────────────────
def layout_pillar_field() -> frozenset:
    """Regular matrix of 2x2 structural support pillars across the warehouse."""
    obs: Set[Tuple[int, int]] = set()
    for px in [6, 12, 18, 24]:
        for py in [6, 12, 18, 24]:
            obs.add((px, py))
            obs.add((px + 1, py))
            obs.add((px, py + 1))
            obs.add((px + 1, py + 1))
    return _bounds_check(obs)


# ── Scenario 5: Asymmetric Distribution Center ──────────────────────
def layout_asymmetric() -> frozenset:
    """L-shaped rack assemblies and staggered loading bays."""
    obs: Set[Tuple[int, int]] = set()
    # North-west L-rack
    for x in range(4, 13):
        obs.add((x, 6))
    for y in range(6, 15):
        obs.add((4, y))

    # South-east L-rack
    for x in range(17, 26):
        obs.add((x, 23))
    for y in range(15, 24):
        obs.add((25, y))

    # Center-diagonal pods
    for d in range(4):
        obs.add((12 + d, 16 - d))
        obs.add((13 + d, 16 - d))
    return _bounds_check(obs)


# ── Scenario 6: Random Clutter & Debris ─────────────────────────────
def layout_cluttered(seed: int = 42) -> frozenset:
    """Procedurally clustered pallets and equipment boxes."""
    rng = random.Random(seed)
    obs: Set[Tuple[int, int]] = set()
    # 8 random clusters
    for _ in range(8):
        cx = rng.randint(4, 25)
        cy = rng.randint(4, 25)
        w = rng.randint(2, 4)
        h = rng.randint(1, 3)
        for dx in range(w):
            for dy in range(h):
                obs.add((cx + dx, cy + dy))
    return _bounds_check(obs)


# ── Registry ────────────────────────────────────────────────────────
LAYOUT_REGISTRY: Dict[str, callable] = { # type: ignore
    "Standard Dual-Corridor":   layout_standard,
    "Dense Storage Aisles":     layout_dense_aisles,
    "Central Chokepoints":      layout_chokepoints,
    "Pillar / Column Field":    layout_pillar_field,
    "Asymmetric Distribution":  layout_asymmetric,
    "Random Clutter & Debris":  layout_cluttered,
} 

LAYOUT_NAMES: List[str] = list(LAYOUT_REGISTRY.keys())


def get_layout(name_or_index) -> Tuple[str, frozenset]:
    """Retrieve a layout name and its static obstacle set by name or integer index."""
    if isinstance(name_or_index, int):
        name = LAYOUT_NAMES[name_or_index % len(LAYOUT_NAMES)]
    else:
        name = str(name_or_index)
        if name not in LAYOUT_REGISTRY:
            name = LAYOUT_NAMES[0]
    generator = LAYOUT_REGISTRY[name]
    return name, generator()


def get_random_layout() -> Tuple[str, frozenset]:
    """Pick a random warehouse layout."""
    idx = random.randint(0, len(LAYOUT_NAMES) - 1)
    return get_layout(idx)

