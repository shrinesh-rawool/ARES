"""
core/protocol.py — Fleet communication protocol.

Responsibilities
----------------
* Port assignments for every ZeroMQ socket.
* Building structured telemetry + control JSON packets.
* Priority arithmetic and choke-point negotiation.
* Contract Net Protocol (CNP) self-healing logic.
* Heartbeat failure detection.

Time model
----------
Global ticks are derived from the unix clock:

    global_tick() = int(time.time() * TICK_RATE)

Since all processes run on the same host, this gives a shared,
monotonically increasing integer clock with ~200 ms resolution.
Reservation entries always use absolute global ticks so that
packets from different robots can be compared directly.
"""

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────
# Clock
# ──────────────────────────────────────────────────────────────────────

TICK_RATE: int = 5          # 5 Hz  →  0.2 s per step
TICK_INTERVAL: float = 1.0 / TICK_RATE


def global_tick() -> int:
    """Shared integer clock: ticks at TICK_RATE per second."""
    return int(time.time() * TICK_RATE)


# ──────────────────────────────────────────────────────────────────────
# Port layout
# ──────────────────────────────────────────────────────────────────────

TELEMETRY_PORT_BASE: int = 5550   # node i → port 5550+i  (i in {1,2,3})
CONTROL_PORT: int = 5560          # visualizer PUBs here; nodes SUB here

RESERVATION_HORIZON: int = 10     # steps ahead each node broadcasts
HEARTBEAT_TIMEOUT: float = 1.5    # seconds before a node is declared dead


def telemetry_port(robot_id) -> int:
    return TELEMETRY_PORT_BASE + int(robot_id)


# ──────────────────────────────────────────────────────────────────────
# Packet builders
# ──────────────────────────────────────────────────────────────────────

def build_telemetry(
    robot_id: str,
    x: int,
    y: int,
    goal: Tuple[int, int],
    status: str,
    planned_path: List[Tuple[int, int]],
    step_idx: int,
    urgency: int = 1,
) -> str:
    """
    Build a JSON telemetry string to broadcast via ZMQ PUB.

    Reservations use *absolute* global ticks so that any subscriber
    can directly compare against its own global_tick() value.

    Format of each reservation entry: [x, y, absolute_global_tick]
    The first entry is always the robot's current position at current_t.
    """
    current_t = global_tick()
    dist = abs(x - goal[0]) + abs(y - goal[1])
    priority = calc_priority(urgency, dist)

    # Current position at current_t (offset 0)
    reservations: List[List[int]] = [[x, y, current_t]]

    # Future positions at current_t + 1 … + RESERVATION_HORIZON
    future = planned_path[step_idx: step_idx + RESERVATION_HORIZON]
    for offset, (rx, ry) in enumerate(future, start=1):
        reservations.append([rx, ry, current_t + offset])

    # Pad remaining horizon slots with final position so peers don't plan through stationary robots
    pad_pos = future[-1] if future else (x, y)
    for offset in range(len(future) + 1, RESERVATION_HORIZON + 1):
        reservations.append([pad_pos[0], pad_pos[1], current_t + offset])

    return json.dumps({
        "id":           str(robot_id),
        "x":            x,
        "y":            y,
        "goal":         list(goal),
        "status":       status,
        "priority":     round(priority, 3),
        "urgency":      urgency,
        "reservations": reservations,
        "timestamp":    time.time(),
        "global_tick":  current_t,
    })


def build_control(ctrl_type: str, **kwargs) -> str:
    """Build a JSON control message (obstacle injection, kill signal, etc.)."""
    msg = {"type": ctrl_type}
    msg.update(kwargs)
    return json.dumps(msg)


# ──────────────────────────────────────────────────────────────────────
# Priority & negotiation
# ──────────────────────────────────────────────────────────────────────

def calc_priority(urgency: int, remaining_distance: int) -> float:
    """
    Higher score → higher right-of-way at choke points.

    priority = urgency × 50  +  100 / (remaining_distance + 1)

    The distance term rewards robots that are close to finishing their task.
    """
    return urgency * 50 + 100.0 / (remaining_distance + 1)


def negotiate_choke_point(my_priority: float, peer_priority: float) -> str:
    """Return 'ADVANCE' if I have equal or higher priority, else 'YIELD'."""
    return "ADVANCE" if my_priority >= peer_priority else "YIELD"


# ──────────────────────────────────────────────────────────────────────
# Heartbeat & failure detection
# ──────────────────────────────────────────────────────────────────────

def detect_failure(last_heartbeat_ts: float, timeout: float = HEARTBEAT_TIMEOUT) -> bool:
    """Return True if a peer has been silent longer than `timeout` seconds."""
    return (time.time() - last_heartbeat_ts) > timeout


# ──────────────────────────────────────────────────────────────────────
# Contract Net Protocol (CNP) — self-healing task re-allocation
# ──────────────────────────────────────────────────────────────────────

def cnp_winner(candidates: Sequence[Tuple[str, float]]) -> str:
    """
    Given a list of (robot_id, marginal_distance_to_target),
    return the robot_id with the lowest marginal distance.

    The 'winner' of the CNP auction is the robot best placed to
    take on the failed peer's task.
    """
    if not candidates:
        raise ValueError("cnp_winner: candidates list is empty")
    return min(candidates, key=lambda c: c[1])[0]

