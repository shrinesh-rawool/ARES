"""
visualizer.py — Passive observer dashboard (Pygame + ZeroMQ).

30 × 30 warehouse grid with a telemetry side panel.

Start / stop
------------
  [SPACE]   Broadcast START — releases all robot nodes from WAITING state.
            Shows a "Press SPACE to begin" overlay until first press.

Obstacle injection
------------------
  [1]   3-cell wall across AMR 1's horizontal lane (y=14, x=11..13)
  [2]   3-cell wall across AMR 2's vertical lane   (x=14, y=11..13)
  [B]   Both walls simultaneously
  [O]   Centre block at (14,14)–(14,15)
  [C]   Clear ALL dynamic obstacles (robots replan to direct paths)

Scenario
--------
  [K]   Kill AMR 2 (triggers CNP self-healing in surviving nodes)
  [ESC] Quit
"""

import json
import sys
import time
from typing import Dict, List, Set, Tuple

import pygame
import zmq

from core.protocol import (
    CONTROL_PORT,
    build_control,
    telemetry_port,
)

# ─────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────

GRID_SIZE = 30           # 30 × 30 warehouse
CELL_SIZE = 24           # px per cell  →  720 × 720 grid area
PANEL_W   = 275          # side-panel width
SCREEN_W  = GRID_SIZE * CELL_SIZE + PANEL_W   # 995
SCREEN_H  = GRID_SIZE * CELL_SIZE             # 720
FPS       = 30

# ─────────────────────────────────────────────────────────────────────
# Obstacle presets  (in each AMR's direct path — verified clear of shelves)
# ─────────────────────────────────────────────────────────────────────
#   AMR 1: y=14 horizontal lane  (left → right, x=2..27)
#   AMR 2: x=14 vertical lane    (top → bottom, y=2..27)

AMR1_WALL:   List[Tuple[int,int]] = [(11,14), (12,14), (13,14)]   # blocks AMR 1
AMR2_WALL:   List[Tuple[int,int]] = [(14,11), (14,12), (14,13)]   # blocks AMR 2
CENTRE_OBS:  List[Tuple[int,int]] = [(14,14), (14,15)]            # [O] quick block

# Shelf positions (must match core/grid.py STATIC_OBSTACLES exactly)
SHELVES: frozenset = frozenset({
    (4, 8),  (5, 8),  (6, 8),  (7, 8),  (8, 8),  (9, 8),
    (18, 8), (19, 8), (20, 8), (21, 8), (22, 8), (23, 8),
    (4, 21), (5, 21), (6, 21), (7, 21), (8, 21), (9, 21),
    (18,21), (19,21), (20,21), (21,21), (22,21), (23,21),
})

# ─────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────

C = {
    "1":        (80,  160, 255),   # Blue   – AMR 1
    "2":        (80,  230, 110),   # Green  – AMR 2
    "3":        (255, 200,  60),   # Yellow – AMR 3
    "4":        (255, 120, 200),   # Magenta – AMR 4
    "bg":       (22,  22,  32),
    "grid":     (46,  46,  60),
    "shelf":    (85,  55,  55),
    "obstacle": (210,  55,  55),
    "panel_bg": (13,  13,  22),
    "divider":  (48,  48,  66),
    "text":     (210, 210, 225),
    "dim":      (108, 108, 132),
    "active":   (70,  200,  80),
    "waiting":  (200, 160,  40),
    "dead":     (210,  70,  70),
    "goal_ok":  (70,  200, 200),
    "warn":     (255, 150,  35),
    "event_ok": (80,  220, 120),
    "overlay":  (10,  10,  20),
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def cell_rect(x: int, y: int) -> Tuple[int,int,int,int]:
    return (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

def cell_center(x: int, y: int) -> Tuple[int,int]:
    return (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)

def darken(color: Tuple[int,int,int], f: float = 0.38) -> Tuple[int,int,int]:
    return tuple(max(0, int(c * f)) for c in color) # type: ignore 


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("AMR Fleet MVP — Passive Observer  |  30×30 Warehouse")

    font_hd  = pygame.font.SysFont("monospace", 14, bold=True)
    font_md  = pygame.font.SysFont("monospace", 13)
    font_sm  = pygame.font.SysFont("monospace", 11)
    font_xl  = pygame.font.SysFont("monospace", 28, bold=True)
    font_lg  = pygame.font.SysFont("monospace", 18, bold=True)
    clock    = pygame.time.Clock()

    # ── ZMQ ────────────────────────────────────────────────────────
    ctx = zmq.Context()

    telem = ctx.socket(zmq.SUB)
    for rid in [1, 2, 3, 4]:
        telem.connect(f"tcp://localhost:{telemetry_port(rid)}")
    telem.setsockopt_string(zmq.SUBSCRIBE, "")

    ctrl = ctx.socket(zmq.PUB)
    ctrl.bind(f"tcp://*:{CONTROL_PORT}")

    time.sleep(0.4)   # ZMQ handshake

    # ── State ───────────────────────────────────────────────────────
    robots:    Dict[str, dict] = {}
    dyn_obs:   Set[Tuple[int,int]] = set()
    events:    List[Tuple[float, str, Tuple]] = []
    MAX_EVENTS = 7

    fleet_started = False   # True once SPACE is pressed

    latencies: list = []
    bw_bytes   = 0
    bw_ts      = time.time()
    bw_display = 0.0
    deadlocks  = 0

    # ── Helper: inject obstacles ─────────────────────────────────────
    def inject(cells: List[Tuple[int,int]], label: str) -> None:
        for x, y in cells:
            if (x, y) not in SHELVES:
                dyn_obs.add((x, y))
                ctrl.send_string(build_control("OBSTACLE", x=x, y=y))
        events.append((time.time(), f"▶ {label}", C["warn"]))
        if len(events) > MAX_EVENTS:
            events.pop(0)
        print(f"[Visualizer] {label}")

    def clear_obs() -> None:
        dyn_obs.clear()
        ctrl.send_string(build_control("CLEAR_OBSTACLES"))
        events.append((time.time(), "▶ Obstacles cleared", C["event_ok"]))
        if len(events) > MAX_EVENTS:
            events.pop(0)
        print("[Visualizer] All obstacles cleared.")

    def fire_start() -> None:
        nonlocal fleet_started
        fleet_started = True
        ctrl.send_string(build_control("START"))
        events.append((time.time(), "▶ Fleet STARTED", C["active"]))
        print("[Visualizer] START broadcast → nodes released.")

    # ── Main loop ───────────────────────────────────────────────────
    running = True
    while running:

        # ── Events ──────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_SPACE:
                    if not fleet_started:
                        fire_start()

                elif event.key == pygame.K_1:
                    inject(AMR1_WALL, "Wall on AMR 1 lane (y=14)")

                elif event.key == pygame.K_2:
                    inject(AMR2_WALL, "Wall on AMR 2 lane (x=14)")

                elif event.key == pygame.K_b:
                    inject(AMR1_WALL + AMR2_WALL, "Both walls — AMR 1 + 2 blocked!")

                elif event.key == pygame.K_o:
                    inject(CENTRE_OBS, "Centre block (14,14)")

                elif event.key == pygame.K_c:
                    clear_obs()

                elif event.key == pygame.K_k:
                    ctrl.send_string(build_control("KILL", target_id="2"))
                    events.append((time.time(), "▶ Kill → AMR 2 (CNP)", C["dead"]))
                    if len(events) > MAX_EVENTS:
                        events.pop(0)
                    print("[Visualizer] Kill → AMR 2.")

        # ── Drain telemetry ──────────────────────────────────────────
        try:
            while True:
                raw = telem.recv_string(flags=zmq.NOBLOCK)
                bw_bytes += len(raw.encode())
                data = json.loads(raw)
                robots[str(data["id"])] = data
                ts = data.get("timestamp", 0)
                if ts:
                    latencies.append((time.time() - ts) * 1000)
                    if len(latencies) > 80:
                        latencies.pop(0)
        except zmq.Again:
            pass

        now = time.time()
        if now - bw_ts >= 1.0:
            bw_display = bw_bytes / 1024
            bw_bytes   = 0
            bw_ts      = now
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        # ── Clear screen ─────────────────────────────────────────────
        screen.fill(C["bg"])

        # ── Draw grid ────────────────────────────────────────────────
        for gx in range(GRID_SIZE):
            for gy in range(GRID_SIZE):
                rect = cell_rect(gx, gy)
                if (gx, gy) in SHELVES:
                    pygame.draw.rect(screen, C["shelf"], rect)
                elif (gx, gy) in dyn_obs:
                    pygame.draw.rect(screen, C["obstacle"], rect)
                pygame.draw.rect(screen, C["grid"], rect, 1)

        # ── Ghost outlines for obstacle zones ────────────────────────
        for (x, y), alpha_col in (
            *[(c, (255, 80, 80, 28))  for c in AMR1_WALL],
            *[(c, (80, 80, 255, 28))  for c in AMR2_WALL],
        ):
            if (x, y) not in dyn_obs and (x, y) not in SHELVES:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                s.fill(alpha_col)
                screen.blit(s, (x * CELL_SIZE, y * CELL_SIZE))

        # ── Draw planned path breadcrumbs ─────────────────────────────
        for rid, data in robots.items():
            if data.get("status") == "DEAD":
                continue
            color    = C.get(rid, (200, 200, 200))
            path_col = darken(color, 0.44)
            for i, entry in enumerate(data.get("reservations", [])[1:], start=1):
                rx, ry = int(entry[0]), int(entry[1])
                cx, cy = cell_center(rx, ry)
                r = max(2, CELL_SIZE // 6 - i // 2)
                pygame.draw.circle(screen, path_col, (cx, cy), r)

        # ── Draw goal markers ─────────────────────────────────────────
        for rid, data in robots.items():
            if data.get("status") == "DEAD":
                continue
            color = C.get(rid, (200, 200, 200))
            g = data.get("goal", [0, 0])
            gcx, gcy = cell_center(int(g[0]), int(g[1]))
            pygame.draw.circle(screen, color, (gcx, gcy), 5, 2)

        # ── Draw robots ───────────────────────────────────────────────
        active_count  = 0
        waiting_count = 0
        for rid, data in robots.items():
            status = data.get("status", "WAITING")
            rx, ry = int(data.get("x", 0)), int(data.get("y", 0))
            cx, cy = cell_center(rx, ry)
            color  = C.get(rid, (200, 200, 200))

            if status == "DEAD":
                d = CELL_SIZE // 3
                pygame.draw.line(screen, color, (cx-d, cy-d), (cx+d, cy+d), 3)
                pygame.draw.line(screen, color, (cx+d, cy-d), (cx-d, cy+d), 3)
                continue

            if status == "WAITING":
                waiting_count += 1
                # Pulsing ring for waiting robots
                t_pulse = int(now * 3) % 2
                pulse_r = CELL_SIZE // 3 + (3 if t_pulse else 1)
                pygame.draw.circle(screen, C["waiting"], (cx, cy), CELL_SIZE // 3)
                pygame.draw.circle(screen, C["waiting"], (cx, cy), pulse_r, 2)
            else:
                active_count += 1
                pygame.draw.circle(screen, color, (cx, cy), CELL_SIZE // 3)
                if status == "GOAL_REACHED":
                    pygame.draw.circle(screen, C["goal_ok"], (cx, cy), CELL_SIZE // 3 + 3, 2)

            lbl = font_sm.render(rid, True, (15, 15, 15))
            screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

        # ── Side panel ────────────────────────────────────────────────
        px = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(screen, C["panel_bg"], (px, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, C["divider"],   (px, 0), (px, SCREEN_H), 2)

        py_cur = [14]

        def text(msg: str, bold: bool = False, color=None, indent: int = 0) -> None:
            f = font_hd if bold else font_md
            surf = f.render(msg, True, color or C["text"])
            screen.blit(surf, (px + 12 + indent, py_cur[0]))
            py_cur[0] += 20

        def dim(msg: str, indent: int = 6) -> None:
            surf = font_sm.render(msg, True, C["dim"])
            screen.blit(surf, (px + 12 + indent, py_cur[0]))
            py_cur[0] += 16

        def sep(n: int = 6) -> None:
            py_cur[0] += n

        # ── KPIs ──────────────────────────────────────────────────────
        status_line = "RUNNING" if fleet_started else "WAITING FOR START"
        status_col  = C["active"] if fleet_started else C["waiting"]
        text("FLEET TELEMETRY",     bold=True)
        dim("─" * 25, indent=0)
        text(f"Status       : {status_line}", color=status_col)
        text(f"Active Nodes : {active_count} / 3")
        text(f"Waiting      : {waiting_count}")
        text(f"Avg Latency  : {avg_lat:.1f} ms")
        text(f"Bandwidth    : {bw_display:.2f} KB/s")
        text(f"Deadlocks    : {deadlocks}")
        sep()

        # ── Node status ────────────────────────────────────────────────
        text("NODE STATUS",         bold=True)
        dim("─" * 25, indent=0)
        for rid in ["1", "2", "3"]:
            if rid not in robots:
                dim(f"AMR {rid}: connecting…", indent=0)
                sep(3)
                continue
            d      = robots[rid]
            st     = d.get("status", "?")
            rx, ry = d.get("x", 0), d.get("y", 0)
            goal   = d.get("goal", [0, 0])
            pri    = d.get("priority", 0.0)

            sc = (C["dead"]    if st == "DEAD"         else
                  C["goal_ok"] if st == "GOAL_REACHED" else
                  C["waiting"] if st == "WAITING"      else
                  C["active"])
            color = C.get(rid, C["text"])

            sq_x, sq_y = px + 12, py_cur[0] + 3
            pygame.draw.rect(screen, color, (sq_x, sq_y, 8, 14))
            surf = font_md.render(f" {rid}  {st}", True, sc)
            screen.blit(surf, (sq_x + 10, py_cur[0]))
            py_cur[0] += 20
            dim(f"({rx:2d},{ry:2d}) → ({goal[0]:2d},{goal[1]:2d})  pri={pri:.0f}")
            sep(3)

        # ── Event log ──────────────────────────────────────────────────
        sep(2)
        text("EVENT LOG",           bold=True)
        dim("─" * 25, indent=0)
        if not events:
            dim("(no events yet)", indent=0)
        else:
            for ts, msg, col in reversed(events[-MAX_EVENTS:]):
                age = now - ts
                c = C["dim"] if age > 10 else col
                dim(f"{msg[:27]}", indent=0)
        sep()

        # ── Keys ───────────────────────────────────────────────────────
        text("CONTROLS",            bold=True)
        dim("─" * 25, indent=0)
        dim("[SPC] Start fleet", indent=0)
        dim("[1]   AMR 1 wall", indent=0)
        dim("[2]   AMR 2 wall", indent=0)
        dim("[B]   Both walls", indent=0)
        dim("[O]   Centre block", indent=0)
        dim("[C]   Clear obstacles", indent=0)
        dim("[K]   Kill AMR 2 (CNP)", indent=0)
        dim("[ESC] Quit", indent=0)

        # ── START overlay ─────────────────────────────────────────────
        if not fleet_started:
            overlay = pygame.Surface((GRID_SIZE * CELL_SIZE, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((10, 10, 20, 185))
            screen.blit(overlay, (0, 0))

            # Count connected nodes
            connected = len(robots)
            all_waiting = all(
                v.get("status") == "WAITING" for v in robots.values()
            ) if robots else False

            # Status message
            if connected == 0:
                sub = "Waiting for nodes to connect…"
                sub_col = C["dim"]
            elif connected < 3:
                sub = f"{connected}/3 nodes connected…"
                sub_col = C["waiting"]
            else:
                sub = "All 3 nodes ready  ✓" if all_waiting else f"{connected}/3 nodes online"
                sub_col = C["active"] if all_waiting else C["waiting"]

            cx_grid = (GRID_SIZE * CELL_SIZE) // 2
            cy_grid = SCREEN_H // 2

            title = font_xl.render("AMR FLEET MVP", True, C["text"])
            screen.blit(title, (cx_grid - title.get_width()//2, cy_grid - 80))

            ready_txt = font_lg.render("Press  SPACE  to start", True, (255, 255, 255))
            screen.blit(ready_txt, (cx_grid - ready_txt.get_width()//2, cy_grid - 10))

            sub_surf = font_md.render(sub, True, sub_col)
            screen.blit(sub_surf, (cx_grid - sub_surf.get_width()//2, cy_grid + 30))

            # Pulsing border on the SPACE prompt
            pulse_w = ready_txt.get_width() + 24
            pulse_h = ready_txt.get_height() + 12
            pulse_x = cx_grid - pulse_w // 2
            pulse_y = cy_grid - 10 - 6
            t_pulse = int(now * 2) % 2
            border_col = (200, 200, 200) if t_pulse else (100, 100, 100)
            pygame.draw.rect(screen, border_col, (pulse_x, pulse_y, pulse_w, pulse_h), 2, border_radius=6)

        pygame.display.flip()
        clock.tick(FPS)

    # ── Cleanup ───────────────────────────────────────────────────────
    telem.setsockopt(zmq.LINGER, 0)
    ctrl.setsockopt(zmq.LINGER, 0)
    telem.close()
    ctrl.close()
    ctx.term()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
