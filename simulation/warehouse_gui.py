"""
simulation/warehouse_gui.py
Pygame Visualizer with Goal Reticles and Package Indicators.
"""

import sys
import time
import zmq
import pygame
from typing import Dict, Tuple
from core.grid import WarehouseGrid

# Visual configuration
CELL_SIZE = 24
GRID_DIM = 30
MAP_SIZE = CELL_SIZE * GRID_DIM
PANEL_WIDTH = 280
SCREEN_SIZE = (MAP_SIZE + PANEL_WIDTH, MAP_SIZE)
FPS = 30

# Color definitions
COLOR_BG = (245, 246, 250)
COLOR_GRID_LINE = (220, 221, 225)
COLOR_STATIC_WALL = (47, 53, 66)
COLOR_DYNAMIC_OBS = (235, 77, 75)
COLOR_CHOKE_ACCENT = (255, 165, 2)
COLOR_PACKAGE = (184, 115, 51)       # Cardboard / copper payload color
COLOR_PACKAGE_BORDER = (90, 45, 15)

ROBOT_COLORS = {
    "amr_1": (46, 204, 113),   # Emerald Green
    "amr_2": (52, 152, 219),   # Sky Blue
    "amr_3": (155, 89, 182),   # Amethyst Purple
}
COLOR_DEFAULT_ROBOT = (230, 126, 34)


class WarehouseSimulation:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555, headless: bool = False):
        self.grid = WarehouseGrid()
        self.headless = headless

        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

        # Ground-truth tracking for the map and fleet telemetry panel.
        self.robots: Dict[str, dict] = {}

        if self.headless:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        self.font = None
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 11, bold=True)
            self.panel_font = pygame.font.SysFont("Consolas", 13)
            self.panel_bold = pygame.font.SysFont("Consolas", 13, bold=True)
            self.panel_title = pygame.font.SysFont("Consolas", 15, bold=True)
        except Exception:
            self.font = None
            self.panel_font = None
            self.panel_bold = None
            self.panel_title = None

        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("ARES — Multi-Agent Warehouse Visualizer")
        self.clock = pygame.time.Clock()

    def handle_network_requests(self):
        while True:
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            msg_type = message.get("type")
            robot_id = message.get("robot_id", "unknown")

            if msg_type == "REGISTER":
                pos = (int(message["x"]), int(message["y"]))
                self.robots[robot_id] = {
                    "pos": pos,
                    "goal": tuple(message.get("goal", pos)),
                    "has_payload": message.get("has_payload", True),
                    "last_update": time.monotonic(),
                    "step_count": 0,
                    "collision_count": 0,
                    "state": message.get("state", "STOPPED"),
                }
                response = {
                    "status": "REGISTERED",
                    "pos": pos,
                    "static_map": list(self.grid.static_obstacles),
                    "dynamic_map": list(self.grid.all_obstacles - self.grid.static_obstacles),
                }

            elif msg_type == "STEP":
                nx, ny = int(message["next_x"]), int(message["next_y"])
                goal = tuple(message.get("goal", (0, 0)))
                has_payload = message.get("has_payload", False)
                state = message.get("state", "MOVING")

                wall_conflict = not self.grid.is_valid(nx, ny)
                robot_conflict = any(
                    data["pos"] == (nx, ny) for r_id, data in self.robots.items() if r_id != robot_id
                )

                collision = wall_conflict or robot_conflict

                if not collision:
                    if robot_id not in self.robots:
                        self.robots[robot_id] = {}
                    self.robots[robot_id]["pos"] = (nx, ny)

                if robot_id in self.robots:
                    self.robots[robot_id]["last_update"] = time.monotonic()
                    self.robots[robot_id]["step_count"] = self.robots[robot_id].get("step_count", 0) + 1
                    if collision:
                        self.robots[robot_id]["collision_count"] = self.robots[robot_id].get("collision_count", 0) + 1

                if robot_id in self.robots:
                    self.robots[robot_id]["goal"] = goal
                    self.robots[robot_id]["has_payload"] = has_payload
                    self.robots[robot_id]["state"] = state

                response = {
                    "status": "OK",
                    "current_pos": self.robots[robot_id]["pos"],
                    "collision": collision,
                    "static_map": list(self.grid.static_obstacles),
                    "dynamic_map": list(self.grid.all_obstacles - self.grid.static_obstacles),
                }

            elif msg_type == "INJECT_OBSTACLE":
                ox, oy = int(message["x"]), int(message["y"])
                self.grid.add_obstacle(ox, oy)
                response = {"status": "OBSTACLE_ADDED", "pos": (ox, oy)}

            elif msg_type == "REMOVE_OBSTACLE":
                ox, oy = int(message["x"]), int(message["y"])
                self.grid.remove_obstacle(ox, oy)
                response = {"status": "OBSTACLE_REMOVED", "pos": (ox, oy)}

            else:
                response = {"status": "UNKNOWN_ACTION"}

            self.socket.send_json(response)

    def render(self):
        self.screen.fill(COLOR_BG)

        # 1. Subtle grid lines
        for x in range(0, MAP_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (x, 0), (x, MAP_SIZE))
        for y in range(0, MAP_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (0, y), (MAP_SIZE, y))

        # 2. Highlight intersection choke point at (14, 14)
        choke_rect = pygame.Rect(14 * CELL_SIZE, 14 * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, COLOR_CHOKE_ACCENT, choke_rect)

        # 3. Static shelf walls
        for (ox, oy) in self.grid.static_obstacles:
            rect = pygame.Rect(ox * CELL_SIZE, oy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, COLOR_STATIC_WALL, rect)

        # 4. Dynamic obstacles placed via mouse click
        for (dx, dy) in self.grid.all_obstacles - self.grid.static_obstacles:
            rect = pygame.Rect(dx * CELL_SIZE, dy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, COLOR_DYNAMIC_OBS, rect)

        # 5. Draw Target/Goal Markers
        for r_id, data in self.robots.items():
            gx, gy = data.get("goal", (0, 0))
            if gx or gy:
                color = ROBOT_COLORS.get(r_id, COLOR_DEFAULT_ROBOT)
                g_center = (gx * CELL_SIZE + CELL_SIZE // 2, gy * CELL_SIZE + CELL_SIZE // 2)

                # Outer target ring
                pygame.draw.circle(self.screen, color, g_center, CELL_SIZE // 2 - 1, 2)
                # Inner bulls-eye dot
                pygame.draw.circle(self.screen, color, g_center, 3)

        # 6. Draw AMRs & Payloads
        for r_id, data in self.robots.items():
            rx, ry = data["pos"]
            has_payload = data.get("has_payload", False)
            color = ROBOT_COLORS.get(r_id, COLOR_DEFAULT_ROBOT)
            center = (rx * CELL_SIZE + CELL_SIZE // 2, ry * CELL_SIZE + CELL_SIZE // 2)

            # Outer chassis
            pygame.draw.circle(self.screen, color, center, CELL_SIZE // 2 - 2)
            pygame.draw.circle(self.screen, (25, 25, 25), center, CELL_SIZE // 2 - 2, 1)

            # Cargo indicator: Draw miniature package box inside chassis
            if has_payload:
                box_w = CELL_SIZE // 2
                box_rect = pygame.Rect(center[0] - box_w // 2, center[1] - box_w // 2, box_w, box_w)
                pygame.draw.rect(self.screen, COLOR_PACKAGE, box_rect)
                pygame.draw.rect(self.screen, COLOR_PACKAGE_BORDER, box_rect, 1)
            else:
                # Unladen state: Small hollow core
                pygame.draw.circle(self.screen, (255, 255, 255), center, 3)

            # Robot ID text label
            if self.font:
                label = self.font.render(r_id[-1], True, (255, 255, 255))
                label_rect = label.get_rect(center=center)
                self.screen.blit(label, label_rect)

        self.render_fleet_panel()

        pygame.display.flip()

    def render_fleet_panel(self):
        panel_x = MAP_SIZE
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, MAP_SIZE)
        pygame.draw.rect(self.screen, (32, 36, 48), panel_rect)
        pygame.draw.line(self.screen, (80, 86, 100), (panel_x, 0), (panel_x, MAP_SIZE), 2)

        if not self.panel_font:
            return

        def draw_text(text, x, y, color=(220, 223, 230), font=None):
            self.screen.blit((font or self.panel_font).render(text, True, color), (x, y))

        left = panel_x + 16
        line_height = 20
        active_count = sum(
            time.monotonic() - data.get("last_update", 0) < 2.0
            for data in self.robots.values()
        )
        payload_count = sum(data.get("has_payload", False) for data in self.robots.values())
        collision_count = sum(data.get("collision_count", 0) for data in self.robots.values())

        draw_text("FLEET TELEMETRY", left, 18, (255, 204, 77), self.panel_title)
        pygame.draw.line(self.screen, (87, 91, 105), (left, 45), (panel_x + PANEL_WIDTH - 16, 45))
        draw_text(f"Active Nodes : {active_count} / 3", left, 58)
        draw_text(f"Loaded       : {payload_count}", left, 78)
        draw_text(f"Collisions   : {collision_count}", left, 98)
        draw_text(f"Obstacles    : {len(self.grid.all_obstacles)}", left, 118)

        section_y = 158
        draw_text("NODE STATUS", left, section_y, (255, 204, 77), self.panel_title)
        pygame.draw.line(self.screen, (87, 91, 105), (left, section_y + 27), (panel_x + PANEL_WIDTH - 16, section_y + 27))

        robot_ids = sorted(self.robots, key=lambda robot_id: (robot_id not in ROBOT_COLORS, robot_id))
        for index, robot_id in enumerate(robot_ids):
            data = self.robots[robot_id]
            block_y = section_y + 42 + index * 126
            color = ROBOT_COLORS.get(robot_id, COLOR_DEFAULT_ROBOT)
            is_active = time.monotonic() - data.get("last_update", 0) < 2.0
            status = data.get("state", "WAITING") if is_active else "STOPPED"
            rx, ry = data.get("pos", (0, 0))
            gx, gy = data.get("goal", (0, 0))
            distance = abs(rx - gx) + abs(ry - gy)
            cargo = "LOADED" if data.get("has_payload", False) else "EMPTY"

            pygame.draw.rect(self.screen, color, (left, block_y + 3, 8, 16))
            draw_text(f"{robot_id.upper()}  {status}", left + 16, block_y, color, self.panel_bold)
            draw_text(f"Position     ({rx:2}, {ry:2})", left + 16, block_y + line_height)
            draw_text(f"Target       ({gx:2}, {gy:2})", left + 16, block_y + line_height * 2)
            draw_text(f"Distance     {distance:2} cells", left + 16, block_y + line_height * 3)
            draw_text(f"Cargo        {cargo}", left + 16, block_y + line_height * 4)
            draw_text(f"Steps        {data.get('step_count', 0):2}   Hits {data.get('collision_count', 0):2}", left + 16, block_y + line_height * 5, (170, 174, 185))

        footer_y = MAP_SIZE - 54
        pygame.draw.line(self.screen, (87, 91, 105), (left, footer_y), (panel_x + PANEL_WIDTH - 16, footer_y))
        draw_text("MAP CONTROLS", left, footer_y + 10, (255, 204, 77), self.panel_bold)
        draw_text("Click: add/remove obstacle", left, footer_y + 30, (170, 174, 185))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    if mx >= MAP_SIZE or my >= MAP_SIZE:
                        continue
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    if (gx, gy) in self.grid.all_obstacles:
                        self.grid.remove_obstacle(gx, gy)
                    else:
                        self.grid.add_obstacle(gx, gy)

            self.handle_network_requests()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = WarehouseSimulation(host="0.0.0.0", port=5555, headless=False)
    sim.run()
