"""
simulation/warehouse_gui.py
Pygame Visualizer with Goal Reticles and Package Indicators.
"""

import sys
import zmq
import pygame
from typing import Dict, Tuple
from core.grid import WarehouseGrid

# Visual configuration
CELL_SIZE = 24
GRID_DIM = 30
SCREEN_SIZE = CELL_SIZE * GRID_DIM
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

        # Ground-truth tracking: {robot_id: {"pos": (x, y), "goal": (x, y), "has_payload": bool}}
        self.robots: Dict[str, dict] = {}

        if self.headless:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        self.font = None
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 11, bold=True)
        except Exception:
            self.font = None

        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
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
                }
                response = {
                    "status": "REGISTERED",
                    "pos": pos,
                    "static_map": list(self.grid.static_obstacles),
                }

            elif msg_type == "STEP":
                nx, ny = int(message["next_x"]), int(message["next_y"])
                goal = tuple(message.get("goal", (0, 0)))
                has_payload = message.get("has_payload", False)

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
                    self.robots[robot_id]["goal"] = goal
                    self.robots[robot_id]["has_payload"] = has_payload

                response = {
                    "status": "OK",
                    "current_pos": self.robots[robot_id]["pos"],
                    "collision": collision,
                    "static_map": list(self.grid.static_obstacles),
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
        for x in range(0, SCREEN_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (x, 0), (x, SCREEN_SIZE))
        for y in range(0, SCREEN_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (0, y), (SCREEN_SIZE, y))

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

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = pygame.mouse.get_pos()
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
