"""
simulation/warehouse_server.py
Central physics and environment simulation engine.
"""
import time
import zmq
import json
from core.grid import WarehouseGrid

class WarehouseSimServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.grid = WarehouseGrid()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self.robot_positions = {}
        print(f"[Warehouse Server] Running physics engine on port {port}...")

    def run(self):
        while True:
            message = self.socket.recv_json()
            msg_type = message.get("type")
            robot_id = message.get("robot_id")

            if msg_type == "REGISTER":
                init_x, init_y = message["x"], message["y"]
                self.robot_positions[robot_id] = (init_x, init_y)
                reply = {
                    "status": "OK",
                    "pos": (init_x, init_y),
                    "static_map": list(self.grid.static_obstacles)
                }

            elif msg_type == "STEP":
                next_x, next_y = message["next_x"], message["next_y"]

                is_cell_free = self.grid.is_valid(next_x, next_y)
                is_occupied = any(
                    pos == (next_x, next_y)
                    for r_id, pos in self.robot_positions.items()
                    if r_id != robot_id
                )

                if is_cell_free and not is_occupied:
                    self.robot_positions[robot_id] = (next_x, next_y)
                    collision = False
                else:
                    collision = True

                reply = {
                    "status": "OK",
                    "current_pos": self.robot_positions[robot_id],
                    "collision": collision,
                    "static_map": list(self.grid.static_obstacles)
                }

            elif msg_type == "INJECT_OBSTACLE":
                ox, oy = message["x"], message["y"]
                self.grid.add_obstacle(ox, oy)
                reply = {"status": "OBSTACLE_ADDED", "pos": (ox, oy)}

            else:
                reply = {"status": "UNKNOWN_COMMAND"}

            self.socket.send_json(reply)

if __name__ == "__main__":
    server = WarehouseSimServer()
    server.run()
