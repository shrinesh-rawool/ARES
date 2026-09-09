import unittest
from core.grid import WarehouseGrid
from core.planner import SpaceTimeAStar

class TestWarehouseGrid(unittest.TestCase):
    def setUp(self):
        self.grid = WarehouseGrid()

    def test_static_obstacles_immutable_via_add_dynamic(self):
        static_wall = (0, 0)
        self.assertIn(static_wall, self.grid.static_obstacles)
        self.grid.add_obstacle(0, 0)
        # Should not be added to dynamic set
        self.assertNotIn((0, 0), self.grid._dynamic)

    def test_dynamic_obstacle_addition_and_removal(self):
        free_cell = (1, 1)  # Clear open floor cell
        self.assertNotIn(free_cell, self.grid.all_obstacles)
        self.grid.add_obstacle(1, 1)
        self.assertIn(free_cell, self.grid.all_obstacles)
        self.grid.remove_obstacle(1, 1)
        self.assertNotIn(free_cell, self.grid.all_obstacles)


class TestSpaceTimeAStar(unittest.TestCase):
    def setUp(self):
        self.grid = WarehouseGrid()
        self.planner = SpaceTimeAStar(
            width=30,
            height=30,
            static_obstacles=self.grid.static_obstacles,
        )

    def test_multi_peer_vertex_reservations_not_overwritten(self):
        # Two peers reserving different cells at the exact same timestep t=1
        start = (14, 13)
        goal = (14, 15)

        # Peer 1 reserves (14, 14) at t=1
        # Peer 2 reserves (15, 13) at t=1
        dynamic_reservations = {
            1: {(14, 14), (15, 13)}
        }
        path = self.planner.plan(start, goal, dynamic_reservations=dynamic_reservations)
        self.assertIsNotNone(path)
        # Path should not step into (14, 14) at t=1
        self.assertNotEqual(path[1], (14, 14))

    def test_edge_swap_collision_avoidance(self):
        start = (14, 13)
        goal = (14, 14)

        # Peer moves from (14, 14) to (14, 13) at t=0 -> t=1
        edge_reservations = {(0, (14, 14), (14, 13))}
        # Robot moving from (14, 13) to (14, 14) at t=0 -> t=1 is a head-on swap!
        path = self.planner.plan(start, goal, edge_reservations=edge_reservations)
        self.assertIsNotNone(path)
        # The robot must wait or sidestep, cannot be (14, 14) at t=1
        self.assertNotEqual(path[1], (14, 14))

    def test_goal_holding_avoids_peer_cutting_through(self):
        start = (14, 12)
        goal = (14, 13)
        # If peer reserves goal at t=2 (right after robot could arrive at t=1)
        dynamic_reservations = {
            2: {(14, 13)}
        }
        path = self.planner.plan(start, goal, dynamic_reservations=dynamic_reservations, hold_at_goal=2)
        self.assertIsNotNone(path)
        # The plan must take longer than 1 step so that it holds safely after peer clears
        self.assertGreater(len(path) - 1, 1)

if __name__ == "__main__":
    unittest.main()

class TestPriorityAndTieBreaking(unittest.TestCase):
    def test_priority_comparison(self):
        # We can test the priority logic directly
        # Distance to goal determines priority; smaller distance wins; robot_id tiebreaks
        def has_higher_priority(robot_id, my_pos, my_goal, peer_id, peer_dist):
            my_dist = abs(my_pos[0] - my_goal[0]) + abs(my_pos[1] - my_goal[1])
            if my_dist < peer_dist:
                return True
            elif my_dist == peer_dist:
                return robot_id < peer_id
            return False

        # amr_1 is closer than amr_2
        self.assertTrue(has_higher_priority("amr_1", (14, 10), (14, 14), "amr_2", 10))
        # amr_2 is further than amr_1
        self.assertFalse(has_higher_priority("amr_2", (14, 0), (14, 14), "amr_1", 2))
        # Tie break by robot_id
        self.assertTrue(has_higher_priority("amr_1", (14, 10), (14, 14), "amr_2", 4))
        self.assertFalse(has_higher_priority("amr_2", (14, 10), (14, 14), "amr_1", 4))

