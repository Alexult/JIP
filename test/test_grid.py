import unittest
from src.grid import *


class GridTest(unittest.TestCase):
    def test_grid_constructor(self):
        grid = read_grid("test_graph.txt")
        self.assertEqual(type(grid), Grid)
        self.assertEqual(type(grid.graph), nx.Graph)

        self.assertEqual(grid.graph.number_of_nodes(), 8)
        self.assertEqual(grid.graph.number_of_edges(), 7)
        self.assertEqual(grid.graph.nodes["Transformer"]["capacity"], 3)

        self.assertEqual(grid.graph.edges[("Transformer", "Feeder1")]["capacity"], -1)
        self.assertEqual(grid.graph.edges[("2", "3")]["capacity"], 1)

    def test_update_capacity(self):
        grid = read_grid("test_graph.txt")
        self.assertEqual(grid.graph.edges[("2", "3")]["flow"], 0)
        grid.update_capacity([("2", "3", 1)])
        self.assertEqual(grid.graph.edges[("2", "3")]["flow"], 1)
        grid.update_capacity([("2", "3", -2)])
        self.assertEqual(grid.graph.edges[("2", "3")]["flow"], -1)

        with self.assertRaises(Exception) as err:
            grid.update_capacity([("2", "3", -2)])

        self.assertEqual(str(err.exception), "New flow exceeds capacity")
        self.assertEqual(grid.graph.edges[("2", "3")]["flow"], -1)

        grid.update_capacity([("Transformer", "4", 2)])
        self.assertEqual(grid.graph.nodes["Transformer"]["flow"], 2)
        with self.assertRaises(Exception) as err:
            grid.update_capacity([("Transformer", "1", 2)])

        self.assertEqual(str(err.exception), "New flow exceeds transformer capacity")

    def test_retrieve_capacity(self):
        grid = read_grid("test_graph.txt")
        grid.update_capacity([("Transformer", "3", 1), ("4", "1", 2)])

        self.assertEqual(grid.retrieve_capacity("Transformer", "3"), 0)
        self.assertEqual(grid.retrieve_capacity("3", "Transformer"), 2)
        self.assertEqual(grid.retrieve_capacity("4", "1"), 0)
        self.assertEqual(grid.retrieve_capacity("1", "4"), 4)
        self.assertEqual(grid.retrieve_capacity("1", "Transformer"), 2)


if __name__ == "__main__":
    unittest.main()
