import unittest
from src.grid import *


class GridTest(unittest.TestCase):
    def test_gridconstructor(self):
        grid = read_grid("graph.txt")
        self.assertEqual(type(grid), Grid)
        self.assertEqual(type(grid.graph), nx.Graph)




if __name__ == "__main__":
    unittest.main()
