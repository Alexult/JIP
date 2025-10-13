import unittest
import numpy as np
from energymarket import *


class EnergyMarketTest(unittest.TestCase):
    def test_clear_market(self):
        market = DoubleAuctionClearingAgent()

        # test errors
        bids = np.array([[0, 5], [2, 20]])
        offers = np.array([[3, 5, 20, 2], [4, 5, 20, 2]])

        with self.assertRaises(Exception) as err:
            market.clear_market(bids, offers)
        self.assertEqual(ValueError, type(err.exception))

        # test single bid clearing
        bids = np.array([[0, 5, 20]])
        offers = np.array([[3, 5, 20]])

        result = market.clear_market(bids, offers)
        self.assertEqual((5, 20), result)

        # test single bid clearing
        bids = np.array([[0, 5, 20]])
        offers = np.array([[3, 7, 20]])

        result = market.clear_market(bids, offers)
        self.assertEqual((0, 0), result)

        bids = np.array([[0, 5, 20]])
        offers = np.array([[3, 5, 400]])

        result = market.clear_market(bids, offers)
        self.assertEqual((5, 20), result)

        # test simple two dim bids
        bids = np.array([[0, 5, 20], [1, 5, 20]])
        offers = np.array([[3, 5, 20], [4, 5, 20]])

        result = market.clear_market(bids, offers)
        self.assertEqual((5, 40), result)

        bids = np.array([[0, 3, 20], [1, 5, 20]])
        offers = np.array([[3, 4, 20], [4, 4.5, 20]])

        result = market.clear_market(bids, offers)
        self.assertEqual((4.5, 20), result)


if __name__ == "__main__":
    unittest.main()
