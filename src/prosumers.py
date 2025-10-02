import numpy as np
from gymnasium.spaces import Box
from custom_types import *
import random


class ProsumerAgent:
    """
    Represents a prosumer (producer + consumer) agent.
    This base class manages state, profit calculation, and defines a default strategy.
    """

    def __init__(
        self,
        agent_id: int,
        fixed_load: float,
        flexible_load_max: float,
        generation_capacity: float,
    ):
        self.agent_id = agent_id
        self.fixed_load = fixed_load
        self.flexible_load_max = flexible_load_max
        self.generation_capacity = generation_capacity
        self.net_demand = 0.0
        self.last_bid_offer: tuple[float, float] | None = None  # (price, quantity)
        self.profit = 0.0

    def calculate_net_demand(self):
        """Simulates the agent's internal state for a new timestep (i.e., external factors)."""
        current_flexible_load = random.uniform(0, self.flexible_load_max)
        total_load = self.fixed_load + current_flexible_load
        self.net_demand = total_load - self.generation_capacity

    def get_market_submission(
        self, price: float, quantity: float
    ) -> tuple[list[Bid], list[Offer]]:
        """
        Takes the action (price, quantity) from the strategy method and translates it
        into a bid or offer based on current net demand.
        """
        bids: list[Bid] = []
        offers: list[Offer] = []

        quantity = max(0.0, quantity)

        # Action is interpreted as buying (bid) if net_demand > 0 (short)
        if self.net_demand > 0:
            bids.append((self.agent_id, price, quantity))
            self.last_bid_offer = (price, quantity)
        # Action is interpreted as selling (offer) if net_demand < 0 (surplus)
        elif self.net_demand < 0:
            offers.append((self.agent_id, price, quantity))
            self.last_bid_offer = (price, quantity)
        else:
            self.last_bid_offer = None

        return bids, offers

    def calculate_profit(self, clearing_price: float) -> float:
        """
        Calculates the agent's profit or loss and stores it.
        Returns the profit for the reward signal.
        """
        profit = 0.0
        if self.last_bid_offer:
            price_submitted, qty_submitted = self.last_bid_offer

            # Agent was a Buyer (bid)
            if self.net_demand > 0:
                # A bid clears if the bid price is greater than or equal to the clearing price
                if price_submitted >= clearing_price:
                    # Profit is the consumer surplus (willingness to pay - price paid)
                    profit = (price_submitted - clearing_price) * qty_submitted

            # Agent was a Seller (offer)
            elif self.net_demand < 0:
                # An offer clears if the offer price is less than or equal to the clearing price
                if price_submitted <= clearing_price:
                    # Profit is the producer surplus (price received - cost of selling)
                    profit = (clearing_price - price_submitted) * qty_submitted

        self.profit = profit
        return profit

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        """
        [DEFAULT STRATEGY]
        This method defines the agent's action based on its current observation.
        This is the method to be overridden by RL policies or other heuristic agents.

        Args:
            obs (np.ndarray): The 5-feature observation for the current agent.
                              [ND_i, P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1]
            action_space (Box): The action space definition for bounds clipping.

        Returns:
            np.ndarray: The chosen action [Price, Quantity].
        """
        # Observation features: [ND_i, P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1]
        net_demand = obs[0]
        last_price = obs[1]

        # --- Simple Strategy Logic (Policy) ---

        # Quantity: Agent bids/offers 80% of its current net demand imbalance
        quantity = np.clip(
            abs(net_demand) * 0.8, action_space.low[1], action_space.high[1]
        )

        # Price: Adjust the bid/offer price relative to the last clearing price
        if net_demand > 0:  # Buyer (needs energy, submits a BID)
            # Strategy: Bid slightly higher than last price to ensure clearance
            price = last_price * 1.05 + 0.1
        elif net_demand < 0:  # Seller (has surplus, submits an OFFER)
            # Strategy: Offer slightly lower than last price to ensure clearance
            price = last_price * 0.95 - 0.1
        else:  # Net demand is zero (no action)
            price = 0.0
            quantity = 0.0

        # Ensure price is within the defined market bounds
        price = np.clip(price, action_space.low[0], action_space.high[0])

        # Return the final action [Price, Quantity]
        return np.array([price, quantity], dtype=np.float32)


class AggressiveSellerAgent(ProsumerAgent):
    """
    Example subclass: This agent tries to sell its entire surplus (if negative net demand)
    at the lowest possible price ($0.01) to maximize clearance, or buys normally.
    """

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        """Overrides the default strategy with an aggressive selling policy."""
        net_demand = obs[0]
        last_price = obs[1]

        quantity = np.clip(
            abs(net_demand),  # Use full net demand magnitude
            action_space.low[1],
            action_space.high[1],
        )

        if net_demand > 0:  # Buyer (use base class's smart buying logic)
            # Strategy: Bid slightly higher than last price
            price = last_price * 1.05 + 0.1
        elif net_demand < 0:  # Seller (Aggressive Selling)
            # Strategy: Offer at the absolute minimum price to guarantee being the lowest bid
            price = 0.01
        else:
            price = 0.0
            quantity = 0.0

        price = np.clip(price, action_space.low[0], action_space.high[0])
        return np.array([price, quantity], dtype=np.float32)


class AggressiveBuyerAgent(ProsumerAgent):
    """
    Example subclass: This agent tries to buy aggressively at the max possible price ($10.00)
    when it has positive net demand to maximize clearance, or sells normally.
    """

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        """Overrides the default strategy with an aggressive buying policy."""
        net_demand = obs[0]
        last_price = obs[1]

        quantity = np.clip(
            abs(net_demand),  # Use full net demand magnitude
            action_space.low[1],
            action_space.high[1],
        )

        MAX_PRICE = action_space.high[0]

        if net_demand > 0:
            # Strategy: Bid at the absolute maximum price ($10.00) to guarantee being the highest bid
            price = MAX_PRICE
        elif net_demand < 0:  # Seller (use base class's smart selling logic)
            # Strategy: Offer slightly lower than last price
            price = last_price * 0.95 - 0.1
        else:
            price = 0.0
            quantity = 0.0

        price = np.clip(price, action_space.low[0], action_space.high[0])
        return np.array([price, quantity], dtype=np.float32)
