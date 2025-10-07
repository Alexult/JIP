import numpy as np
from gymnasium.spaces import Box
from custom_types import *
import random

FORECAST_HORIZON = 24


class ProsumerAgent:
    """
    Represents a prosumer (producer + consumer) agent.
    Manages state, profit calculation, and defines a default strategy for a 24-hour horizon.
    """

    def __init__(
        self,
        agent_id: int,
        fixed_load: float,
        flexible_load_max: float,
        generation_capacity: float,
        generation_type: str = "solar",
    ):
        self.agent_id = agent_id
        self.fixed_load = fixed_load
        self.flexible_load_max = flexible_load_max
        self.generation_capacity = generation_capacity
        self.net_demand = 0.0
        self.last_bid_offer: tuple[float, float] | None = (
            None  # (price, quantity) for the *cleared* timestep
        )
        self.profit = 0.0
        self.generation_type = generation_type

    def _calc_solar_generation(self, hour_of_day: int):
        # Model a simple solar generation curve (peaks at noon, zero at night)
        effective_generation = 0.0
        # Assume sun is out from 6:00 (hour 6) to 18:00 (hour 18)
        if 6 <= hour_of_day <= 18:
            # Use a sine curve to model the rise and fall of solar power
            # The argument to sin() goes from 0 to pi over the 12 daylight hours
            peak_hour = 12
            daylight_duration = 12
            radians = (hour_of_day - 6) * np.pi / daylight_duration
            solar_factor: float = np.sin(radians)
            effective_generation = self.generation_capacity * solar_factor
        return effective_generation

    def _calc_wind_generation(self, hour_of_day: int) -> float:
        """
        Models a simple onshore wind generation curve using a cosine function
        (peaks at midnight, troughs at noon).

        Args:
            hour_of_day (int): The hour (0 to 23).

        Returns:
            float: The effective generation (in capacity units, e.g., MW).
        """
        radians: float = hour_of_day * 2 * np.pi / 24

        capacity_factor: float = 0.425 + 0.225 * np.cos(radians)

        # Effective generation is Capacity * CF
        effective_generation = self.generation_capacity * capacity_factor

        return effective_generation

    def calculate_net_demand(self, timestep: int):
        """
        Simulates the agent's internal state, with solar generation varying by hour.
        """
        # Determine the hour of the day (0-23) from the simulation timestep
        hour_of_day = timestep % 24
        effective_generation = (
            self._calc_solar_generation(hour_of_day)
            if self.generation_type == "solar"
            else self._calc_wind_generation(hour_of_day)
        )
        current_flexible_load = random.uniform(0, self.flexible_load_max)
        total_load = self.fixed_load + current_flexible_load
        self.net_demand = total_load - effective_generation

    def get_market_submission(
        self, price: float, quantity: float
    ) -> tuple[list[Bid], list[Offer]]:
        """
        Takes a single action (price, quantity) and translates it into a bid or offer.
        Also stores this as the last submitted action for profit calculation.
        """
        bids: list[Bid] = []
        offers: list[Offer] = []
        quantity = max(0.0, quantity)

        # Store the action for the current hour for profit calculation
        self.last_bid_offer = (price, quantity)

        if self.net_demand > 0:
            bids.append((self.agent_id, price, quantity))
        elif self.net_demand < 0:
            offers.append((self.agent_id, price, quantity))

        return bids, offers

    def calculate_profit(self, clearing_price: float) -> float:
        """Calculates profit based on the last action submitted for the cleared hour."""
        profit = 0.0
        if self.last_bid_offer:
            price_submitted, qty_submitted = self.last_bid_offer
            if self.net_demand > 0 and price_submitted >= clearing_price:
                profit = (price_submitted - clearing_price) * qty_submitted
            elif self.net_demand < 0 and price_submitted <= clearing_price:
                profit = (clearing_price - price_submitted) * qty_submitted
        self.profit = profit
        return profit

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        """
        [DEFAULT STRATEGY]
        Defines the agent's 24-hour action plan based on its current observation.

        Returns:
            np.ndarray: The chosen action plan of shape (24, 2) -> [[P_0, Q_0], [P_1, Q_1], ...].
        """
        # Obs: [ND_i, P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1, P_f_1, ..., P_f_23]
        net_demand = obs[0]
        last_price = obs[1]

        actions = []

        # --- Simple Strategy Logic for 24 hours ---
        for h in range(FORECAST_HORIZON):
            quantity = np.clip(
                abs(net_demand), action_space.low[h, 1], action_space.high[h, 1]
            )

            # Price: Adjust price relative to the last clearing price, with slight variation
            price_noise = 1 + (random.uniform(-0.1, 0.1) * (h / FORECAST_HORIZON))

            if net_demand > 0:  # Buyer
                price = (last_price * 1.05 + 0.1) * price_noise
            elif net_demand < 0:  # Seller
                price = (last_price * 0.95 - 0.1) * price_noise
            else:
                price, quantity = 0.0, 0.0

            price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
            actions.append([price, quantity])

        return np.array(actions, dtype=np.float32)


class AggressiveSellerAgent(ProsumerAgent):
    """Aggressive seller: sells entire surplus at minimum price for all 24 hours."""

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        net_demand = obs[0]
        last_price = obs[1]

        quantity = np.clip(
            abs(net_demand), action_space.low[0, 1], action_space.high[0, 1]
        )

        if net_demand > 0:  # Buyer (use base logic)
            price = last_price * 1.05 + 0.1
        elif net_demand < 0:  # Seller (Aggressive)
            price = 0.01  # Offer at absolute minimum
        else:
            price, quantity = 0.0, 0.0

        price = np.clip(price, action_space.low[0, 0], action_space.high[0, 0])

        # Create a (24, 2) array by repeating the same action 24 times
        action_plan = np.tile([price, quantity], (FORECAST_HORIZON, 1))
        return action_plan.astype(np.float32)


class AggressiveBuyerAgent(ProsumerAgent):
    """Aggressive buyer: buys to cover deficit at maximum price for all 24 hours."""

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        net_demand = obs[0]
        last_price = obs[1]
        MAX_PRICE = action_space.high[0, 0]

        quantity = np.clip(
            abs(net_demand), action_space.low[0, 1], action_space.high[0, 1]
        )

        if net_demand > 0:  # Buyer (Aggressive)
            price = MAX_PRICE  # Bid at absolute maximum
        elif net_demand < 0:  # Seller (use base logic)
            price = last_price * 0.95 - 0.1
        else:
            price, quantity = 0.0, 0.0

        price = np.clip(price, action_space.low[0, 0], action_space.high[0, 0])

        # Create a (24, 2) array by repeating the same action 24 times
        action_plan = np.tile([price, quantity], (FORECAST_HORIZON, 1))
        return action_plan.astype(np.float32)
