import numpy as np
import scipy.optimize as sc
from gymnasium.spaces import Box
from custom_types import *
import random
import pandas as pd

FORECAST_HORIZON = 24


class ProsumerAgent:
    """
    Represents a prosumer (producer + consumer) agent.
    Manages state, profit calculation, and defines a default strategy for a 24-hour horizon.
    cost_per_unit: the cost per unit of producing their solar/wind energy
    margin: how much profit margin the producer needs to make minimum [0,1] corrosponds to 0%-100%
    """

    def __init__(
            self,
            agent_id: int,
            load: list[float],
            flexibility: float,
            generation_capacity: float,
            cost_per_unit: float,
            margin: float,
            generation_type: str = "solar",
    ):
        self.agent_id = agent_id
        self.load = load  # energy demand must be met entirely or not. can move according to the lambda in the job
        self.generation_capacity = generation_capacity
        self.generation_type = generation_type
        self.last_bid_offer: tuple[float, float] | None = None  # (price, quantity)
        self.profit = 0.0
        self.flexibility = flexibility
        self.schedule = [l for l in load]  # current schedule to buy energy. Change this to change behaviour
        self.profit_margin = margin
        self.cost_per_unit = cost_per_unit
        self.price_per_unit = (1 + self.profit_margin) * self.cost_per_unit
        self.net_demand = None
        self.total_energy = sum(self.schedule)

        generation_data_file = "./data/hourly_wind_solar_data.csv"
        df = pd.read_csv(generation_data_file)
        self.solar_data = df["Solar - Actual Aggregated [MW] (D)"].to_numpy()
        self.wind_data = df["Wind Onshore - Actual Aggregated [MW] (D)"].to_numpy()
        self.multiplicative_factor = [
            self.generation_capacity / self.solar_data.max(),
            self.generation_capacity / self.wind_data.max(),
        ]
        del df
        del generation_data_file

    def _calc_solar_generation(self, hour_of_day: int):
        effective_generation = (
                self.solar_data[hour_of_day] * self.multiplicative_factor[0]
        )

        return effective_generation

    def _calc_wind_generation(self, hour_of_day: int) -> float:
        effective_generation = (
                self.wind_data[hour_of_day] * self.multiplicative_factor[1]
        )
        return effective_generation

    def calculate_net_demand(self):
        """
        Calculates and updates the net_demand at the timestep
        """
        self.net_demand = [self._calculate_demand(t) for t in range(FORECAST_HORIZON)]

    def _calculate_demand(self, timestep: int) -> float:
        """
        Calculates the net_demand at the timestep
        """
        # Determine the hour of the day (0-23) from the simulation timestep
        hour_of_day = timestep % 24
        effective_generation = (
            self._calc_solar_generation(hour_of_day)
            if self.generation_type == "solar"
            else self._calc_wind_generation(hour_of_day)
        )
        return self.schedule[timestep] - effective_generation

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

    # def calculate_profit(
    #     self,
    #     clearing_price: float,
    #     qty_got: float,
    #     buy_tariff: float,
    #     sell_tariff: float,
    # ) -> float:
    #     """Calculates profit based on the last action submitted for the cleared hour."""
    #     profit = 0.0
    #     if self.last_bid_offer:
    #         price_submitted, qty_submitted = self.last_bid_offer
    #         if self.net_demand > 0 and price_submitted >= clearing_price:
    #             profit = (price_submitted - clearing_price - buy_tariff) * qty_got
    #         elif self.net_demand < 0 and price_submitted <= clearing_price:
    #             profit = (clearing_price - price_submitted - sell_tariff) * qty_got
    #     self.profit = profit
    #     # TODO: handle case where you were not cleared in the auction,
    #     # i.e your submitted bid is lower than clearing price or
    #     # your ask was more than clearing price.
    #     return profit

    def devise_strategy(self, obs: dict[str, np.ndarray], action_space: Box, buy_tariff=0.23, sell_tariff=0.10,
                        ) -> np.ndarray:
        """
        Strategy: price-responsive flexible prosumer.
        Shifts flexible load to cheaper forecast hours and sets bid/offer prices relative to last known clearing price.
        """
        forecast_prices = obs["price_forecast"]

        buy_prices = np.append(
            forecast_prices, forecast_prices[-1]
        )

        buy_prices = [buy_prices[h] if buy_prices[h] > 0 else buy_prices[h - 1] for h in range(FORECAST_HORIZON)]

        new_schedule = self.schedule

        def objective(x: list) -> float:
            return sum(i[0] * i[1] for i in zip(x, buy_prices))

        x0 = new_schedule
        row = [1] + [0] * (FORECAST_HORIZON - 1)
        # A = [row[-i:] + row[:-i] for i in range(FORECAST_HORIZON)] + [[1] * FORECAST_HORIZON]
        bnds = [(0, action_space.high[0, 1]) for i in range(FORECAST_HORIZON)]
        # ub = [action_space.high[0,1]] * FORECAST_HORIZON + [self.total_energy]
        # lb = [0] * FORECAST_HORIZON + [self.total_energy]
        # constraint = sc.LinearConstraint(A, lb, ub)
        cons = {"type": "eq", "fun": lambda x: sum(x) - self.total_energy}
        sol = sc.minimize(objective, x0, bounds=bnds, constraints=cons)

        a = 0.4
        y = [self.schedule[i] - sol.get("x")[i] for i in range(FORECAST_HORIZON)]
        new_schedule = [a * sol.get("x")[i] + (1 - a) * self.schedule[i] for i in range(FORECAST_HORIZON)]
        z = [self.schedule[i] - new_schedule[i] for i in range(FORECAST_HORIZON)]
        self.schedule = new_schedule

        # Secondly compute net demand profile
        self.calculate_net_demand()

        # Then generate bids/offers
        bids = np.zeros((FORECAST_HORIZON, 2))
        offers = np.zeros((FORECAST_HORIZON, 2))
        for h in range(FORECAST_HORIZON):
            nd = self.net_demand[h]

            price_noise = 1 + random.uniform(-0.05, 0.05)

            if nd > 0:  # needs to buy
                base_price = buy_prices[h]
                price = base_price * price_noise * (1 + 0.1 * (1.01) ** (-nd))
                price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
                qty = np.clip(nd, action_space.low[h, 1], action_space.high[h, 1])
                bids[h] = [price, qty]
                offers[h] = [0, 0]
            elif nd < 0:  # has surplus to sell
                price = self.price_per_unit
                price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
                qty = np.clip(abs(nd), action_space.low[h, 1], action_space.high[h, 1])
                offers[h] = [price, qty]
                bids[h] = [0, 0]

        self.step()

        return np.array([bids, offers], dtype=np.float32)

    def step(self):
        self.schedule = self.schedule[1:] + [0]


class AggressiveSellerAgent(ProsumerAgent):
    """Aggressive seller: sells entire surplus at minimum price for all 24 hours."""

    def devise_strategy(self, obs: np.ndarray, action_space: Box, buy_tariff=0.23, sell_tariff=0.10,
                        ) -> np.ndarray:
        return super().devise_strategy(obs, action_space)
        # net_demand = obs[0]
        # last_price = obs[1]
        #
        # quantity = np.clip(
        #     abs(net_demand), action_space.low[0, 1], action_space.high[0, 1]
        # )
        #
        # if net_demand > 0:  # Buyer (use base logic)
        #     price = last_price * 0.95 - 0.1
        # elif net_demand < 0:  # Seller (Aggressive)
        #     price = 0.01  # Offer at absolute minimum
        # else:
        #     price, quantity = 0.0, 0.0
        #
        # price = np.clip(price, action_space.low[0, 0], action_space.high[0, 0])
        #
        # # Create a (24, 2) array by repeating the same action 24 times
        # action_plan = np.tile([price, quantity], (FORECAST_HORIZON, 1))
        # return action_plan.astype(np.float32)

        # bids = np.zeros((FORECAST_HORIZON, 2))
        # offers = np.zeros((FORECAST_HORIZON, 2))
        # x = np.array([bids, offers], dtype=np.float32)
        # return x


class AggressiveBuyerAgent(ProsumerAgent):
    """Aggressive buyer: buys to cover deficit at maximum price for all 24 hours."""

    def devise_strategy(self, obs: np.ndarray, action_space: Box, buy_tariff=0.23,
                        sell_tariff=0.10,
                        ) -> np.ndarray:
        return super().devise_strategy(obs, action_space)
        # net_demand = obs[0]
        # last_price = obs[1]
        # MAX_PRICE = action_space.high[0, 0]
        #
        # quantity = np.clip(
        #     abs(net_demand), action_space.low[0, 1], action_space.high[0, 1]
        # )
        #
        # if net_demand > 0:  # Buyer (Aggressive)
        #     price = MAX_PRICE  # Bid at absolute maximum
        # elif net_demand < 0:  # Seller (use base logic)
        #     price = last_price * 0.95 - 0.1
        # else:
        #     price, quantity = 0.0, 0.0
        #
        # price = np.clip(price, action_space.low[0, 0], action_space.high[0, 0])
        #
        # # Create a (24, 2) array by repeating the same action 24 times
        # action_plan = np.tile([price, quantity], (FORECAST_HORIZON, 1))
        # return action_plan.astype(np.float32)
        # bids = np.zeros((FORECAST_HORIZON, 2))
        # offers = np.zeros((FORECAST_HORIZON, 2))
        # x = np.array([bids, offers], dtype=np.float32)
        # return x
