import numpy as np
from gymnasium.spaces import Box
from custom_types import *
import random
import pandas as pd
from loguru import logger

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
        load: [Job],
        flexible_load: float,
        fixed_load: float,
        generation_capacity: float,
        cost_per_unit: float,
        margin: float,
        generation_type: str = "solar",
    ):
        self.agent_id = agent_id
        self.load = load  # energy demand must be met entirely or not. can move according to the lambda in the job
        self.fixed_load = fixed_load
        self.flexible_load = flexible_load  # can be assigned anywhere in any amount
        self.generation_capacity = generation_capacity
        self.generation_type = generation_type
        self.last_bid_offer: tuple[float, float] | None = None  # (price, quantity)
        self.profit = 0.0
        self.profit_margin = margin
        self.cost_per_unit = cost_per_unit
        self.price_per_unit = (1 + self.profit_margin) * self.cost_per_unit
        self.schedule = [
            sum([job[0] for job in load if job[1] == t])
            + flexible_load / FORECAST_HORIZON
            + fixed_load
            for t in range(FORECAST_HORIZON)
        ]  # current schedule to buy energy. Change this to change behaviour
        self.net_demand = None

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

    def calculate_profit(
        self,
        clearing_price: float,
        qty_got: float,
        buy_tariff: float,
        sell_tariff: float,
    ) -> float:
        """Calculates profit based on the last action submitted for the cleared hour."""
        profit = 0.0
        if self.last_bid_offer:
            price_submitted, qty_submitted = self.last_bid_offer
            if self.net_demand > 0 and price_submitted >= clearing_price:
                profit = (price_submitted - clearing_price - buy_tariff) * qty_got
            elif self.net_demand < 0 and price_submitted <= clearing_price:
                profit = (clearing_price - price_submitted - sell_tariff) * qty_got
        self.profit = profit
        # TODO: handle case where you were not cleared in the auction,
        # i.e your submitted bid is lower than clearing price or
        # your ask was more than clearing price.
        return profit

    def devise_strategy(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        """
        [DEFAULT STRATEGY]
        Defines the agent's 24-hour action plan based on its current observation.

        Returns:
            np.ndarray: The chosen action plan of shape (24, 2) -> [[P_0, Q_0], [P_1, Q_1], ...].
        """
        # Obs: [ND_i, P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1, P_f_1, ..., P_f_23]

        info = obs[FORECAST_HORIZON : FORECAST_HORIZON + 4]
        prediction = [obs[FORECAST_HORIZON + 5 :]]

        last_price = info[0]
        price_noise = 1 + (random.uniform(-0.1, 0.1) * (1 / FORECAST_HORIZON))

        # if info[2] == 0 and info[3] == 0:
        zero = np.zeros(FORECAST_HORIZON)
        bids = np.where(self.net_demand >= zero, self.net_demand, zero)
        offers = np.where(self.net_demand < zero, self.net_demand, zero)
        bids = [
            ((last_price * 1.05 - 0.1) * price_noise**i, quantity)
            for i, quantity in enumerate(bids)
        ]
        offers = [
            ((last_price * 0.95 - 0.1) * price_noise**i, np.abs(quantity))
            for i, quantity in enumerate(offers)
        ]
        self.net_demand = self.net_demand[1:] + self.net_demand[:1]
        self.schedule = self.schedule[1:] + self.schedule[:1]
        return np.array([bids, offers], dtype=np.float32)

        # actions = ]
        #
        # prices = np.zeros(FORECAST_HORIZON)
        # # --- Simple Strategy Logic for 24 hours ---
        # for h in range(FORECAST_HORIZON):
        #     self.calculate_net_demand(h)
        #     # Price: Adjust price relative to the last clearing price, with slight variation
        #     price_noise = 1 + (random.uniform(-0.1, 0.1) * (h / FORECAST_HORIZON))
        #
        #     if net_demand > 0:  # Buyer
        #         price = (last_price * 0.95 - 0.1) * price_noise
        #     elif net_demand < 0:  # Seller
        #         price = (last_price * 1.05 + 0.1) * price_noise
        #     else:
        #         price = 0.0
        #
        #     prices[h] = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
        #
        # quantity = np.clip(
        #     abs(net_demand), action_space.low[h, 1], action_space.high[h, 1]
        # )

        # actions.append([price, quantity])
        bids = np.ones((FORECAST_HORIZON, 2))
        offers = np.ones((FORECAST_HORIZON, 2))
        x = np.array([bids, offers], dtype=np.float32)
        return x

    def devise_strategy_smarter(
        self, obs: dict[str, np.ndarray], action_space: Box
    ) -> np.ndarray:
        """
        [ALTERNATIVE STRATEGY]
        Strategy: price-responsive flexible prosumer.
        Shifts flexible load to cheaper forecast hours and sets bid/offer prices relative to last known clearing price.
        """

        # Observation breakdown: observe forecast prices for next 24h
        # last_price = obs[FORECAST_HORIZON]
        last_price = obs["market_stats"][0]  # last market clearing price
        # forecast_prices = obs[FORECAST_HORIZON+4:]
        forecast_prices = obs["price_forecast"]

        forecast_prices = np.append(
            forecast_prices, forecast_prices[-1] * (1 + random.uniform(-0.1, 0.1))
        )
        # np.array(obs[FORECAST_HORIZON:]))[:FORECAST_HORIZON]  # next 24h price forecast

        # First shift flexible load to low-price hours
        sorted_hours = np.argsort(forecast_prices)  # cheapest -> most expensive
        flex_load = self.flexible_load
        new_schedule = np.array(
            [
                sum([job[0] for job in self.load if job[1] == t])
                for t in range(FORECAST_HORIZON)
            ],
            dtype=float,
        )

        # Allocate flexible load to cheapest 25% of hours
        cheap_hours = sorted_hours[: FORECAST_HORIZON // 4]
        for h in cheap_hours:
            new_schedule[h] += flex_load / len(cheap_hours)

        new_schedule = new_schedule + self.fixed_load

        self.schedule = new_schedule.tolist()

        # Secondly compute net demand profile
        self.calculate_net_demand()

        # Then generate bids/offers
        bids = np.zeros((FORECAST_HORIZON, 2))
        offers = np.zeros((FORECAST_HORIZON, 2))
        for h in range(FORECAST_HORIZON):
            nd = self.net_demand[h]
            base_price = forecast_prices[h]
            price_noise = 1 + random.uniform(-0.05, 0.05)

            if nd > 0:  # needs to buy
                price = base_price * price_noise * (1 + 0.1 * (1.01) ** (-nd))
                price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
                qty = np.clip(nd, action_space.low[h, 1], action_space.high[h, 1])
                bids[h] = [price, qty]
                offers[h] = [0, 0]
            elif nd < 0:  # has surplus to sell
                price = base_price * price_noise * (1 - 0.1 * (0.99) ** nd)
                price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
                qty = np.clip(abs(nd), action_space.low[h, 1], action_space.high[h, 1])
                price = max(price, self.price_per_unit*qty)
                offers[h] = [price, qty]
                bids[h] = [0, 0]

        self.step()

        return np.array([bids, offers], dtype=np.float32)

    def step(self):
        load = self.load
        new_load = [(job[0], job[1] - 1, job[2]) for job in load]
        self.load = new_load
        self.schedule = self.schedule[1:] + [0]

    def optimize_schedule(self, obs: np.ndarray) -> None:
        self.schedule = []


class AggressiveSellerAgent(ProsumerAgent):
    """Aggressive seller: sells entire surplus at minimum price for all 24 hours."""

    def devise_strategy_smarter(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        return super().devise_strategy_smarter(obs, action_space)
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

    def devise_strategy_smarter(self, obs: np.ndarray, action_space: Box) -> np.ndarray:
        return super().devise_strategy_smarter(obs, action_space)
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
