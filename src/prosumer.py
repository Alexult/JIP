import numpy as np
import scipy.optimize as sc
from gymnasium.spaces import Box
from custom_types import *
import random
import pandas as pd
from loguru import logger
import os

FORECAST_HORIZON = 10

PRICE_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "representative_wholesale_price_2025.csv",
)


def load_data(csv_path: str):
    """Return dict {day_str: DataFrame(hour, price)} for each calendar day in file."""
    df = pd.read_csv(csv_path)
    p_col = "Price (EUR/MWhe)"


    days = {}
    for d, sub in df.groupby("Day"):
        sub = sub.sort_values("Hour_of_Day")[["Hour_of_Day", p_col]].reset_index(drop=True)
        if len(sub) != 24:
            print(
                f"Warning: day {d} has {len(sub)} rows (expected 24). Using what's available."
            )
        days[str(d)] = sub
    return days


NATIONAL_MARKET_DATA = load_data(PRICE_CSV_PATH)


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
        generation_capacity: float,
        marginal_price: float,
        generation_type: str = "solar",
    ):
        self.agent_id = agent_id
        self.load = load  # energy demand must be met entirely or not. can move according to the lambda in the job
        self.generation_capacity = generation_capacity
        self.generation_type = generation_type
        self.last_bid_offer: tuple[float, float] | None = None  # (price, quantity)
        self.costs = [0.0] * len(self.load)
        self.schedule = [
            l for l in load
        ]  # current schedule to buy energy. Change this to change behaviour
        self.marginal_price = marginal_price
        self.net_demand = [0] * FORECAST_HORIZON
        self.total_energy = sum(self.schedule[0:FORECAST_HORIZON])
        self.national_consumption = [0.0] * len(self.load)

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

        self.calculate_net_demand(0)

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

    def calculate_net_demand(self, time_step: int):
        """
        Calculates and updates the net_demand at the timestep
        """
        self.net_demand = [
            self._calculate_demand(t + time_step)
            for t in range(len(self.load[time_step : time_step + FORECAST_HORIZON]))
        ]

    def _calculate_demand(self, timestep: int) -> float:
        """
        Calculates the net_demand at the timestep
        """
        # Determine the hour of the day (0-23) from the simulation timestep
        hour_of_day = timestep % 24
        effective_generation = 0
        if self.generation_type == "solar":
            effective_generation = self._calc_solar_generation(hour_of_day)
        elif self.generation_type == "wind":
            effective_generation = self._calc_wind_generation(hour_of_day)
        return self.schedule[timestep] - effective_generation

    def get_demand_consumption(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_steps = len(self.load)

        if self.generation_type == "solar":
            pattern = self.solar_data
            factor = self.multiplicative_factor[0]
        elif self.generation_type == "wind":
            pattern = self.wind_data
            factor = self.multiplicative_factor[1]
        else:
            pattern = np.zeros(24)
            factor = 1.0

        # Tile the pattern to fill time_steps, then truncate
        supply = np.tile(pattern, time_steps // len(pattern) + 1)[:time_steps] * factor

        initial_net_demand = np.array(self.load) - supply
        actual_net_demand = np.array(self.schedule) - supply

        return initial_net_demand, actual_net_demand, supply

    def purchase_from_national_market(
        self, qty_got: float, bid_price: float, bid_qty: float, timestep: int
    ):
        t = timestep % 24
        price = NATIONAL_MARKET_DATA.get("Day_1").iloc[t,1]
        qty = 0
        if bid_qty > 0:
            if bid_price >= price:
                qty = qty_got - bid_qty
        else:
            if bid_price <= price:
                qty = bid_qty - qty_got

        cost: float = price * qty
        self.national_consumption[timestep] = qty
        self.costs[timestep] += -cost
        return cost

    def devise_strategy(
        self, obs: dict[str, np.ndarray], action_space: Box, timestep: int
    ) -> np.ndarray:
        """
        Strategy: price-responsive flexible prosumer.
        Shifts flexible load to cheaper forecast hours and sets bid/offer prices relative to last known clearing price.
        """
        forecast_prices = obs["price_forecast"]

        new_schedule = self.schedule
        x0 = new_schedule[timestep : timestep + FORECAST_HORIZON]
        size = len(x0)

        if size == FORECAST_HORIZON and timestep > 0:
            # if obs["agent_state"][0] < 0:
            self.total_energy = (
                self.total_energy
                + self.schedule[timestep + FORECAST_HORIZON - 1]
                - self.schedule[timestep - 1]
            )
            # else:
            #     self.total_energy = (self.total_energy + self.schedule[timestep + FORECAST_HORIZON - 1]
            #                          - obs["agent_state"][0])
            #     self.schedule[timestep-1] = obs["agent_state"][0]
        elif timestep > 0:
            # if obs["agent_state"][-1] < 0:
            self.total_energy = self.total_energy - self.schedule[timestep - 1]
            # else:
            #     self.total_energy = self.total_energy - obs["agent_state"][0]
            #     self.schedule[timestep-1] = obs["agent_state"][0]

        buy_prices = np.append(forecast_prices, forecast_prices[-1])

        buy_prices = [
            buy_prices[h] if buy_prices[h] > 0 else buy_prices[h - 1]
            for h in range(size)
        ]

        def objective(x: list) -> float:
            return sum(
                i[0] * (i[1] + obs["agent_state"][1])
                if i[0] >= 0
                else i[0] * -(i[1] - obs["agent_state"][2])
                for i in zip(x, buy_prices)
            )

        bnds = [(0, action_space.high[0, 1]) for i in range(size)]
        cons = {"type": "eq", "fun": lambda x: sum(x) - self.total_energy}
        sol = sc.minimize(objective, x0, bounds=bnds, constraints=cons)

        new_schedule = np.concatenate(
            (new_schedule[:timestep], sol.get("x"), new_schedule[timestep + size :])
        )

        a = 0.03
        # y = [self.schedule[i] - sol.get("x")[i] for i in range(FORECAST_HORIZON)]
        new_schedule[timestep:] = (
            a * np.array(new_schedule)[timestep:]
            + (1 - a) * np.array(self.schedule)[timestep:]
        )
        z = [val - new_schedule[i] for i, val in enumerate(self.schedule)]
        self.schedule = new_schedule

        # Secondly compute net demand profile
        self.calculate_net_demand(timestep)

        # Then generate bids/offers
        bids = np.zeros((FORECAST_HORIZON, 2))
        offers = np.zeros((FORECAST_HORIZON, 2))
        for h in range(size):
            nd = self.net_demand[h]

            price_noise = 1 + random.uniform(-0.05, 0.05)

            if nd > 0:  # needs to buy
                base_price = buy_prices[h]
                price = base_price * price_noise * ((1.01) ** (nd))
                price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
                qty = np.clip(nd, action_space.low[h, 1], action_space.high[h, 1])
                bids[h] = [price, qty]
                offers[h] = [0, 0]
            elif nd < 0:  # has surplus to sell
                price = self.marginal_price
                price = np.clip(price, action_space.low[h, 0], action_space.high[h, 0])
                qty = np.clip(abs(nd), action_space.low[h, 1], action_space.high[h, 1])
                offers[h] = [price, qty]
                bids[h] = [0, 0]

        return np.array([bids, offers], dtype=np.float32)
