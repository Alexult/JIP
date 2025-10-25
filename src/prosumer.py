import numpy as np
import scipy.optimize as sc
from gymnasium.spaces import Box
from custom_types import *
import random
import pandas as pd

FORECAST_HORIZON = 10


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
        self.profit = 0.0
        self.schedule = [l for l in load]  # current schedule to buy energy. Change this to change behaviour
        self.marginal_price = marginal_price
        self.net_demand = [0] * FORECAST_HORIZON
        self.total_energy = sum(self.schedule[0:FORECAST_HORIZON])

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
        self.net_demand = [self._calculate_demand(t + time_step) for t in range(len(self.load[time_step:time_step+FORECAST_HORIZON]))]

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

    def get_demand_consumption(self) -> tuple[list[float], list[float]]:
        time_steps = len(self.load)

        supply = np.zeros(time_steps)
        if self.generation_type == "solar":
            supply = np.concatenate((np.repeat(self.solar_data, time_steps // 24),
                                        self.solar_data[:time_steps % 24])) * self.multiplicative_factor[0]
        elif self.generation_type == "wind":
            supply = np.concatenate((np.repeat(self.wind_data, time_steps // 24),
                                        self.wind_data[:time_steps % 24])) * self.multiplicative_factor[1]
        initial_net_demand = np.array(self.load) - supply
        actual_net_demand = np.array(self.schedule) - supply
        return initial_net_demand, actual_net_demand

    def devise_strategy(self, obs: dict[str, np.ndarray], action_space: Box, timestep: int, buy_tariff=0.23, sell_tariff=0.10,
                        ) -> np.ndarray:
        """
        Strategy: price-responsive flexible prosumer.
        Shifts flexible load to cheaper forecast hours and sets bid/offer prices relative to last known clearing price.
        """
        forecast_prices = obs["price_forecast"]

        new_schedule = self.schedule
        x0 = new_schedule[timestep:timestep + FORECAST_HORIZON]
        size = len(x0)

        if size == FORECAST_HORIZON and timestep > 0:
            if obs["agent_state"][-1] < 0:
                self.total_energy = self.total_energy + self.schedule[timestep+FORECAST_HORIZON-1] - self.schedule[timestep-1]
            else:
                self.total_energy = self.total_energy + self.schedule[timestep+FORECAST_HORIZON-1] - obs["agent_state"][-1]
                self.schedule[timestep-1] = obs["agent_state"][-1]
        elif timestep > 0:
            if obs["agent_state"][-1] < 0:
                self.total_energy = self.total_energy - self.schedule[timestep-1]
            else:
                self.total_energy = self.total_energy - obs["agent_state"][-1]
                self.schedule[timestep - 1] = obs["agent_state"][-1]

        buy_prices = np.append(
            forecast_prices, forecast_prices[-1]
        )

        buy_prices = [buy_prices[h] if buy_prices[h] > 0 else buy_prices[h - 1] for h
                      in range(size)]

        def objective(x: list) -> float:
            return sum(i[0] * (i[1] + buy_tariff) if i[0] >= 0 else i[0] * -(i[1] - sell_tariff) for i in zip(x, buy_prices))

        bnds = [(0, action_space.high[0, 1]) for i in range(size)]
        cons = {"type": "eq", "fun": lambda x: sum(x) - self.total_energy}
        sol = sc.minimize(objective, x0, bounds=bnds, constraints=cons)

        new_schedule = np.concatenate((new_schedule[:timestep], sol.get("x"), new_schedule[timestep + size:]))

        a = 0.2
        # y = [self.schedule[i] - sol.get("x")[i] for i in range(FORECAST_HORIZON)]
        if size == FORECAST_HORIZON:
            new_schedule = a * np.array(new_schedule) + (1 - a) * np.array(self.schedule)
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
                price = base_price * price_noise * (1 + 0.1 * (1.01) ** (-nd))
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
