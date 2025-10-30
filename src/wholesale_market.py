from typing import override, Any
import numpy as np
from custom_types import *  # Assuming custom_types.py exists
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from prosumer import ProsumerAgent  # Import the agent from prosumer.py
import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd


class WholesaleMarketEnv(Env):
    """
    Simplified wholesale market environment where agents act as price-takers.

    Agents observe the wholesale price forecast and optimize their internal
    consumption schedule (self.schedule) via their 'devise_strategy' method.

    The environment's 'step' function then executes trades for the current
    timestep (t=0) based on the agent's resulting 'net_demand[0]' at the
    fixed 'current_wholesale_price'.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        agent_configs: list[dict[str, Any]],
        buy_tariff: float,
        sell_tariff: float,
        wholesale_csv_path: str = "./data/representative_wholesale_price_2025.csv",
        max_timesteps: int = 96,
    ):
        super().__init__()
        self.buy_tariff = buy_tariff
        self.sell_tariff = sell_tariff
        self.n_agents = len(agent_configs)
        self.agent_ids = list(range(self.n_agents))
        self.max_timesteps = max_timesteps
        self.current_timestep = 0

        # Load wholesale prices from CSV
        try:
            df = pd.read_csv(wholesale_csv_path)
            ts_col = "Datetime (Local)"
            p_col = "Price (EUR/MWhe)"

            # Parse timestamps and split into days
            # df[ts_col] = pd.to_datetime(df[ts_col])
            # df["day"] = df[ts_col].dt.date
            # df["hour"] = df[ts_col].dt.hour

            days = {}
            for d, sub in df.groupby("Day"):
                sub = sub.sort_values("Hour_of_Day")[
                    ["Hour_of_Day", p_col]
                ].reset_index(drop=True)
                if len(sub) != 24:
                    print(
                        f"Warning: day {d} has {len(sub)} rows (expected 24). Using what's available."
                    )
                days[str(d)] = sub
            self.wholesale_prices = days.get("Day_4").iloc[:, 1].to_list()
        except FileNotFoundError:
            logger.error(f"Wholesale price file not found: {wholesale_csv_path}")
            logger.warning("Using default placeholder prices.")
            self.wholesale_prices = np.array(
                [50.0 + 10 * np.sin(i / 4) for i in range(max(max_timesteps, 1000))]
            )
        self.FORECAST_HORIZON = 10

        # Public Market Stats
        self.current_wholesale_price = (
            self.wholesale_prices[0] if len(self.wholesale_prices) > 0 else 50.0
        )
        self.last_total_traded_qty = 0.0
        # Track last traded qty *per agent* for observation
        self.last_agent_trades: dict[int, float] = {i: 0.0 for i in self.agent_ids}
        self.initial_net_demand_history: list[list[float]] = []  # Per agent, pre-opt
        self.total_generation_history: list[float] = []
        self.total_price_paid_history: list[float] = []
        self.cumulative_price_paid_history: list[float] = []
        # --- Initialize Agents ---
        # We only use ProsumerAgent, as bidding strategies are irrelevant
        self.agents: list[ProsumerAgent] = []
        for i, config in enumerate(agent_configs):
            if (
                "load" not in config
                or "generation_capacity" not in config
                or "marginal_price" not in config
            ):
                raise ValueError(f"Agent config {i} is missing required keys.")

            self.agents.append(
                ProsumerAgent(
                    agent_id=i,
                    load=config["load"],
                    generation_capacity=config["generation_capacity"],
                    marginal_price=config["marginal_price"],
                    generation_type=config.get("generation_type", "solar"),
                )
            )

        # --- Define Action Space (Per Agent) ---
        # This action space matches the *output* of ProsumerAgent.devise_strategy
        # even though this env doesn't use the bids/offers.
        # This ensures compatibility with the main.py loop.
        MAX_PRICE = 500.0
        MAX_QTY = 50.0  # Max qty per bid/offer
        # Create the high array with the correct structure

        low_action = np.array([0.0, 0.0], dtype=np.float32)
        high_action = np.array([MAX_PRICE, MAX_QTY], dtype=np.float32)

        self.action_space = Box(
            low=np.tile(low_action, (self.FORECAST_HORIZON, 1)),
            high=np.tile(high_action, (self.FORECAST_HORIZON, 1)),
            shape=(self.FORECAST_HORIZON, 2),
            dtype=np.float32,
        )

        # --- Define Observation Space (Per Agent) ---
        MAX_DEMAND_ABS = (
            max(max(c.schedule) for c in self.agents) if self.agents else 100.0
        )
        MAX_CAPACITY = (
            max(c.generation_capacity for c in self.agents) if self.agents else 100.0
        )
        MAX_NET_DEMAND = MAX_DEMAND_ABS
        MIN_NET_DEMAND = -MAX_CAPACITY
        MAX_TRADED_QTY = max(MAX_NET_DEMAND, abs(MIN_NET_DEMAND))
        MAX_TOTAL_TRADED = self.n_agents * MAX_TRADED_QTY
        MAX_WH_PRICE = 1000.0
        MIN_WH_PRICE = -100.0

        self.observation_space = Dict(
            {
                # Agent State: [Net Demand (t=0), Last Traded Qty (agent)]
                "agent_state": Box(
                    low=np.array([MIN_NET_DEMAND, -MAX_TRADED_QTY], dtype=np.float32),
                    high=np.array([MAX_NET_DEMAND, MAX_TRADED_QTY], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                # Market Stats: [Wholesale Price (t), Total Traded Qty (t-1)]
                "market_stats": Box(
                    low=np.array([MIN_WH_PRICE, 0.0], dtype=np.float32),
                    high=np.array([MAX_WH_PRICE, MAX_TOTAL_TRADED], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                # Price Forecast for next 23 hours (t+1 ... t+23)
                "price_forecast": Box(
                    low=MIN_WH_PRICE,
                    high=MAX_WH_PRICE,
                    shape=(23,),
                    dtype=np.float32,
                ),
            }
        )

        # History tracking
        self.wholesale_prices_history: list[float] = []
        self.traded_quantities_history: list[float] = []  # Total traded
        self.profit_history: list[np.ndarray] = []  # Per agent
        self.net_demand_history: list[list[float]] = []  # Per agent
        self.agent_trades_history: list[list[dict]] = []  # Full trade info

        logger.info(f"WholesaleMarketEnv initialized with {self.n_agents} agents")
        if len(self.wholesale_prices) > 0:
            logger.info(f"Loaded {len(self.wholesale_prices)} wholesale prices")
            logger.info(
                f"Price range: €{min(self.wholesale_prices):.2f} - €{max(self.wholesale_prices):.2f}"
            )

    def _get_wholesale_price_forecast(self) -> list[float]:
        """Get next 23 hours of wholesale prices for agent planning."""
        # Forecast for t+1 to t+23
        start_idx = self.current_timestep + 1
        end_idx = start_idx + 23

        prices = []
        if len(self.wholesale_prices) > 0:
            prices = list(self.wholesale_prices[start_idx:end_idx])
            # Pad if simulation runs longer than price data
            while len(prices) < 23:
                prices.append(self.wholesale_prices[-1])
        else:
            prices = [50.0] * 23  # Default if no data

        return prices

    def _get_obs(self) -> dict[int, dict[str, np.ndarray]]:
        """Get observations for all agents."""
        market_stats_arr = np.array(
            [
                self.current_wholesale_price,
                self.last_total_traded_qty,
            ],
            dtype=np.float32,
        )

        # Get next 23 hours of wholesale prices for forecast
        price_forecast = self._get_wholesale_price_forecast()
        price_forecast_arr = np.array(price_forecast, dtype=np.float32)

        observations: dict[int, dict[str, np.ndarray]] = {}
        for agent in self.agents:
            agent_id = agent.agent_id

            # Get agent's net demand for the current hour (t=0 of their plan)
            # This value was set by agent.devise_strategy()
            current_net_demand = agent.net_demand[0] if agent.net_demand else 0.0

            agent_state_arr = np.array(
                [current_net_demand, self.last_agent_trades[agent_id]], dtype=np.float32
            )

            observations[agent_id] = {
                "agent_state": agent_state_arr,
                "market_stats": market_stats_arr,
                "price_forecast": price_forecast_arr,
            }

        return observations

    def _calculate_rewards(self, trades: list[dict]) -> dict[int, float]:
        """Calculate rewards based on wholesale trading"""
        rewards: dict[int, float] = {agent_id: 0.0 for agent_id in self.agent_ids}

        for trade in trades:
            agent_id = trade["agent_id"]
            trade_value = trade["trade_value"]
            rewards[agent_id] = (
                trade_value  # Positive for selling (revenue), negative for buying (cost)
            )

        return rewards

    def _execute_wholesale_trading(self) -> list[dict]:
        """
        Execute wholesale trading for all agents at current wholesale price.
        This function READS the agent's state (net_demand[0]), which was
        set by the agent.devise_strategy() call in the main loop.
        """
        trades = []
        total_traded = 0.0

        # Reset per-agent trade tracker for this step
        self.last_agent_trades = {i: 0.0 for i in self.agent_ids}

        for agent in self.agents:
            # Get the net demand for the *current hour*
            # This was calculated by the agent's strategy in the main loop
            net_demand = agent.net_demand[0] if agent.net_demand else 0.0

            trade_value = 0.0
            action_type = "none"
            trade_quantity = abs(net_demand)

            # In wholesale market, agent trades their net demand/surplus at wholesale price
            if net_demand > 0:
                # Agent needs energy (buying)
                trade_value = -net_demand * (
                    self.current_wholesale_price + self.buy_tariff
                )  # Negative (cost)
                action_type = "buy"
                self.last_agent_trades[agent.agent_id] = (
                    net_demand  # Positive for bought
                )
            elif net_demand < 0:
                # Agent has surplus energy (selling)
                trade_value = -net_demand * (
                    self.current_wholesale_price - self.sell_tariff
                )  # Positive (revenue)
                action_type = "sell"
                self.last_agent_trades[agent.agent_id] = net_demand  # Negative for sold
            else:
                # No trading needed
                trade_value = 0.0
                action_type = "none"
                self.last_agent_trades[agent.agent_id] = 0.0

            trade_info = {
                "agent_id": agent.agent_id,
                "net_demand": net_demand,
                "trade_quantity": trade_quantity,
                "trade_value": trade_value,  # Cost (-) or Revenue (+)
                "wholesale_price": self.current_wholesale_price,
                "action_type": action_type,
            }

            trades.append(trade_info)
            total_traded += trade_quantity

        self.last_total_traded_qty = total_traded
        return trades

    @override
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_timestep = 0
        self.current_wholesale_price = (
            self.wholesale_prices[0] if len(self.wholesale_prices) > 0 else 50.0
        )
        self.last_total_traded_qty = 0.0
        self.last_agent_trades = {i: 0.0 for i in self.agent_ids}

        # Reset history
        self.wholesale_prices_history = []
        self.traded_quantities_history = []
        self.profit_history = []
        self.net_demand_history = []
        self.agent_trades_history = []
        self.initial_net_demand_history = []
        self.total_generation_history = []
        self.total_price_paid_history = []
        self.cumulative_price_paid_history = []
        # Reset agents and calculate initial net demand
        for agent in self.agents:
            agent.calculate_net_demand(
                0
            )  # Let agents calculate their t=0 to t=23 demand
            agent.costs = [0] * len(agent.load)  # Reset agent's internal cost tracker

        observation = self._get_obs()
        info = {"timestep": self.current_timestep}
        return observation, info

    def step(
        self, actions: dict[int, Any]
    ) -> tuple[
        dict[int, dict[str, np.ndarray]],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[str, Any],
    ]:
        """
        Step the environment forward.

        NOTE: The 'actions' parameter is ignored. The main.py loop calls
        agent.devise_strategy() which updates the agent's internal 'self.schedule'
        and 'self.net_demand'. This 'step' function simply reads that
        updated state and executes the trades for t=0.
        """

        # Execute wholesale trading based on agent's current state
        # (which was set by devise_strategy in the main loop)
        trades = self._execute_wholesale_trading()

        # Calculate rewards
        rewards = self._calculate_rewards(trades)

        # Update agent's internal cost tracker
        for trade in trades:
            agent_id = trade["agent_id"]
            # trade_value is cost (negative) or revenue (positive)
            # agent.costs expects costs to be positive numbers
            cost = -trade["trade_value"]
            self.agents[agent_id].costs[self.current_timestep] = cost

        # Update history
        self.wholesale_prices_history.append(self.current_wholesale_price)
        self.traded_quantities_history.append(self.last_total_traded_qty)
        self.profit_history.append(np.array(list(rewards.values())))
        self.net_demand_history.append([agent.net_demand[0] for agent in self.agents])
        self.agent_trades_history.append(trades)
        total_gen_t = 0
        initial_nd_list_t = []
        hour_of_day = self.current_timestep % 24

        for agent in self.agents:
            gen = 0
            if agent.generation_type == "solar":
                gen = agent._calc_solar_generation(hour_of_day)
            elif agent.generation_type == "wind":
                gen = agent._calc_wind_generation(hour_of_day)
            total_gen_t += gen

            # Use agent.load[t] for the *original* inflexible load
            if self.current_timestep < len(agent.load):
                initial_nd_list_t.append(agent.load[self.current_timestep] - gen)
            else:
                initial_nd_list_t.append(0 - gen)  # Assume 0 load if out of bounds

        self.total_generation_history.append(total_gen_t)
        self.initial_net_demand_history.append(initial_nd_list_t)

        # Calculate total price paid by *buyers only*
        total_price_paid_t = 0
        for trade in trades:
            total_price_paid_t += -trade[
                "trade_value"
            ]  # trade_value is negative for cost

        self.total_price_paid_history.append(total_price_paid_t)

        # Calculate cumulative price paid
        cumulative = (
            self.cumulative_price_paid_history[-1]
            if self.cumulative_price_paid_history
            else 0.0
        ) + total_price_paid_t
        self.cumulative_price_paid_history.append(cumulative)
        # Update timestep and price for the *next* step
        self.current_timestep += 1
        is_truncated = self.current_timestep >= self.max_timesteps

        if not is_truncated and self.current_timestep < len(self.wholesale_prices):
            self.current_wholesale_price = self.wholesale_prices[self.current_timestep]
        elif len(self.wholesale_prices) > 0:
            self.current_wholesale_price = self.wholesale_prices[
                -1
            ]  # Use last known price
        else:
            self.current_wholesale_price = 50.0  # Default

        terminated = {i: False for i in self.agent_ids}
        truncated = {i: is_truncated for i in self.agent_ids}

        # Get observations for the *next* state
        observation = self._get_obs()
        info = {"timestep": self.current_timestep, "trades": trades}

        return (
            observation,
            rewards,
            terminated,
            truncated,
            info,
        )

    def render(self, mode="human"):
        """Print current state"""
        if mode == "human":
            last_total_reward = (
                self.profit_history[-1].sum() if self.profit_history else 0.0
            )
            logger.info(
                f"T={self.current_timestep:02d} | Wholesale Price=€{self.current_wholesale_price:.2f} | "
                f"Total Traded={self.last_total_traded_qty:.2f} MWh | Step Rewards=€{last_total_reward:.2f}"
            )

    def plot_results(self):
        """Generate plots for wholesale market results"""
        if not self.profit_history:
            logger.warning("No history to plot. Run a simulation first.")
            return
        self.plot_consumption_and_costs()
        timesteps = range(1, len(self.wholesale_prices_history) + 1)

        # Plot 1: Wholesale Price and Total Traded Quantity Over Time
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(
            timesteps,
            self.wholesale_prices_history,
            marker="o",
            linestyle="-",
            color="orange",
        )
        plt.title("Wholesale Price Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Price (€/MWh)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            timesteps,
            self.traded_quantities_history,
            marker="o",
            linestyle="-",
            color="purple",
        )
        plt.title("Total Traded Quantity Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Quantity (MWh)")
        plt.grid(True)
        plt.tight_layout()

        # Plot 2: Cumulative Profit/Loss per Agent
        plt.figure(figsize=(10, 6))
        profit_matrix = np.array(self.profit_history)

        for i in range(self.n_agents):
            cumulative_profit = np.cumsum(profit_matrix[:, i])
            # Simple coloring by agent ID
            plt.plot(
                timesteps,
                cumulative_profit,
                marker=".",
                linestyle="-",
                label=f"Agent {i}",
                alpha=0.7,
            )

        plt.title("Cumulative Profit/Loss per Agent (Wholesale Market)")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Profit/Loss (€)")
        plt.grid(True)
        if self.n_agents <= 20:  # Only show legend if not too crowded
            plt.legend()
        plt.tight_layout()

        plt.show()

    def plot_price_change_for_single_day(self, day: int = 0):
        """Plot wholesale price for each hour of a specific day"""
        plt.figure(figsize=(10, 5))
        start_index, end_index = day * 24, (day + 1) * 24

        if self.current_timestep < start_index:
            print(f"Simulation has not reached day {day}. No data to plot.")
            return

        # Ensure we don't try to plot data that hasn't happened yet
        end_index = min(end_index, len(self.wholesale_prices_history))
        daily_prices = self.wholesale_prices_history[start_index:end_index]
        hours = range(len(daily_prices))

        if not daily_prices:
            print(f"No price history found for Day {day}.")
            return

        plt.plot(hours, daily_prices, marker="o", linestyle="-", color="orange")
        plt.title(f"Wholesale Price vs. Hour for Day {day}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Wholesale Price (€/MWh)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(0, 24, 2))
        plt.xlim(-0.5, 23.5)
        plt.tight_layout()
        plt.show()

    def plot_consumption_and_costs(self, return_only_data=False):
        """
        Plots:
        1) total preferred (initial) consumption vs total actual (optimized) consumption over time
        2) total price paid per timestep (by buyers)
        3) cumulative total price paid over time (by buyers)
        4) total generation over time
        """
        T = len(self.wholesale_prices_history)
        if T == 0:
            logger.warning("No history to plot (run an episode first).")
            return
        timesteps = list(range(1, T + 1))

        initial_net_demand = np.zeros(T)
        actual_net_demand = np.zeros(T)
        total_generation = np.zeros(T)
        for agent in self.agents:
            init, actual, supply = agent.get_demand_consumption()
            logger.debug(f"init size: {len(init)}")
            initial_net_demand = [
                initial_net_demand[i] + val for i, val in enumerate(init)
            ]
            actual_net_demand = [
                actual_net_demand[i] + val for i, val in enumerate(actual)
            ]
            total_generation = [
                total_generation[i] + val for i, val in enumerate(supply)
            ]
        if return_only_data:
            return initial_net_demand, actual_net_demand, total_generation

        # Plot 1: preferred vs actual consumption
        plt.figure(figsize=(10, 5))
        plt.plot(
            timesteps,
            initial_net_demand,
            marker="o",
            linestyle="--",
            label="Preferred Net Demand (Pre-Optimization)",
        )
        plt.plot(
            timesteps,
            actual_net_demand,
            marker="o",
            linestyle="-",
            label="Actual Net Demand (Post-Optimization)",
        )
        plt.title("Total Preferred vs Actual Net Demand Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Energy (MWh)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Plot 2: total price paid per timestep
        plt.figure(figsize=(10, 4))
        plt.bar(timesteps, price_paid, color="red")
        plt.title("Total Price Paid by Buyers per Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Price Paid (monetary units)")
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        # plt.show()

        # Plot 3: cumulative price paid over time
        plt.figure(figsize=(10, 4))
        plt.plot(timesteps, cumulative_paid, marker="o", linestyle="-", color="red")
        plt.title("Cumulative Total Price Paid Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Price Paid (monetary units)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # plt.show()

        # Plot 4: total generation
        plt.figure(figsize=(10, 4))
        plt.plot(timesteps, total_generation, marker="o", linestyle="-", color="green")
        plt.title("Total Energy Generation Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Energy (MWh)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # plt.show()
