from typing import override, Any
import numpy as np
from custom_types import *
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from abc import ABC, abstractmethod
from prosumer import *
import matplotlib.pyplot as plt
import math
from loguru import logger
import pandas as pd


class WholesaleMarketEnv(Env):
    """
    Simplified wholesale market environment where agents trade at fixed wholesale prices.
    Similar structure to DoubleAuctionEnv but without market clearing or forecasting.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        agent_configs: list[dict[str, Any]],
        wholesale_csv_path: str = "./data/representative_wholesale_price_2025.csv",
        max_timesteps: int = 96,
    ):
        super().__init__()

        self.agents: list[ProsumerAgent] = []
        self.n_agents = len(agent_configs)
        self.agent_ids = list(range(self.n_agents))
        self.max_timesteps = max_timesteps
        self.current_timestep = 0

        # Load wholesale prices from CSV
        self.wholesale_df = pd.read_csv(wholesale_csv_path)
        self.wholesale_prices = self.wholesale_df["Price (EUR/MWhe)"].values

        # Public Market Stats (simplified - just wholesale price)
        self.current_wholesale_price = (
            self.wholesale_prices[0] if len(self.wholesale_prices) > 0 else 50.0
        )
        self.last_total_traded_qty = 0.0

        self.AGENT_CLASS_MAP = {
            "ProsumerAgent": ProsumerAgent,
            "AggressiveSellerAgent": AggressiveSellerAgent,
            "AggressiveBuyerAgent": AggressiveBuyerAgent,
        }

        for i, config in enumerate(agent_configs):
            agent_class_name = config.get("class", "ProsumerAgent")
            AgentClass = self.AGENT_CLASS_MAP.get(agent_class_name, ProsumerAgent)
            self.agents.append(
                AgentClass(
                    agent_id=i,
                    load=config["load"],
                    flexible_load=config["flexible_load"],
                    fixed_load=config["fixed_load"],
                    generation_capacity=config["generation_capacity"],
                    generation_type=config["generation_type"]
                    if "generation_type" in config
                    and config["generation_type"] is not None
                    else "solar",
                )
            )

        # --- Define Action Space (Per Agent) ---
        MAX_PRICE = 500.0
        MAX_QTY = 50.0

        # Simple Box action space (same structure as DoubleAuctionEnv)
        self.action_space = Box(
            low=np.zeros((24, 2), dtype=np.float32),
            high=np.array([[MAX_PRICE, MAX_QTY]] * 24, dtype=np.float32),
            shape=(24, 2),
            dtype=np.float32,
        )

        # --- MODIFIED: Define Observation Space as a Dictionary ---
        MAX_DEMAND_ABS = max(max(c.schedule) for c in self.agents)
        MAX_CAPACITY = max(c["generation_capacity"] for c in agent_configs)
        MAX_NET_DEMAND = MAX_DEMAND_ABS
        MIN_NET_DEMAND = -MAX_CAPACITY
        MAX_SUM_QTY = self.n_agents * MAX_QTY

        self.observation_space = Dict(
            {
                # Agent State: [Net Demand, Last Cleared Qty]
                "agent_state": Box(low=0.0, high=MAX_QTY, shape=(1,), dtype=np.float32),
                # Market Stats: [P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1]
                "market_stats": Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array(
                        [MAX_PRICE, 1000.0], dtype=np.float32
                    ),  # [price, total_traded]
                    shape=(2,),
                    dtype=np.float32,
                ),
                "price_forecast": Box(
                    low=np.array([-100.0] * 23, dtype=np.float32),
                    high=np.array([500.0] * 23, dtype=np.float32),
                    shape=(23,),
                    dtype=np.float32,
                ),
            }
        )

        # History tracking (similar to DoubleAuctionEnv)
        self.wholesale_prices_history: list[float] = []
        self.traded_quantities: list[float] = []
        self.profit_history = []
        self.net_demand_history = []
        self.agent_trades_history = []  # Track individual agent trades

        print(f"WholesaleMarketEnv initialized with {self.n_agents} agents")
        print(f"Loaded {len(self.wholesale_prices)} wholesale prices")
        print(
            f"Price range: €{min(self.wholesale_prices):.2f} - €{max(self.wholesale_prices):.2f}"
        )

    def _get_obs(self) -> dict[int, dict[str, np.ndarray]]:
        """Get observations for all agents (simplified)."""
        market_stats_arr = np.array(
            [
                self.current_wholesale_price,
                self.last_total_traded_qty,
            ],
            dtype=np.float32,
        )

        # Get next 24 hours of wholesale prices for forecast
        price_forecast = self._get_wholesale_price_forecast()
        price_forecast_arr = np.array(price_forecast, dtype=np.float32)

        observations: dict[int, dict[str, np.ndarray]] = {}
        for agent in self.agents:
            agent_id = agent.agent_id

            # Handle net_demand properly
            if isinstance(agent.net_demand, list):
                net_demand_value = sum(agent.net_demand) if agent.net_demand else 0.0
            else:
                net_demand_value = float(agent.net_demand)

            agent_state_arr = np.array([abs(net_demand_value)], dtype=np.float32)

            observations[agent_id] = {
                "agent_state": agent_state_arr,
                "market_stats": market_stats_arr,
                "price_forecast": price_forecast_arr,
            }

        return observations

    def _get_wholesale_price_forecast(self) -> list[float]:
        """Get next 24 hours of wholesale prices for agent planning."""
        start_idx = self.current_timestep
        end_idx = min(start_idx + 24, len(self.wholesale_prices))

        # Get available prices
        prices = list(self.wholesale_prices[start_idx:end_idx])

        # Ensure forecast is exactly 24 hours
        while len(prices) < 24:
            prices.append(
                self.wholesale_prices[-1] if len(self.wholesale_prices) > 0 else 50.0
            )

        return prices[:23]  # Ensure exactly 24 prices

    def _calculate_rewards(self, trades: list[dict]) -> dict[int, float]:
        """Calculate rewards based on wholesale trading"""
        rewards: dict[int, float] = {agent_id: 0.0 for agent_id in self.agent_ids}

        for trade in trades:
            agent_id = trade["agent_id"]
            trade_value = trade["trade_value"]
            rewards[agent_id] = trade_value  # Positive for selling, negative for buying

        return rewards

    def _execute_wholesale_trading(self) -> list[dict]:
        """Execute wholesale trading for all agents at current wholesale price"""
        trades = []
        total_traded = 0.0

        for agent in self.agents:
            agent.calculate_net_demand()

            if isinstance(agent.net_demand, list):
                net_demand = sum(agent.net_demand) if agent.net_demand else 0.0
            else:
                net_demand = float(agent.net_demand)

            # In wholesale market, agent trades their net demand/surplus at wholesale price
            if net_demand > 0:
                # Agent needs energy (buying)
                trade_value = (
                    -net_demand * self.current_wholesale_price
                )  # Negative (cost)
                action_type = "buy"
            elif net_demand < 0:
                # Agent has surplus energy (selling)
                trade_value = (
                    -net_demand * self.current_wholesale_price
                )  # Positive (revenue)
                action_type = "sell"
            else:
                # No trading needed
                trade_value = 0.0
                action_type = "none"

            trade_info = {
                "agent_id": agent.agent_id,
                "net_demand": net_demand,
                "trade_quantity": abs(net_demand),
                "trade_value": trade_value,
                "wholesale_price": self.current_wholesale_price,
                "action_type": action_type,
            }

            trades.append(trade_info)
            total_traded += abs(net_demand)

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

        # Reset history
        self.wholesale_prices_history = []
        self.traded_quantities = []
        self.profit_history = []
        self.net_demand_history = []
        self.agent_trades_history = []

        # Reset agents
        for agent in self.agents:
            agent.calculate_net_demand()  # Let agents handle their own reset logic
            agent.profit = 0.0

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
        """Step the environment forward."""
        self.current_timestep += 1
        self.current_wholesale_price = (
            self.wholesale_prices[self.current_timestep]
            if self.current_timestep < len(self.wholesale_prices)
            else self.wholesale_prices[-1]
        )

        # Let agents adjust their strategies based on the wholesale price
        for agent_id, agent in enumerate(self.agents):
            obs = self._get_obs()[agent_id]
            agent_action = agent.devise_strategy_smarter(obs, self.action_space)
            # print(f"Agent {agent_id} action: {agent_action}")

        # Execute wholesale trading (no bidding/clearing in WholesaleMarketEnv)
        trades = self._execute_wholesale_trading()

        # Calculate rewards
        rewards = self._calculate_rewards(trades)

        # Update history
        self.wholesale_prices_history.append(self.current_wholesale_price)
        self.traded_quantities.append(self.last_total_traded_qty)
        self.profit_history.append(np.array(list(rewards.values())))
        self.net_demand_history.append([agent.net_demand for agent in self.agents])

        is_truncated = self.current_timestep >= self.max_timesteps
        terminated = {i: False for i in self.agent_ids}
        truncated = {i: is_truncated for i in self.agent_ids}

        observation = self._get_obs()
        info = {"timestep": self.current_timestep}
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
                f"Total Traded={self.last_total_traded_qty:.2f} MWh | Total Rewards=€{last_total_reward:.2f}"
            )

    def get_agent_color(self, agent_id: int) -> str:
        """Returns a color based on the agent's class name (same as DoubleAuctionEnv)"""
        agent_class_name = type(self.agents[agent_id]).__name__
        if agent_class_name == "AggressiveSellerAgent":
            return "r"
        elif agent_class_name == "AggressiveBuyerAgent":
            return "m"
        elif agent_class_name == "ProsumerAgent":
            return "b"
        return "k"

    def get_class_label(self, agent_id: int) -> str:
        """Returns the agent's class name for legend purposes"""
        return type(self.agents[agent_id]).__name__

    def plot_results(self):
        """Generate plots similar to DoubleAuctionEnv results"""
        timesteps = range(1, self.current_timestep + 1)

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
            timesteps, self.traded_quantities, marker="o", linestyle="-", color="purple"
        )
        plt.title("Total Traded Quantity Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Quantity (MWh)")
        plt.grid(True)
        plt.tight_layout()

        # Plot 2: Cumulative Profit/Loss per Agent
        plt.figure(figsize=(10, 6))
        profit_matrix = np.array(self.profit_history)
        plotted_classes = set()

        for i in range(self.n_agents):
            cumulative_profit = np.cumsum(profit_matrix[:, i])
            agent_class_name = self.get_class_label(i)
            color = self.get_agent_color(i)

            if agent_class_name not in plotted_classes:
                label = f"{agent_class_name} (Agent {i})"
                plotted_classes.add(agent_class_name)
            else:
                label = f"Agent {i}"

            plt.plot(
                timesteps,
                cumulative_profit,
                marker="o",
                linestyle="-",
                label=label,
                color=color,
                alpha=0.7,
            )

        plt.title("Cumulative Profit/Loss per Agent (Wholesale Market)")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Profit/Loss (€)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_price_change_for_single_day(self, day: int = 0):
        """Plot wholesale price for each hour of a specific day (same as DoubleAuctionEnv)"""
        plt.figure(figsize=(10, 5))
        start_index, end_index = day * 24, (day + 1) * 24

        if self.current_timestep < start_index:
            print(f"Simulation has not reached day {day}. No data to plot.")
            return

        daily_prices = self.wholesale_prices_history[start_index:end_index]
        hours = range(len(daily_prices))

        plt.plot(hours, daily_prices, marker="o", linestyle="-", color="orange")
        plt.title(f"Wholesale Price vs. Hour for Day {day}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Wholesale Price (€/MWh)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(0, 24, 2))
        plt.xlim(-0.5, 23.5)
        plt.tight_layout()
        plt.show()
