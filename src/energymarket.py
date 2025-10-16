from typing import override, Any
import numpy as np
from custom_types import *
from gymnasium import Env
from gymnasium.spaces import Box
from abc import ABC, abstractmethod
from prosumer import *
import matplotlib.pyplot as plt
import math
from loguru import logger


class BaseMarket(ABC):
    def __init__(self):
        pass

    def match_bids(self):
        pass


class MarketClearingAgent(ABC):
    """
    An abstract base class for any type of clearing agent
    """

    def __init__(self):
        pass

    @abstractmethod
    def clear_market(
            self, bids_array: np.ndarray, offers_array: np.ndarray
    ) -> MarketResult:
        pass


class DoubleAuctionClearingAgent(MarketClearingAgent):
    """
    A market agent responsible for collecting bids and offers and clearing the market.
    """

    def __init__(self):
        super().__init__()

    @override
    def clear_market(
            self, bids_array: np.ndarray, offers_array: np.ndarray
    ) -> MarketResult:
        """
        Determines the market clearing price and quantity.

        The algorithm finds the intersection point of the cumulative supply (offers) and demand (bids) curves.

        bids_array and offers_array are assumed to be Nx3 arrays:
        [Agent ID (0), Price (1), Quantity (2)]
        """
        # Check if arrays are empty
        if bids_array.size == 0 or offers_array.size == 0:
            return 0.0, 0.0

        if (
                bids_array.ndim != 2
                or bids_array.shape[1] != 3
                or offers_array.ndim != 2
                or offers_array.shape[1] != 3
        ):
            raise ValueError(
                f"Input arrays must be of shape (N, 3) ([Agent ID, Price, Quantity]). "
                f"Received shapes: bids {bids_array.shape}, offers {offers_array.shape}."
            )
        # Bids (Demand) sorted descending by price (highest price first)
        bids_sorted = bids_array[bids_array[:, 1].argsort()[::-1]]
        # Offers (Supply) sorted ascending by price (lowest price first)
        offers_sorted = offers_array[offers_array[:, 1].argsort()]

        # Extract prices and quantities
        bid_prices = bids_sorted[:, 1]
        bid_quantities = bids_sorted[:, 2]
        offer_prices = offers_sorted[:, 1]
        offer_quantities = offers_sorted[:, 2]

        # Find the Marginal Match Index (k)
        min_len = min(len(bid_prices), len(offer_prices))
        match_indices = np.where(bid_prices[:min_len] >= offer_prices[:min_len])[0]

        if len(match_indices) == 0:
            return 0.0, 0.0

        k = match_indices[-1]
        marginal_bid_price = bid_prices[k]
        marginal_offer_price = offer_prices[k]
        clearing_price = (marginal_bid_price + marginal_offer_price) / 2.0
        bid_cumsum = np.cumsum(bid_quantities[: k + 1])
        offer_cumsum = np.cumsum(offer_quantities[: k + 1])
        clearing_quantity = min(bid_cumsum[-1], offer_cumsum[-1])


        # bought_energy = {}
        #
        # quantity = 0
        # i = 0
        # while quantity < clearing_quantity:
        #
        #     bids = np.where(bids_array[0] = bid_prices[i])
        #
        #     i += 1

        return (clearing_price, clearing_quantity)
                    # , bought_energy)


class DoubleAuctionEnv(Env):
    """
    Gymnasium-compatible environment for a Double Auction Energy Market Simulation.
    Agents submit bids for a 24-hour horizon. The environment clears the current
    hour and provides a price forecast for the next 23 hours.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            agent_configs: list[dict[str, Any]],
            market_clearing_agent: MarketClearingAgent,
            max_timesteps: int = 100,
    ):
        super().__init__()

        self.market_agent = market_clearing_agent
        self.agents: list[ProsumerAgent] = []
        self.n_agents = len(agent_configs)
        self.agent_ids = list(range(self.n_agents))
        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.FORECAST_HORIZON = 24

        # Public Market Stats for Observation (P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1)
        self.last_clearing_price = 5.0
        self.last_clearing_quantity = 0.0
        self.last_total_bids_qty = 0.0
        self.last_total_offers_qty = 0.0

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
        # Action: Array of [Price, Quantity] for the next 24 hours
        MAX_PRICE = 10.0
        MAX_QTY = 50.0

        low_action = np.array([0.0, 0.0], dtype=np.float32)
        high_action = np.array([MAX_PRICE, MAX_QTY], dtype=np.float32)

        self.action_space = Box(
            low=np.tile(low_action, (self.FORECAST_HORIZON, 1)),
            high=np.tile(high_action, (self.FORECAST_HORIZON, 1)),
            shape=(self.FORECAST_HORIZON, 2),
            dtype=np.float32,
        )

        # --- Define Observation Space (Per Agent) ---
        # Obs: [ND_i] + [Market_Stats (4)] + [Price_Forecast (23)]

        # This section is crazy and idk what it does
        MAX_DEMAND_ABS = max(
            max(c.schedule) for c in self.agents
        )
        MAX_CAPACITY = max(c["generation_capacity"] for c in agent_configs)
        MAX_NET_DEMAND = MAX_DEMAND_ABS
        MIN_NET_DEMAND = -MAX_CAPACITY
        MAX_SUM_QTY = self.n_agents * MAX_QTY

        # Features: ND_i, P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1
        low_obs_base = [MIN_NET_DEMAND, 0.0, 0.0, 0.0, 0.0]
        high_obs_base = [MAX_NET_DEMAND, MAX_PRICE, MAX_QTY, MAX_SUM_QTY, MAX_SUM_QTY]

        # Add forecast prices to observation space
        low_obs_forecast = [0.0] * (self.FORECAST_HORIZON - 1)
        high_obs_forecast = [MAX_PRICE] * (self.FORECAST_HORIZON - 1)

        low_obs = low_obs_base + low_obs_forecast
        high_obs = high_obs_base + high_obs_forecast

        self.observation_space = Box(
            low=np.array(low_obs, dtype=np.float32),
            high=np.array(high_obs, dtype=np.float32),
            shape=(len(low_obs),),
            dtype=np.float32,
        )

        # History logging
        self.clearing_prices: list[float] = []
        self.clearing_quantities: list[float] = []
        self.profit_history = []
        self.net_demand_history = []
        self.action_history = []
        self.market_orders_history = []

    def _forecast_prices(self, actions: dict[int, np.ndarray]) -> list[float]:
        """
        Simulates market clearing for future timesteps (t+1 to t+23) based on
        submitted actions to generate a price forecast.

        NOTE: This forecast uses the agents' *current* net_demand to interpret
        their future bids/offers. It does not simulate changes in their demand.
        """
        # if self.current_timestep == 0:
        #     return 5 * np.ones(FORECAST_HORIZON, dtype=np.float32)

        forecasted_prices = []
        for h in range(1, self.FORECAST_HORIZON):  # Iterate from hour t+1 to t+23
            future_bids, future_offers = [], []
            for agent in self.agents:
                agent_id = agent.agent_id
                if agent_id in actions:
                    # Get the planned action for future hour 'h'
                    price, quantity = actions[agent_id][0][h]
                    if quantity == 0:
                        price, quality = actions[agent_id][1][h]
                        future_offers.extend([(agent_id, price, quality)])
                    else:
                        future_bids.extend([(agent_id, price, quantity)])

            bids_arr = (
                np.array(future_bids, dtype=float) if future_bids else np.array([])
            )
            offers_arr = (
                np.array(future_offers, dtype=float) if future_offers else np.array([])
            )

            price, _ = self.market_agent.clear_market(bids_arr, offers_arr)
            forecasted_prices.append(price)

        return forecasted_prices

    def _get_obs(self, price_forecast: list[float]) -> dict[int, np.ndarray]:
        """Compiles observations for each agent, including the price forecast."""
        market_stats = [
            self.last_clearing_price,
            self.last_clearing_quantity,
            self.last_total_bids_qty,
            self.last_total_offers_qty,
        ]

        observations: dict[int, np.ndarray] = {}
        for agent in self.agents:
            obs_i = agent.net_demand + market_stats + price_forecast
            observations[agent.agent_id] = np.array(obs_i, dtype=np.float32)

        return observations

    def _calculate_rewards(self, clearing_price: float) -> dict[int, float]:
        """Calculates the reward (profit) for all agents for the current timestep."""
        rewards: dict[int, float] = {}
        for agent in self.agents:
            rewards[agent.agent_id] = agent.calculate_profit(clearing_price)
        return rewards

    @override
    def reset(
            self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed)
        self.current_timestep = 0
        self.last_clearing_price = 5.0
        self.last_clearing_quantity = 0.0
        self.last_total_bids_qty = 0.0
        self.last_total_offers_qty = 0.0
        (
            self.clearing_prices,
            self.clearing_quantities,
            self.profit_history,
            self.net_demand_history,
            self.action_history,
        ) = [], [], [], [], []

        for agent in self.agents:
            agent.calculate_net_demand()
            agent.profit = 0.0

        # Initial forecast is zero
        initial_forecast = [5.0] * (self.FORECAST_HORIZON - 1)
        observation = self._get_obs(initial_forecast)
        info = {"timestep": self.current_timestep}
        return observation, info

    def step(
            self, actions: dict[int, np.ndarray]
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[str, int | float],
    ]:
        """
        Runs one time step.
        - Clears the market for the first hour (t=0) of the action plan.
        - Calculates rewards for t=0.
        - Creates a price forecast for the next 23 hours.
        - Returns observations including this forecast.
        """
        self.current_timestep += 1

        # Compile Market Orders for the CURRENT HOUR (t=0)
        all_bids, all_offers = [], []
        for agent in self.agents:
            agent_id = agent.agent_id
            if agent_id in actions:
                # Use only the action for the current hour (index 0)
                price, quantity = actions[agent_id][0][0]
                if quantity == 0:
                    price, quantity = actions[agent_id][1][0]
                    all_offers.extend([(agent_id, price, quantity)])
                else:
                    all_bids.extend([(agent_id, price, quantity)])

        total_bids_qty = sum(b[2] for b in all_bids)
        total_offers_qty = sum(o[2] for o in all_offers)

        # Market Clearing for the CURRENT HOUR
        bids_array = np.array(all_bids, dtype=float) if all_bids else np.array([])
        offers_array = np.array(all_offers, dtype=float) if all_offers else np.array([])
        self.market_orders_history.append((bids_array.copy(), offers_array.copy()))

        clearing_price, clearing_quantity = self.market_agent.clear_market(
            bids_array, offers_array
        )

        # Calculate Rewards based on CURRENT HOUR's outcome
        reward = self._calculate_rewards(clearing_price)

        # Generate Price Forecast for the NEXT 23 HOURS
        price_forecast = self._forecast_prices(actions)

        # Update State for the next observation
        self.last_clearing_price = clearing_price
        self.last_clearing_quantity = clearing_quantity
        self.last_total_bids_qty = total_bids_qty
        self.last_total_offers_qty = total_offers_qty

        # Update agents' internal states for the next real timestep
        for agent in self.agents:
            agent.calculate_net_demand()

        # Check Termination and Get Next Observation
        is_truncated = self.current_timestep >= self.max_timesteps
        terminated = {i: False for i in self.agent_ids}
        truncated = {i: is_truncated for i in self.agent_ids}

        observation = self._get_obs(price_forecast)

        # Log history
        self.clearing_prices.append(clearing_price)
        self.clearing_quantities.append(clearing_quantity)
        self.profit_history.append(np.array(list(reward.values())))
        self.net_demand_history.append([agent.net_demand for agent in self.agents])

        info = {
            "timestep": self.current_timestep,
            "clearing_price": clearing_price,
            "clearing_quantity": clearing_quantity,
            "price_forecast": price_forecast,
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Prints the state."""
        if mode == "human":
            last_total_reward = (
                self.profit_history[-1].sum() if self.profit_history else 0.0
            )
            logger.info(
                f"T={self.current_timestep:02d} | Price={self.last_clearing_price:.2f} | Qty={self.clearing_quantities[-1]:.2f} | Rewards={last_total_reward:.2f}"
            )

    def get_agent_color(self, agent_id: int) -> str:
        """Returns a color based on the agent's class name."""
        agent_class_name = type(self.agents[agent_id]).__name__
        if agent_class_name == "AggressiveSellerAgent":
            return "r"  # Red
        elif agent_class_name == "AggressiveBuyerAgent":
            return "m"  # Magenta
        elif agent_class_name == "ProsumerAgent":
            return "b"  # Blue
        return "k"  # Black for unknown

    def get_class_label(self, agent_id: int) -> str:
        """Returns the agent's class name for legend purposes."""
        return type(self.agents[agent_id]).__name__

    def plot_results(self):
        """
        Generates and displays graphs of the simulation results from history.
        All plots created here will be displayed simultaneously.
        """

        timesteps = range(1, self.current_timestep + 1)

        # Plot 1: Market Clearing Price and Quantity Over Time
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(timesteps, self.clearing_prices, marker="o", linestyle="-", color="b")
        plt.title("Market Clearing Price Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Price (units)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            timesteps, self.clearing_quantities, marker="o", linestyle="-", color="g"
        )
        plt.title("Market Clearing Quantity Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Quantity (MWh)")
        plt.grid(True)
        plt.tight_layout()

        # Plot 2: Cumulative Profit/Loss per Agent
        plt.figure(figsize=(10, 6))
        profit_matrix = np.array(self.profit_history)

        # Keep track of which class/color combos have been plotted for the legend
        plotted_classes = set()

        for i in range(self.n_agents):
            cumulative_profit = np.cumsum(profit_matrix[:, i])

            agent_class_name = self.get_class_label(i)
            color = self.get_agent_color(i)

            # Use label only if the class hasn't been plotted yet to avoid cluttering the legend
            if agent_class_name not in plotted_classes:
                label = f"{agent_class_name} (Agent {i})"
                plotted_classes.add(agent_class_name)
            else:
                label = f"Agent {i}"  # Just show agent ID if class is redundant

            plt.plot(
                timesteps,
                cumulative_profit,
                marker="o",
                linestyle="-",
                label=label,
                color=color,
                alpha=0.7,
            )

        plt.title("Cumulative Profit/Loss per Agent (RL Target)")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Profit/Loss (units)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()

    def plot_bid_ask_curves(self, num_plots=10):
        """
        Generates bid-ask curves for the last N timesteps, splitting the total
        into multiple figures, with up to 10 plots per figure, for better readability.
        The individual steps are consistently colored (Blue=Demand, Red=Supply).
        All figures are displayed simultaneously at the end.
        """

        if len(self.market_orders_history) < 1:
            print("Not enough market history to plot bid-ask curves.")
            return

        # Select the last N timesteps
        start_index = max(0, len(self.market_orders_history) - num_plots)
        plot_data = []

        for t in range(start_index, len(self.market_orders_history)):
            # data['bids/offers'] is now List[Tuple[agent_id, price, quantity]]
            bids_list, offers_list = self.market_orders_history[t]

            clearing_price = self.clearing_prices[t]
            clearing_quantity = self.clearing_quantities[t]

            plot_data.append(
                {
                    "t": t + 1,
                    "bids": bids_list,
                    "offers": offers_list,
                    "price": clearing_price,
                    "quantity": clearing_quantity,
                }
            )

        TOTAL_PLOTS = len(plot_data)
        if TOTAL_PLOTS == 0:
            return

        # Configuration for multi-figure plotting: 10 plots per figure (2 rows x 5 columns)
        PLOTS_PER_FIGURE = 10
        N_COLS = 5
        N_ROWS = math.ceil(PLOTS_PER_FIGURE / N_COLS)

        # Define standard colors for curves
        BID_COLOR = "b"  # Blue for Demand (Bids)
        OFFER_COLOR = "r"  # Red for Supply (Offers)

        # Iterate over plot_data in chunks
        num_figures = math.ceil(TOTAL_PLOTS / PLOTS_PER_FIGURE)

        for fig_index in range(num_figures):
            chunk_start = fig_index * PLOTS_PER_FIGURE
            chunk_end = min((fig_index + 1) * PLOTS_PER_FIGURE, TOTAL_PLOTS)
            current_chunk = plot_data[chunk_start:chunk_end]

            # Determine grid size for the current figure
            num_in_chunk = len(current_chunk)
            # Use 1 row if 5 or fewer plots, otherwise use N_ROWS (2)
            actual_rows = 1 if num_in_chunk <= N_COLS else N_ROWS

            fig, axes = plt.subplots(
                actual_rows,
                N_COLS,
                figsize=(15, 3 * actual_rows + 1),  # Adjusted size for better fit
                constrained_layout=True,
            )

            # Flatten the axes array for easier indexing, ensuring it's always iterable
            axes = np.array(axes).flatten()

            for i, data in enumerate(current_chunk):
                ax = axes[i]

                # --- Demand Curve (Bids) ---
                if len(data["bids"]) > 0:
                    bids_all = np.array(data["bids"], dtype=float)
                    bids_sorted = bids_all[bids_all[:, 1].argsort()[::-1]]
                    bid_prices = bids_sorted[:, 1]
                    bid_quantities = bids_sorted[:, 2]

                    # Plotting the segmented steps
                    current_qty = 0.0
                    for j in range(len(bid_prices)):
                        price = bid_prices[j]
                        qty = bid_quantities[j]
                        next_qty = current_qty + qty

                        # 1. Horizontal segment (Demand, Blue)
                        ax.plot(
                            [current_qty, next_qty],
                            [price, price],
                            color=BID_COLOR,
                            linewidth=2.0,
                            linestyle="-",
                            zorder=2,
                        )

                        # 2. Vertical jump (generic gray)
                        if j < len(bid_prices) - 1:
                            next_price = bid_prices[j + 1]
                            ax.plot(
                                [next_qty, next_qty],
                                [price, next_price],
                                color="gray",
                                linewidth=0.8,
                                linestyle="--",
                                zorder=1,
                            )

                        current_qty = next_qty

                # --- Supply Curve (Offers) ---
                if len(data["offers"]) > 0:
                    offers_all = np.array(data["offers"], dtype=float)
                    offers_sorted = offers_all[offers_all[:, 1].argsort()]
                    offer_prices = offers_sorted[:, 1]
                    offer_quantities = offers_sorted[:, 2]

                    # Plotting the segmented steps
                    current_qty = 0.0
                    for j in range(len(offer_prices)):
                        price = offer_prices[j]
                        qty = offer_quantities[j]
                        next_qty = current_qty + qty

                        # 1. Horizontal segment (Supply, Red)
                        ax.plot(
                            [current_qty, next_qty],
                            [price, price],
                            color=OFFER_COLOR,
                            linewidth=2.0,
                            linestyle="-",
                            zorder=2,
                        )

                        # 2. Vertical jump (generic gray)
                        if j < len(offer_prices) - 1:
                            next_price = offer_prices[j + 1]
                            ax.plot(
                                [next_qty, next_qty],
                                [price, next_price],
                                color="gray",
                                linewidth=0.8,
                                linestyle="--",
                                zorder=1,
                            )

                        current_qty = next_qty

                # --- Intersection Point (Clearing Price/Quantity) ---
                if data["price"] > 0 or data["quantity"] > 0:
                    ax.plot(
                        data["quantity"], data["price"], "go", markersize=6, zorder=5
                    )
                    # Dotted lines to axes
                    ax.axhline(
                        data["price"],
                        color="gray",
                        linestyle="--",
                        linewidth=0.5,
                        zorder=0,
                    )
                    ax.axvline(
                        data["quantity"],
                        color="gray",
                        linestyle="--",
                        linewidth=0.5,
                        zorder=0,
                    )

                # --- Legend Generation (using proxy artists for curve colors) ---
                proxy_handles = []

                # Add proxy for Demand Curve (Blue)
                proxy_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=BID_COLOR,
                        linewidth=3,
                        linestyle="-",
                        label="Demand Curve",
                    )
                )

                # Add proxy for Supply Curve (Red)
                proxy_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=OFFER_COLOR,
                        linewidth=3,
                        linestyle="-",
                        label="Supply Curve",
                    )
                )

                # Add proxy for Market Clearing Point (Green)
                proxy_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color="g",
                        marker="o",
                        linestyle="",
                        markersize=6,
                        label=f"Clearance (${data['price']:.2f})",
                    )
                )

                ax.set_title(f"T={data['t']} | P={data['price']:.2f}", fontsize=10)
                ax.set_xlabel("Quantity (MWh)", fontsize=8)
                ax.set_ylabel("Price ($)", fontsize=8)
                ax.legend(handles=proxy_handles, fontsize=7, loc="lower right")
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

            # Hide unused subplots in the last figure
            for j in range(num_in_chunk, len(axes)):
                fig.delaxes(axes[j])

            start_t = current_chunk[0]["t"]
            end_t = current_chunk[-1]["t"]

            fig.suptitle(
                f"Bid-Ask Curves and Market Clearing (Timesteps {start_t} to {end_t})",
                fontsize=14,
            )

        # Only call show once to display all generated Bid-Ask Curve figures simultaneously
        plt.show()

    def plot_price_change_for_single_day(self, day: int = 0):
        """
        Plots the market clearing price for each hour of a specific simulation day.

        Args:
            day (int): The day number to plot (e.g., 0 for the first day).
        """
        plt.figure(figsize=(10, 5))

        # Calculate the start and end timesteps for the requested day
        start_index = day * 24
        end_index = start_index + 24

        if self.current_timestep < start_index:
            print(f"Simulation has not reached day {day}. No data to plot.")
            return

        # Slice the price history for the specific day
        daily_prices = self.clearing_prices[start_index:end_index]
        hours = range(len(daily_prices))

        plt.plot(hours, daily_prices, marker="o", linestyle="-", color="c")
        plt.title(f"Market Clearing Price vs. Hour for Day {day}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Clearing Price ($)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(np.arange(0, 24, 2))  # Set x-axis ticks for every 2 hours
        plt.xlim(-0.5, 23.5)  # Set x-axis limits
        plt.tight_layout()
        plt.show()


class P2P(BaseMarket):
    def __init__(self, market_price):
        BaseMarket.__init__(self)
