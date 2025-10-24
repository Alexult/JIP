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
            self, bids_array: np.ndarray, offers_array: np.ndarray, debug=False
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
            self, bids_array: np.ndarray, offers_array: np.ndarray, debug=False
    ) -> MarketResult:
        """
        Determines the market clearing price and quantity by finding the intersection
        point of the aggregate supply and demand curves.
        """
        # Check for empty orders
        if bids_array.size == 0 or offers_array.size == 0:
            return 0.0, 0.0, []

        if (
                bids_array.ndim != 2
                or bids_array.shape[1] != 3
                or offers_array.ndim != 2
                or offers_array.shape[1] != 3
        ):
            raise ValueError(
                f"Input arrays must be of shape (N, 3). "
                f"Received shapes: bids {bids_array.shape}, offers {offers_array.shape}."
            )

        # Build the demand and supply curves by sorting the orders.
        # Demand curve steps from high to low prices.
        bids_sorted = bids_array[bids_array[:, 1].argsort()[::-1]]
        # Supply curve steps from low to high prices.
        offers_sorted = offers_array[offers_array[:, 1].argsort()]

        bid_prices = bids_sorted[:, 1]
        bid_quantities = bids_sorted[:, 2]
        offer_prices = offers_sorted[:, 1]
        offer_quantities = offers_sorted[:, 2]

        # Find the intersection quantity (max_trade_volume).
        # We test every unique price as a potential clearing price to see where the
        # trade volume is maximized. This is where the curves cross.
        all_prices = np.unique(np.concatenate((bid_prices, offer_prices)))

        max_trade_volume = 0.0
        # TODO: Is a hotfix, is in-efficient, can use prefix-sum
        for price in all_prices:
            # Quantity demanded at this price (all bids >= price)
            demand_at_price = np.sum(bid_quantities[bid_prices >= price])
            # Quantity supplied at this price (all offers <= price)
            supply_at_price = np.sum(offer_quantities[offer_prices <= price])

            # The amount that can actually trade is the minimum of the two
            current_trade_volume = min(demand_at_price, supply_at_price)

            # If this price allows for more trading, it's closer to the equilibrium
            if current_trade_volume > max_trade_volume:
                max_trade_volume = current_trade_volume

        clearing_quantity = max_trade_volume

        if clearing_quantity == 0:
            return 0.0, 0.0, []

        # STEP 3: Find the intersection price.
        # The price is set by the marginal traders (the last buyer and seller
        # needed to fulfill the clearing_quantity).
        try:
            # Find the lowest-priced bid that is part of the cleared quantity
            bids_cum_q = np.cumsum(bid_quantities)
            marginal_bid_idx = np.where(bids_cum_q >= clearing_quantity)[0][0]
            marginal_bid_price = bid_prices[marginal_bid_idx]

            # Find the highest-priced offer that is part of the cleared quantity
            offers_cum_q = np.cumsum(offer_quantities)
            marginal_offer_idx = np.where(offers_cum_q >= clearing_quantity)[0][0]
            marginal_offer_price = offer_prices[marginal_offer_idx]

            # The clearing price is the midpoint (a common auction rule)
            clearing_price = (marginal_bid_price + marginal_offer_price) / 2.0
        except IndexError:
            return 0.0, 0.0, []

        debug and logger.debug(f"bids: {bids_sorted}")
        debug and logger.debug(f"asks: {offers_sorted}")
        debug and logger.debug(f"Clearing Price (Intersection Y): {clearing_price}")
        debug and logger.debug(
            f"Clearing Quantity (Intersection X): {clearing_quantity}"
        )

        # Calculate who gets what based on the intersection point
        cleared_participants: list[tuple[int, float]] = []
        buyers_cleared_qty = 0
        for agent_id, price, quantity in bids_sorted:
            if price >= clearing_price:
                remaining_market_qty = clearing_quantity - buyers_cleared_qty
                if remaining_market_qty <= 0:
                    break
                trade_qty = min(quantity, remaining_market_qty)
                cleared_participants.append((int(agent_id), trade_qty))
                buyers_cleared_qty += trade_qty
            else:
                break

        sellers_cleared_qty = 0
        for agent_id, price, quantity in offers_sorted:
            if price <= clearing_price:
                remaining_market_qty = clearing_quantity - sellers_cleared_qty
                if remaining_market_qty <= 0:
                    break
                trade_qty = min(quantity, remaining_market_qty)
                cleared_participants.append((int(agent_id), -trade_qty))
                sellers_cleared_qty += trade_qty
            else:
                break

        debug and logger.debug(f"Cleared participants: {cleared_participants}")

        return (clearing_price, clearing_quantity, cleared_participants)


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
        self.FORECAST_HORIZON = 10

        # Public Market Stats
        self.last_clearing_price = 5.0
        self.last_clearing_quantity = 0.0
        self.last_total_bids_qty = 0.0
        self.last_total_offers_qty = 0.0
        # --- NEW: Track last cleared quantity per agent ---
        self.last_cleared_quantities: dict[int, float] = {
            i: 0.0 for i in self.agent_ids
        }

        for i, config in enumerate(agent_configs):
            self.agents.append(
                ProsumerAgent(
                    agent_id=i,
                    load=config["load"],
                    generation_capacity=config["generation_capacity"],
                    marginal_price=config["marginal_price"],
                    generation_type=config["generation_type"]
                    if "generation_type" in config
                       and config["generation_type"] is not None
                    else "solar",
                )
            )

        net_demand = [agents.load for agents in self.agents]
        net_demand = list(map(list, zip(*net_demand)))
        self.total_demand = [sum(net_demand[i]) for i in range(max_timesteps)]

        # --- Define Action Space (Per Agent) ---
        MAX_PRICE = 20.0
        MAX_QTY = 1000.0

        low_action = np.array([0.0, 0.0], dtype=np.float32)
        high_action = np.array([MAX_PRICE, MAX_QTY], dtype=np.float32)

        self.action_space = Box(
            low=np.tile(low_action, (self.FORECAST_HORIZON, 1)),
            high=np.tile(high_action, (self.FORECAST_HORIZON, 1)),
            shape=(self.FORECAST_HORIZON, 2),
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
                # Agent State: [Last Cleared Qty,Current buy tariff, Current sell tariff]
                "agent_state": Box(low=0.0, high=MAX_QTY, shape=(3,), dtype=np.float32),
                # Market Stats: [P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1]
                "market_stats": Box(
                    low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array(
                        [MAX_PRICE, MAX_SUM_QTY, MAX_SUM_QTY, MAX_SUM_QTY],
                        dtype=np.float32,
                    ),
                    shape=(4,),
                    dtype=np.float32,
                ),
                # Price Forecast for next 23 hours
                "price_forecast": Box(
                    low=0.0,
                    high=MAX_PRICE,
                    shape=(self.FORECAST_HORIZON - 1,),
                    dtype=np.float32,
                ),
            }
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
        """
        forecasted_prices = []
        for h in range(1, self.FORECAST_HORIZON):
            future_bids, future_offers = [], []
            for agent in self.agents:
                agent_id = agent.agent_id
                if agent_id in actions:
                    price, quantity = actions[agent_id][0][h]
                    if quantity == 0:
                        price, quantity = actions[agent_id][1][h]
                        if quantity > 0:
                            future_offers.append((agent_id, price, quantity))
                    else:
                        future_bids.append((agent_id, price, quantity))

            bids_arr = (
                np.array(future_bids, dtype=float) if future_bids else np.empty((0, 3))
            )
            offers_arr = (
                np.array(future_offers, dtype=float)
                if future_offers
                else np.empty((0, 3))
            )

            price, _, _ = self.market_agent.clear_market(bids_arr, offers_arr)
            forecasted_prices.append(price)

        return forecasted_prices

    def _get_obs(self, price_forecast: list[float]) -> dict[int, dict[str, np.ndarray]]:
        """Compiles dictionary observations for each agent."""
        market_stats_arr = np.array(
            [
                self.last_clearing_price,
                self.last_clearing_quantity,
                self.last_total_bids_qty,
                self.last_total_offers_qty,
            ],
            dtype=np.float32,
        )

        price_forecast_arr = np.array(price_forecast, dtype=np.float32)

        observations: dict[int, dict[str, np.ndarray]] = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_state_arr = np.array(
                [
                    self.last_cleared_quantities[agent_id],
                    self.buy_tariff,
                    self.sell_tariff,
                ],
                dtype=np.float32,
            )

            observations[agent_id] = {
                "agent_state": agent_state_arr,
                "market_stats": market_stats_arr,
                "price_forecast": price_forecast_arr,
            }

        return observations

    def _calculate_rewards(
            self, clearing_price: float, cleared_participants: list[tuple[int, float]]
    ) -> dict[int, float]:
        """Calculates the reward (profit) for all agents for the current timestep."""
        rewards: dict[int, float] = {agent_id: 0.0 for agent_id in self.agent_ids}

        # Create a lookup dictionary for cleared quantities
        cleared_map = dict(cleared_participants)

        # for agent in self.agents:
        #     agent_id = agent.agent_id
        #     if agent_id in cleared_map:
        #         cleared_qty = cleared_map[agent_id]
        #         rewards[agent_id] = agent.calculate_profit(
        #             clearing_price, cleared_qty
        #         )

        return rewards

    @override
    def reset(
            self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed)
        self.current_timestep = 0
        self.last_clearing_price = 5.0
        self.last_clearing_quantity = 0.0
        self.last_total_bids_qty = 0.0
        self.last_total_offers_qty = 0.0

        # --- NEW: Reset cleared quantities ---
        self.last_cleared_quantities = {i: 0.0 for i in self.agent_ids}

        (
            self.clearing_prices,
            self.clearing_quantities,
            self.profit_history,
            self.net_demand_history,
            self.action_history,
        ) = [], [], [], [], []

        # NEW histories for consumption & costs #######################
        self.preferred_consumption_history = []  # total requested bids qty each timestep (MWh)
        self.actual_consumption_history = []  # total cleared buyer qty each timestep (MWh)
        self.total_price_paid_history = []  # clearing_price * actual_consumption (monetary units)
        self.cumulative_price_paid_history = []  # cumulative sum over time of total_price_paid_history

        for agent in self.agents:
            agent.calculate_net_demand(0)
            agent.profit = 0.0

        initial_forecast = [0.6] * (self.FORECAST_HORIZON - 1)
        observation = self._get_obs(initial_forecast)
        info = {"timestep": self.current_timestep}
        return observation, info

    def step(
            self, actions: dict[int, np.ndarray]
    ) -> tuple[
        dict[int, dict[str, np.ndarray]],
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

        # Compile Market Orders for the CURRENT HOUR (t=0)
        all_bids, all_offers = [], []
        for agent in self.agents:
            agent_id = agent.agent_id
            if agent_id in actions:
                # Use only the action for the current hour (index 0)
                price, quantity = actions[agent_id][0][0]
                # If bid quantity is 0, check the offers
                if quantity == 0:
                    price, quantity = actions[agent_id][1][0]
                    if quantity > 0:
                        all_offers.append((agent_id, price, quantity))
                else:
                    all_bids.append((agent_id, price, quantity))

        total_bids_qty = sum(b[2] for b in all_bids)
        total_offers_qty = sum(o[2] for o in all_offers)

        bids_array = np.array(all_bids, dtype=float) if all_bids else np.empty((0, 3))
        offers_array = (
            np.array(all_offers, dtype=float) if all_offers else np.empty((0, 3))
        )
        self.market_orders_history.append((bids_array.copy(), offers_array.copy()))

        clearing_price, clearing_quantity, cleared_participants = (
            self.market_agent.clear_market(bids_array, offers_array, debug=True)
        )

        ########## preferred and actual consumption and cost
        # preferred consumption = sum of all bid quantities submitted for this current hour
        preferred_consumption = float(total_bids_qty)  # already computed above

        # Determine which agent_ids were buyers (from bids_array)
        bids_agent_ids = set()
        if bids_array.size > 0:
            # bids_array shape (N,3) with first column agent_id
            bids_agent_ids = set(int(x) for x in bids_array[:, 0])

        # actual consumption = sum of cleared quantities for those agents that were buyers
        actual_consumption = 0.0
        for agent_id, qty in cleared_participants:
            if int(agent_id) in bids_agent_ids:
                actual_consumption += float(qty)

        # total price paid by buyers this timestep
        total_price_paid = float(clearing_price) * actual_consumption

        # append into history arrays (cumulative computed and stored)
        self.preferred_consumption_history.append(preferred_consumption)
        self.actual_consumption_history.append(actual_consumption)
        self.total_price_paid_history.append(total_price_paid)
        cumulative = (
                         self.cumulative_price_paid_history[-1]
                         if self.cumulative_price_paid_history
                         else 0.0
                     ) + total_price_paid
        self.cumulative_price_paid_history.append(cumulative)
        ############

        reward = self._calculate_rewards(clearing_price, cleared_participants)

        price_forecast = self._forecast_prices(actions)

        # --- Update last cleared quantities for the next observation ---
        self.last_cleared_quantities = {i: 0.0 for i in self.agent_ids}
        for agent_id, qty in cleared_participants:
            self.last_cleared_quantities[agent_id] = qty
            self.agents[agent_id].handle_after_auction(
                qty, self.current_timestep, self.buy_tariff, self.sell_tariff
            )

        self.last_clearing_price = clearing_price
        self.last_clearing_quantity = clearing_quantity
        self.last_total_bids_qty = total_bids_qty
        self.last_total_offers_qty = total_offers_qty

        # for agent in self.agents:
        #     agent.calculate_net_demand(self.current_timestep)

        is_truncated = self.current_timestep >= self.max_timesteps - 1
        terminated = {i: False for i in self.agent_ids}
        truncated = {i: is_truncated for i in self.agent_ids}

        self.current_timestep += 1

        observation = self._get_obs(price_forecast)

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

    def plot_results(self):
        self.plot_market_result()
        self.plot_bid_ask_curves(4)
        self.plot_consumption_and_costs()

    def plot_market_result(self):
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

    #
    #     # Plot 2: Cumulative Profit/Loss per Agent
    #     plt.figure(figsize=(10, 6))
    #     profit_matrix = np.array(self.profit_history)
    #     plotted_classes = set()
    #
    #     for i in range(self.n_agents):
    #         cumulative_profit = np.cumsum(profit_matrix[:, i])
    #         agent_class_name = self.get_class_label(i)
    #         color = self.get_agent_color(i)
    #
    #         if agent_class_name not in plotted_classes:
    #             label = f"{agent_class_name} (Agent {i})"
    #             plotted_classes.add(agent_class_name)
    #         else:
    #             label = f"Agent {i}"
    #
    #         plt.plot(
    #             timesteps,
    #             cumulative_profit,
    #             marker="o",
    #             linestyle="-",
    #             label=label,
    #             color=color,
    #             alpha=0.7,
    #         )
    #
    #     plt.title("Cumulative Profit/Loss per Agent (RL Target)")
    #     plt.xlabel("Timestep")
    #     plt.ylabel("Cumulative Profit/Loss (units)")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    def plot_bid_ask_curves(self, num_plots=10):
        """
        Generates bid-ask curves for the last N timesteps, splitting the total
        into multiple figures for better readability.
        """
        if len(self.market_orders_history) < 1:
            print("Not enough market history to plot bid-ask curves.")
            return

        start_index = max(0, len(self.market_orders_history) - num_plots)
        plot_data = []

        for t in range(start_index, len(self.market_orders_history)):
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

        PLOTS_PER_FIGURE = 10
        N_COLS = 5
        N_ROWS = math.ceil(PLOTS_PER_FIGURE / N_COLS)
        BID_COLOR, OFFER_COLOR = "b", "r"
        num_figures = math.ceil(TOTAL_PLOTS / PLOTS_PER_FIGURE)

        for fig_index in range(num_figures):
            chunk_start = fig_index * PLOTS_PER_FIGURE
            chunk_end = min((fig_index + 1) * PLOTS_PER_FIGURE, TOTAL_PLOTS)
            current_chunk = plot_data[chunk_start:chunk_end]
            num_in_chunk = len(current_chunk)
            actual_rows = 1 if num_in_chunk <= N_COLS else N_ROWS
            fig, axes = plt.subplots(
                actual_rows,
                N_COLS,
                figsize=(15, 3 * actual_rows + 1),
                constrained_layout=True,
            )
            axes = np.array(axes).flatten()

            for i, data in enumerate(current_chunk):
                ax = axes[i]
                if len(data["bids"]) > 0:
                    bids_sorted = data["bids"][data["bids"][:, 1].argsort()[::-1]]
                    current_qty = 0.0
                    for j in range(len(bids_sorted)):
                        price, qty = bids_sorted[j, 1], bids_sorted[j, 2]
                        next_qty = current_qty + qty
                        ax.plot(
                            [current_qty, next_qty],
                            [price, price],
                            color=BID_COLOR,
                            lw=2.0,
                        )
                        if j < len(bids_sorted) - 1:
                            next_price = bids_sorted[j + 1, 1]
                            ax.plot(
                                [next_qty, next_qty],
                                [price, next_price],
                                color="gray",
                                lw=0.8,
                                ls="--",
                            )
                        current_qty = next_qty

                if len(data["offers"]) > 0:
                    offers_sorted = data["offers"][data["offers"][:, 1].argsort()]
                    current_qty = 0.0
                    for j in range(len(offers_sorted)):
                        price, qty = offers_sorted[j, 1], offers_sorted[j, 2]
                        next_qty = current_qty + qty
                        ax.plot(
                            [current_qty, next_qty],
                            [price, price],
                            color=OFFER_COLOR,
                            lw=2.0,
                        )
                        if j < len(offers_sorted) - 1:
                            next_price = offers_sorted[j + 1, 1]
                            ax.plot(
                                [next_qty, next_qty],
                                [price, next_price],
                                color="gray",
                                lw=0.8,
                                ls="--",
                            )
                        current_qty = next_qty

                if data["price"] > 0 or data["quantity"] > 0:
                    ax.plot(
                        data["quantity"], data["price"], "go", markersize=6, zorder=5
                    )
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

                proxy_handles = [
                    plt.Line2D([0], [0], color=BID_COLOR, lw=3, label="Demand"),
                    plt.Line2D([0], [0], color=OFFER_COLOR, lw=3, label="Supply"),
                    plt.Line2D(
                        [0],
                        [0],
                        color="g",
                        marker="o",
                        ls="",
                        ms=6,
                        label=f"Clear (${data['price']:.2f})",
                    ),
                ]
                ax.set_title(f"T={data['t']}", fontsize=10)
                ax.set_xlabel("Quantity (MWh)", fontsize=8)
                ax.set_ylabel("Price ($)", fontsize=8)
                ax.legend(handles=proxy_handles, fontsize=7)
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

            for j in range(num_in_chunk, len(axes)):
                fig.delaxes(axes[j])
            start_t, end_t = current_chunk[0]["t"], current_chunk[-1]["t"]
            fig.suptitle(f"Bid-Ask Curves (Timesteps {start_t}-{end_t})", fontsize=14)

        plt.show()

    # def plot_price_change_for_single_day(self, day: int = 0):
    #     """
    #     Plots the market clearing price for each hour of a specific simulation day.
    #     """
    #     plt.figure(figsize=(10, 5))
    #     start_index, end_index = day * 24, (day + 1) * 24
    #
    #     if self.current_timestep < start_index:
    #         print(f"Simulation has not reached day {day}. No data to plot.")
    #         return
    #
    #     daily_prices = self.clearing_prices[start_index:end_index]
    #     hours = range(len(daily_prices))
    #
    #     plt.plot(hours, daily_prices, marker="o", linestyle="-", color="c")
    #     plt.title(f"Market Clearing Price vs. Hour for Day {day}")
    #     plt.xlabel("Hour of Day")
    #     plt.ylabel("Clearing Price ($)")
    #     plt.grid(True, linestyle="--", alpha=0.6)
    #     plt.xticks(np.arange(0, 24, 2))
    #     plt.xlim(-0.5, 23.5)
    #     plt.tight_layout()
    #     plt.show()
    #
    def plot_consumption_and_costs(self):
        """
        Plots:
        1) total preferred consumption vs total actual consumption over time
        2) total price paid per timestep
        3) cumulative total price paid over time
        """
        T = len(self.clearing_quantities)
        if T == 0:
            print("No history to plot (run an episode first).")
            return
        timesteps = list(range(1, T))

        initial_net_demand = np.zeros(T)
        actual_net_demand = np.zeros(T)
        for agent in self.agents:
            init, actual = agent.get_demand_consumption()
            initial_net_demand = [initial_net_demand[i] + val for i, val in enumerate(init)]
            actual_net_demand = [actual_net_demand[i] + val for i, val in enumerate(actual)]


        # print(f"required energy {sum(initial_net_demand)}")
        # print(f"actual demand {sum(actual_net_demand)}")
        # print(f"total met demand: {sum(self.clearing_quantities)}")
        # print(f"total met demand: {2 * sum(self.clearing_quantities) + sum(actual_net_demand)}")

        # Convert to numpy arrays for convenience
        preferred = np.array(np.abs(self.total_demand), dtype=float)
        actual = np.array(self.clearing_quantities, dtype=float)
        price_paid = np.array(self.total_price_paid_history, dtype=float)
        cumulative_paid = np.array(self.cumulative_price_paid_history, dtype=float)

        # Plot 1: preferred vs actual consumption
        plt.figure(figsize=(10, 5))
        plt.plot(
            timesteps,
            initial_net_demand[:-1],
            marker="o",
            linestyle="-",
            label="Preferred net demand",
        )
        plt.plot(
            timesteps,
            actual_net_demand[:-1],
            marker="o",
            linestyle="-",
            label="Actual net demand",
        )
        plt.title("Total Preferred vs Actual Net Demand Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Energy (MWh)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # # Plot 2: total price paid per timestep
        # plt.figure(figsize=(10, 4))
        # plt.bar(timesteps, price_paid)
        # plt.title("Total Price Paid by Buyers per Timestep")
        # plt.xlabel("Timestep")
        # plt.ylabel("Price Paid (monetary units)")
        # plt.grid(axis="y", linestyle=":", alpha=0.6)
        # plt.tight_layout()
        # # plt.show()
        #
        # # Plot 3: cumulative price paid over time
        # plt.figure(figsize=(10, 4))
        # plt.plot(timesteps, cumulative_paid, marker="o", linestyle="-")
        # plt.title("Cumulative Total Price Paid Over Time")
        # plt.xlabel("Timestep")
        # plt.ylabel("Cumulative Price Paid (monetary units)")
        # plt.grid(True, linestyle="--", alpha=0.6)
        # plt.tight_layout()
        # plt.show()


class FlexibilityMarketEnv(DoubleAuctionEnv):
    @override
    def __init__(
            self,
            agent_configs: list[dict[str, Any]],
            market_clearing_agent: MarketClearingAgent,
            discount: tuple[float, float],
            max_timesteps: int = 100,
    ):
        super().__init__(
            agent_configs, market_clearing_agent, buy_tariff, sell_tariff, max_timesteps
        )
        self.costs = 0
        self.min = 1000
        self.discount = discount

    @override
    def _forecast_prices(self, actions: dict[int, np.ndarray]) -> list[float]:
        """
        Simulates market clearing for future timesteps (t+1 to t+23) based on
        submitted actions to generate a price forecast.
        """
        forecasted_prices = []
        for h in range(1, self.FORECAST_HORIZON):
            future_bids, future_offers = [], []
            for agent in self.agents:
                agent_id = agent.agent_id
                if agent_id in actions:
                    price, quantity = actions[agent_id][0][h]
                    if quantity == 0:
                        price, quantity = actions[agent_id][1][h]
                        if quantity > 0:
                            future_offers.append((agent_id, price, quantity))
                    else:
                        future_bids.append((agent_id, price, quantity))

            bids_arr = (
                np.array(future_bids, dtype=float) if future_bids else np.empty((0, 3))
            )
            offers_arr = (
                np.array(future_offers, dtype=float)
                if future_offers
                else np.empty((0, 3))
            )

            price, quantity, _ = self.market_agent.clear_market(bids_arr, offers_arr)
            if quantity < self.discount[1]:
                price = price * self.discount[0]
                forecasted_prices.append(price)
            else:
                forecasted_prices.append(price)

        return forecasted_prices

    @override
    def _get_obs(
            self, price_forecast: list[float]
    ) -> dict[int, dict[str, np.ndarray]]:
        """Compiles dictionary observations for each agent."""
        market_stats_arr = np.array(
            [
                self.last_clearing_price,
                self.last_clearing_quantity,
                self.last_total_bids_qty,
                self.last_total_offers_qty,
            ],
            dtype=np.float32,
        )

        observations: dict[int, dict[str, np.ndarray]] = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_state_arr = np.array(
                [self.last_cleared_quantities[agent_id]],
                dtype=np.float32,
            )

            observations[agent_id] = {
                "agent_state": agent_state_arr,
                "market_stats": market_stats_arr,
                "price_forecast": np.array([price for price in price_forecast]),
            }
        return observations
