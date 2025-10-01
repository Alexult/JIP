from typing import override, Any
import numpy as np
from .custom_types import *
from gymnasium import Env
from gymnasium.spaces import Box
from abc import ABC, abstractmethod


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


class CentralMarket(BaseMarket):
    """
    Manages the double auction process for the energy market.
    """

    def __init__(self):
        super().__init__()


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
        """
        # Check if arrays are empty
        if bids_array.size == 0 or offers_array.size == 0:
            return 0.0, 0.0

        # Bids (Demand) sorted descending by price (highest price first)
        bids_sorted = bids_array[bids_array[:, 0].argsort()[::-1]]
        # Offers (Supply) sorted ascending by price (lowest price first)
        offers_sorted = offers_array[offers_array[:, 0].argsort()]

        # Extract prices and quantities
        bid_prices = bids_sorted[:, 0]
        bid_quantities = bids_sorted[:, 1]
        offer_prices = offers_sorted[:, 0]
        offer_quantities = offers_sorted[:, 1]

        # Find the Marginal Match Index (k)
        # We only need to compare up to the length of the shorter list
        min_len = min(len(bid_prices), len(offer_prices))

        # Find indices where Bid Price >= Offer Price. These orders are profitable to match.
        match_indices = np.where(bid_prices[:min_len] >= offer_prices[:min_len])[0]

        if len(match_indices) == 0:
            # Market does not clear (highest bid < lowest offer)
            return 0.0, 0.0

        # k is the index of the marginal order that defines the clearing price
        # This order is the one corresponding to the largest index where a match occurs.
        k = match_indices[-1]

        # Determine Clearing Price and Quantity
        # Marginal Bid/Offer Prices define the clearing price range
        marginal_bid_price = bid_prices[k]
        marginal_offer_price = offer_prices[k]

        # Clearing price is set as the midpoint of the marginal bid and offer
        clearing_price = (marginal_bid_price + marginal_offer_price) / 2.0

        # The clearing quantity is the total quantity traded up to the marginal index k
        # Calculate cumulative quantities for accepted orders (indices 0 to k, inclusive)
        bid_cumsum = np.cumsum(bid_quantities[: k + 1])
        offer_cumsum = np.cumsum(offer_quantities[: k + 1])

        # The clearing quantity is the minimum of the total demand or supply at the marginal price.
        # This is the last value in the cumulative sums.
        clearing_quantity = min(bid_cumsum[-1], offer_cumsum[-1])

        return clearing_price, clearing_quantity


class DoubleAuctionEnv(Env):
    """
    Gymnasium-compatible environment for a Double Auction Energy Market Simulation.
    for a Multi-Agent RL (MARL) setup where each ProsumerAgent is a separate agent.
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

        # Public Market Stats for Observation (P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1)
        self.last_clearing_price = 5.0  # Historical Price P_t-1
        self.last_clearing_quantity = 0.0  # Historical Quantity Q_t-1
        self.last_total_bids_qty = 0.0
        self.last_total_offers_qty = 0.0

        # Initialize agents (Agent 0 is the aggressive seller for demonstration)
        for i, config in enumerate(agent_configs):
            if i == 0:
                self.agents.append(
                    AggressiveSellerAgent(
                        agent_id=i,
                        fixed_load=config["fixed_load"],
                        flexible_load_max=config["flexible_load_max"],
                        generation_capacity=config["generation_capacity"],
                    )
                )
            else:
                self.agents.append(
                    ProsumerAgent(
                        agent_id=i,
                        fixed_load=config["fixed_load"],
                        flexible_load_max=config["flexible_load_max"],
                        generation_capacity=config["generation_capacity"],
                    )
                )

        # --- Define Action Space (Per Agent) ---
        # Action: [Price, Quantity] (2 size)
        MAX_PRICE = 10.0
        MAX_QTY = 50.0

        low_action = np.array([0.0, 0.0], dtype=np.float32)
        high_action = np.array([MAX_PRICE, MAX_QTY], dtype=np.float32)

        # NOTE: The action space is now for a SINGLE agent.
        self.action_space = Box(
            low=low_action, high=high_action, shape=(2,), dtype=np.float32
        )

        # --- Define Observation Space (Per Agent) ---
        # Observation: [ND_i, P_t-1 (previous clearing price), Q_t-1 (previous capacity), Sum_Bids_t-1, Sum_Offers_t-1] (5 features)

        # Calculate bounds for Net Demand (ND)
        MAX_DEMAND_ABS = max(
            config["fixed_load"] + config["flexible_load_max"]
            for config in agent_configs
        )
        MAX_CAPACITY = max(config["generation_capacity"] for config in agent_configs)
        MAX_NET_DEMAND = MAX_DEMAND_ABS
        MIN_NET_DEMAND = -MAX_CAPACITY

        # Calculate bounds for Market Stats
        MAX_SUM_QTY = self.n_agents * MAX_QTY

        # Observation features: ND_i, P_t-1, Q_t-1, Sum_Bids_t-1, Sum_Offers_t-1
        low_obs = [MIN_NET_DEMAND, 0.0, 0.0, 0.0, 0.0]
        high_obs = [MAX_NET_DEMAND, MAX_PRICE, MAX_QTY, MAX_SUM_QTY, MAX_SUM_QTY]

        # NOTE: The observation space is now for a SINGLE agent.
        self.observation_space = Box(
            low=np.array(low_obs, dtype=np.float32),
            high=np.array(high_obs, dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

        # History logging for plotting outside the RL loop
        self.clearing_prices = []
        self.clearing_quantities = []
        self.profit_history = []
        self.net_demand_history = []
        self.action_history = []

    def _get_obs(self) -> dict[int, np.ndarray]:
        """
        Compiles the current observation vector for each agent.
        Returns a dictionary mapping agent ID to its 5-feature observation array.
        """
        market_stats = [
            self.last_clearing_price,
            self.last_clearing_quantity,
            self.last_total_bids_qty,
            self.last_total_offers_qty,
        ]

        observations: dict[int, np.ndarray] = {}
        for agent in self.agents:
            # Agent's private info: Net Demand
            private_info = [agent.net_demand]

            # Agent's observation: [ND_i] + [Public Market Stats]
            obs_i = private_info + market_stats
            observations[agent.agent_id] = np.array(obs_i, dtype=np.float32)

        return observations

    def _calculate_rewards(self, clearing_price: float) -> dict[int, float]:
        """Calculates the reward (profit) for all agents."""
        rewards: dict[int, float] = {}
        for agent in self.agents:
            rewards[agent.agent_id] = agent.calculate_profit(clearing_price)
        return rewards

    @override
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        """Resets the environment to an initial state, returning agent-specific observations."""
        super().reset(seed=seed)
        self.current_timestep = 0

        # Reset public market stats
        self.last_clearing_price = 5.0
        self.last_clearing_quantity = 0.0
        self.last_total_bids_qty = 0.0
        self.last_total_offers_qty = 0.0

        # Reset history
        self.clearing_prices = []
        self.clearing_quantities = []
        self.profit_history = []
        self.net_demand_history = []
        self.action_history = []

        # Update initial net demands for all agents
        for agent in self.agents:
            agent.calculate_net_demand()
            agent.profit = 0.0  # Reset internal profit state

        observation = self._get_obs()
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
        Runs one time step of the environment's dynamics, taking a dictionary of actions.

        Args:
            actions (dict[int, np.ndarray]): A dictionary mapping agent ID to its [Price, Quantity] action.

        Returns (MARL Format):
            observation (dict[int, np.ndarray]): New state for each agent.
            reward (dict[int, float]): Reward for each agent.
            terminated (dict[int, bool]): Termination status for each agent.
            truncated (dict[int, bool]): Truncation status for each agent.
            info (dict): Additional information.
        """
        self.current_timestep += 1

        # Parse Action and Compile Market Orders
        all_bids, all_offers = [], []

        # Action history logs a list of all actions taken in this step
        current_step_actions = []

        for agent in self.agents:
            agent_id = agent.agent_id
            if agent_id in actions:
                price, quantity = actions[agent_id]
                bids, offers = agent.get_market_submission(price, quantity)
                all_bids.extend(bids)
                all_offers.extend(offers)
                current_step_actions.append(actions[agent_id])
            else:
                # Handle missing action (e.g., if an agent is done, though here all run simultaneously)
                pass

        # Calculate total quantities for logging/next observation
        total_bids_qty = sum(q for _, q in all_bids)
        total_offers_qty = sum(q for _, q in all_offers)

        # Log action (flat array of all actions for plotting simplicity)
        self.action_history.append(
            np.concatenate(current_step_actions)
            if current_step_actions
            else np.array([])
        )

        # Market Clearing
        bids_array = np.array(all_bids, dtype=float) if all_bids else np.array([])
        offers_array = np.array(all_offers, dtype=float) if all_offers else np.array([])

        clearing_price, clearing_quantity = self.market_agent.clear_market(
            bids_array, offers_array
        )

        reward = self._calculate_rewards(clearing_price)

        # Update Public State for Next Timestep's Observation
        self.last_clearing_price = (
            clearing_price  # Storing P_t as P_t-1 for the next step
        )
        self.last_clearing_quantity = clearing_quantity
        self.last_total_bids_qty = total_bids_qty
        self.last_total_offers_qty = total_offers_qty

        # Generate next timestep's net demand (the private information for the next round)
        for agent in self.agents:
            agent.calculate_net_demand()

        # Check Termination and Get Observation/Info
        is_truncated = self.current_timestep >= self.max_timesteps

        # For MARL, termination/truncation status is usually specified per agent
        terminated = {i: False for i in self.agent_ids}
        truncated = {i: is_truncated for i in self.agent_ids}

        observation = self._get_obs()

        # Log history
        self.clearing_prices.append(clearing_price)
        self.clearing_quantities.append(clearing_quantity)
        self.profit_history.append(
            np.array(list(reward.values()))
        )  # Log as numpy array for plotting
        self.net_demand_history.append([agent.net_demand for agent in self.agents])

        info = {
            "timestep": self.current_timestep,
            "clearing_price": clearing_price,
            "clearing_quantity": clearing_quantity,
        }

        # The new MARL return format
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Placeholder for rendering logic, usually involving Matplotlib/Pygame.
        In this case, we just print the state.
        """
        if mode == "human":
            last_total_reward = (
                self.profit_history[-1].sum() if self.profit_history else 0.0
            )
            print(
                f"T={self.current_timestep:02d} | Price={self.last_clearing_price:.2f} | Qty={self.clearing_quantities[-1]:.2f} | Rewards={last_total_reward:.2f}"
            )

    def plot_results(self):
        """Generates and displays graphs of the simulation results from history."""

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
        plt.show()

        # Plot 2: Cumulative Profit/Loss per Agent
        plt.figure(figsize=(10, 6))
        profit_matrix = np.array(self.profit_history)

        for i in range(self.n_agents):
            cumulative_profit = np.cumsum(profit_matrix[:, i])
            # Highlighting Agent 0 (Aggressive Seller)
            color = "r" if i == 0 else "b"
            label = (
                f"Agent {i} (Aggressive Seller)" if i == 0 else f"Agent {i} (Default)"
            )
            plt.plot(
                timesteps,
                cumulative_profit,
                marker="o",
                linestyle="-",
                label=label,
                color=color,
            )

        plt.title("Cumulative Profit/Loss per Agent (RL Target)")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Profit/Loss (units)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


class P2P(BaseMarket):
    def __init__(self, market_price):
        BaseMarket.__init__(self)
