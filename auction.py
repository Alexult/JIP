import random
import matplotlib.pyplot as plt

Bid = tuple[float, float, int]  # (price, quantity, agent_id)
Offer = tuple[float, float, int]  # (price, quantity, agent_id)
Trade = tuple[float, float, int, int]  # (price, quantity, buyer_id, seller_id)


class BaseProsumerAgent:
    """
    Represents a prosumer (producer + consumer) agent in the energy market.
    This base class contains common functionality and data logging.
    """

    def __init__(
        self,
        agent_id: int,
        fixed_load: float,
        flexible_load_max: float,
        generation_capacity: float,
    ):
        self.agent_id = agent_id
        self.fixed_load = fixed_load
        self.flexible_load_max = flexible_load_max
        self.generation_capacity = generation_capacity
        self.current_flexible_load = 0.0
        self.net_demand = 0.0
        self.bid_history = []
        self.offer_history = []
        self.net_demand_history = []
        self.profit_history = []

    def update_state(self):
        """
        Simulates the agent's internal state for a new timestep.
        The flexible load varies, and the net demand is calculated.
        """
        self.current_flexible_load = random.uniform(0, self.flexible_load_max)
        total_load = self.fixed_load + self.current_flexible_load
        self.net_demand = total_load - self.generation_capacity
        self.net_demand_history.append(self.net_demand)

    def devise_strategy(self) -> tuple[list[Bid], list[Offer]]:
        """
        Generates bids and offers based on a simple heuristic.
        This method is designed to be overridden by subclasses.
        """
        bids: list[Bid] = []
        offers: list[Offer] = []

        if self.net_demand > 0:
            # Simple buyer strategy
            bid_price = random.uniform(0.1, 10.0)
            bids.append((bid_price, self.net_demand, self.agent_id))
        elif self.net_demand < 0:
            # Simple seller strategy
            offer_price = random.uniform(0.1, 10.0)
            offers.append((offer_price, abs(self.net_demand), self.agent_id))

        self.bid_history.append((bids[0] if bids else None))
        self.offer_history.append((offers[0] if offers else None))

        return bids, offers

    def calculate_profit(self, trades: list[Trade]):
        """
        Calculates the agent's profit or loss for the current timestep based on
        the trades they were involved in.
        """
        timestep_profit = 0.0
        # Check if the agent submitted a bid or an offer in the last timestep
        last_bid = (
            self.bid_history[-1] if self.bid_history and self.bid_history[-1] else None
        )
        last_offer = (
            self.offer_history[-1]
            if self.offer_history and self.offer_history[-1]
            else None
        )

        for trade in trades:
            price, qty, buyer_id, seller_id = trade

            if buyer_id == self.agent_id:
                # This agent was a buyer in the trade.
                # Profit is the consumer surplus (willingness to pay - price paid)
                if last_bid:
                    timestep_profit += (last_bid[0] - price) * qty
            elif seller_id == self.agent_id:
                # This agent was a seller in the trade.
                # Profit is the producer surplus (price received - cost of production)
                if last_offer:
                    timestep_profit += (price - last_offer[0]) * qty

        self.profit_history.append(timestep_profit)


class AggroBuyDude(BaseProsumerAgent):
    """
    An agent that is aggressive in its buying strategy, always bidding a high price.
    """

    def devise_strategy(self) -> tuple[list[Bid], list[Offer]]:
        bids: list[Bid] = []
        offers: list[Offer] = []

        if self.net_demand > 0:
            # Bid a high, fixed price to ensure purchase
            bid_price = 10.0
            bids.append((bid_price, self.net_demand, self.agent_id))

        self.bid_history.append((bids[0] if bids else None))
        self.offer_history.append((offers[0] if offers else None))

        return bids, offers


class SelfishSellerDude(BaseProsumerAgent):
    """
    An agent that is conservative and only sells at a high price.
    """

    def devise_strategy(self) -> tuple[list[Bid], list[Offer]]:
        bids: list[Bid] = []
        offers: list[Offer] = []

        if self.net_demand < 0:
            # Offer at a high, fixed price to maximize profit
            offer_price = 9.0
            offers.append((offer_price, abs(self.net_demand), self.agent_id))

        self.bid_history.append((bids[0] if bids else None))
        self.offer_history.append((offers[0] if offers else None))

        return bids, offers


# TODO: Use the damn priority queue and make an orderbook then you can use
# multithreaded parallel agent => multi agent game theory, 
# also can do price discovery tactic for clearing price
class DoubleAuctionMarket:
    """
    Manages the double auction process for the energy market.
    """

    def __init__(self, agents: list[BaseProsumerAgent]):
        self.agents = agents

    def run_timestep(self) -> tuple[float, float, list[Trade]]:
        """
        Executes a single timestep of the market simulation.
        1. Agents update their states.
        2. Agents submit bids and offers.
        3. The market performs individual matches and logs trades.
        4. Results are returned.
        """

        # Update agent states for the new timestep
        for agent in self.agents:
            agent.update_state()

        # Collect all bids and offers
        all_bids: list[Bid] = []
        all_offers: list[Offer] = []
        for agent in self.agents:
            bids, offers = agent.devise_strategy()
            all_bids.extend(bids)
            all_offers.extend(offers)

        # Clear the market
        trades, total_traded_qty = self.match_bids_and_offers(all_bids, all_offers)

        # Calculate average trade price for logging
        avg_price = sum(t[0] for t in trades) / len(trades) if trades else 0

        return avg_price, total_traded_qty, trades

    def match_bids_and_offers(
        self, bids: list[Bid], offers: list[Offer]
    ) -> tuple[list[Trade], float]:
        """
        Matches individual bids and offers to create trades.

        The algorithm sorts bids and offers and matches them one-by-one,
        simulating a continuous double auction.
        """
        # Sort bids in descending order (highest price first)
        sorted_bids = sorted(bids, key=lambda x: x[0], reverse=True)

        # Sort offers in ascending order (lowest price first)
        sorted_offers = sorted(offers, key=lambda x: x[0])

        trades: list[Trade] = []
        total_traded_quantity = 0.0

        # While there are still bids and offers to consider, and a trade is possible
        bid_index, offer_index = 0, 0
        while bid_index < len(sorted_bids) and offer_index < len(sorted_offers):
            bid_price, bid_qty, bid_agent_id = sorted_bids[bid_index]
            offer_price, offer_qty, offer_agent_id = sorted_offers[offer_index]

            # Check for a successful match
            if bid_price >= offer_price:
                # A trade occurs
                traded_qty = min(bid_qty, offer_qty)

                # The transaction price is the average of the bid and offer price
                transaction_price = (bid_price + offer_price) / 2

                trades.append(
                    (transaction_price, traded_qty, bid_agent_id, offer_agent_id)
                )
                total_traded_quantity += traded_qty

                # Update remaining quantities
                sorted_bids[bid_index] = (bid_price, bid_qty - traded_qty, bid_agent_id)
                sorted_offers[offer_index] = (
                    offer_price,
                    offer_qty - traded_qty,
                    offer_agent_id,
                )

                # Move to the next bid/offer if the current one is fully matched
                if bid_qty - traded_qty <= 0:
                    bid_index += 1
                if offer_qty - traded_qty <= 0:
                    offer_index += 1
            else:
                # No more profitable matches can be made
                break

        return trades, total_traded_quantity


class MarketSimulation:
    """
    Manages the overall market simulation, including data logging and visualization.
    """

    def __init__(self, agents: list[BaseProsumerAgent], num_timesteps: int):
        self.agents = agents
        self.num_timesteps = num_timesteps
        self.market = DoubleAuctionMarket(self.agents)
        self.clearing_prices = []
        self.clearing_quantities = []
        self.all_bids_history = []
        self.all_offers_history = []

    def run_simulation(self):
        """Runs the market simulation for the specified number of timesteps."""
        print("Starting Double Auction Energy Market Simulation...")
        for t in range(self.num_timesteps):
            print(f"\n--- Timestep {t + 1} ---")
            avg_price, total_qty, trades = self.market.run_timestep()

            # Update agent financials
            for agent in self.agents:
                agent.calculate_profit(trades)

            self.clearing_prices.append(avg_price)
            self.clearing_quantities.append(total_qty)

            if total_qty > 0:
                print(f"Market Cleared! {len(trades)} trades occurred.")
                print(f"Average Trade Price: {avg_price:.2f} units")
                print(f"Total Traded Quantity: {total_qty:.2f} MWh")
            else:
                print("Market did not clear. No trades occurred.")

        print("\nSimulation finished.")

    def plot_results(self):
        """Generates and displays graphs of the simulation results."""
        self.plot_market_metrics()
        self.plot_agent_metrics()
        self.plot_agent_profit_loss()

    def plot_market_metrics(self):
        """Plots the average trade price and total traded quantity over time."""
        timesteps = range(1, self.num_timesteps + 1)

        # Plot Clearing Price
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(timesteps, self.clearing_prices, marker="o", linestyle="-", color="b")
        plt.title("Average Trade Price Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Price (units)")
        plt.grid(True)

        # Plot Clearing Quantity
        plt.subplot(1, 2, 2)
        plt.plot(
            timesteps, self.clearing_quantities, marker="o", linestyle="-", color="g"
        )
        plt.title("Total Traded Quantity Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Quantity (MWh)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_agent_metrics(self):
        """Plots the individual agent's net demand, bids, and offers over time."""
        timesteps = range(1, self.num_timesteps + 1)

        plt.figure(figsize=(15, 10))
        plt.suptitle("Individual Agent Metrics Over Time", fontsize=16)

        num_cols = min(4, len(self.agents))
        num_rows = (len(self.agents) + num_cols - 1) // num_cols

        for i, agent in enumerate(self.agents):
            plt.subplot(num_rows, num_cols, i + 1)

            # Plot Net Demand
            plt.plot(
                timesteps,
                agent.net_demand_history,
                label="Net Demand",
                color="black",
                linestyle="--",
            )

            # Plot Bids
            bid_prices = [b[0] if b else None for b in agent.bid_history]
            plt.plot(
                timesteps,
                bid_prices,
                marker="o",
                linestyle=":",
                color="red",
                label="Bid Price",
            )

            # Plot Offers
            offer_prices = [o[0] if o else None for o in agent.offer_history]
            plt.plot(
                timesteps,
                offer_prices,
                marker="s",
                linestyle=":",
                color="green",
                label="Offer Price",
            )

            plt.title(f"Agent {agent.agent_id} ({type(agent).__name__})")
            plt.xlabel("Timestep")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_agent_profit_loss(self):
        """Plots the cumulative profit/loss for each agent over time."""
        timesteps = range(1, self.num_timesteps + 1)

        plt.figure(figsize=(10, 6))

        for agent in self.agents:
            # Calculate cumulative profit
            cumulative_profit = [
                sum(agent.profit_history[: i + 1])
                for i in range(len(agent.profit_history))
            ]
            plt.plot(
                timesteps,
                cumulative_profit,
                marker="o",
                linestyle="-",
                label=f"Agent {agent.agent_id}",
            )

        plt.title("Cumulative Profit/Loss per Agent")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Profit/Loss (units)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    NUM_TIMESTEPS = 10

    # Create different types of agents
    agents = [
        BaseProsumerAgent(
            agent_id=0, fixed_load=15, flexible_load_max=8, generation_capacity=20
        ),
        AggroBuyDude(
            agent_id=1, fixed_load=18, flexible_load_max=7, generation_capacity=10
        ),
        SelfishSellerDude(
            agent_id=2, fixed_load=12, flexible_load_max=6, generation_capacity=22
        ),
        BaseProsumerAgent(
            agent_id=3, fixed_load=14, flexible_load_max=9, generation_capacity=18
        ),
        AggroBuyDude(
            agent_id=4, fixed_load=19, flexible_load_max=5, generation_capacity=3
        ),
    ]

    simulation = MarketSimulation(agents=agents, num_timesteps=NUM_TIMESTEPS)
    simulation.run_simulation()
    simulation.plot_results()
