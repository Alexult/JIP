import argparse
import random
import json
import os
import perlin_noise
import numpy as np
import matplotlib.pyplot as plt

# --- Import Both Environments ---
from wholesale_market import WholesaleMarketEnv
from energymarket import (
    FlexibilityMarketEnv,
    DoubleAuctionClearingAgent,
)
from loguru import logger

# Use a consistent number of steps for comparison
MAX_STEPS = 50
GENERATION_TYPES = ["solar", "wind", "none"]
FLEXIBILITY_SCALE_FACTOR = 1000


# --- Agent Generation Functions (for consistency) ---


def generate_agents(n=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    agents = []
    for i in range(n):
        load = generate_load()
        generation_capacity = random.randint(150, 200)
        generation_type = random.choice(GENERATION_TYPES)
        agents.append(
            {
                "id": i,
                "load": load,
                "generation_capacity": generation_capacity,
                "generation_type": generation_type,
                # This marginal_price is used by ProsumerAgent for selling
                "marginal_price": random.randint(800, 1200) / 10000,
            }
        )
    return agents


def generate_load():
    noise = perlin_noise.PerlinNoise(octaves=2, seed=random.randint(0, 1000))
    # Generate load for more than just 24 steps
    scale = random.uniform(20, 80)
    # Generate for *2 to ensure ProsumerAgent strategy doesn't go out of bounds
    y = [
        scale * (noise(i * 0.05) * 0.5 + 0.6) + random.uniform(-5, 5)
        for i in range(MAX_STEPS)
    ]
    return [max(0, val) for val in y]  # Ensure no negative load


def save_agents_to_json(agents, filename="agents_100.json"):
    with open(filename, "w") as f:
        json.dump(agents, f, indent=2)
    print(f"Stored {len(agents)} agents in {filename}")
    return filename


def load_agents_from_json(filename="agents_100.json"):
    with open(filename, "r") as f:
        return json.load(f)


def convert_json_agents_configs(json_agents):
    """
    Uses the more comprehensive converter from main_wholesale.py
    to ensure compatibility with both envs.
    """
    configs = []
    for j in json_agents:
        config = {
            "load": j["load"],
            "generation_capacity": int(j["generation_capacity"]),
            "marginal_price": float(j["marginal_price"]),
            "generation_type": j.get("generation_type", "none"),
            # Add other fields for compatibility
            "flexible_load": j.get("flexible_load", []),
            "fixed_load": j.get("fixed_load", []),
        }
        configs.append(config)
    return configs


# --- Episode Runners ---


def run_episode_wholesale(
    agent_configs, max_steps=MAX_STEPS, buy_tariff=0.1, sell_tariff=0.1
):
    """
    Runs the WholesaleMarketEnv and returns its history for plotting.
    """
    # Create a *fresh* set of agent configs to avoid state pollution
    configs_copy = convert_json_agents_configs(json.loads(json.dumps(agent_configs)))

    env = WholesaleMarketEnv(
        agent_configs=configs_copy,
        wholesale_csv_path="./data/representative_wholesale_price_2025.csv",
        max_timesteps=max_steps,
        buy_tariff=buy_tariff,
        sell_tariff=sell_tariff,
    )

    logger.info(f"Starting Wholesale Market Episode ({max_steps} steps)")
    observations, info = env.reset()
    all_terminated = {i: False for i in env.agent_ids}
    all_truncated = {i: False for i in env.agent_ids}
    total_agent_rewards = {i: 0.0 for i in env.agent_ids}  # Cumulative rewards

    t = 0
    while not all(all_terminated.values()) and not all(all_truncated.values()):
        time_step = t
        actions = {}
        for agent_id in env.agent_ids:
            if agent_id in observations:
                obs_i = observations[agent_id]
                actions[agent_id] = env.agents[agent_id].devise_strategy(
                    obs_i, env.action_space, time_step
                )

        observations, rewards, all_terminated, all_truncated, info = env.step(actions)

        for agent_id, reward in rewards.items():
            total_agent_rewards[agent_id] += reward

        if t % 24 == 0:
            env.render()
        t += 1

        if all(all_terminated.values()) or all(all_truncated.values()):
            break

    total_system_profit = sum(total_agent_rewards.values())
    logger.success(
        f"Wholesale Episode Finished. Total System Profit/Loss: €{total_system_profit:.2f}"
    )

    # Return the collected history
    initial_net_demand, actual_net_demand, _ = env.plot_consumption_and_costs(True)
    return {
        "prices": env.wholesale_prices_history,
        "cumulative_cost": env.cumulative_price_paid_history,
        "price_paid_history": env.total_price_paid_history,
        "initial_net_demand": initial_net_demand,
        "actual_net_demand": actual_net_demand,
        "total_profit": total_system_profit,
    }


def run_episode_flexibility(
    agent_configs, max_steps=MAX_STEPS, buy_tariff=0.1, sell_tariff=0.1
):
    """
    Runs the FlexibilityMarketEnv and returns its history for plotting.
    """
    # Create a *fresh* set of agent configs
    configs_copy = convert_json_agents_configs(json.loads(json.dumps(agent_configs)))

    env = FlexibilityMarketEnv(
        agent_configs=configs_copy,
        market_clearing_agent=DoubleAuctionClearingAgent(),
        discount=(1, 400),  # Using default discount from main.py
        max_timesteps=max_steps,
        buy_tariff=buy_tariff,
        sell_tariff=sell_tariff,
    )

    logger.info(f"Starting Flexibility Market Episode ({max_steps} steps)")
    observations, info = env.reset()
    all_terminated = {i: False for i in env.agent_ids}
    all_truncated = {i: False for i in env.agent_ids}
    total_agent_rewards = {i: 0.0 for i in env.agent_ids}  # Cumulative rewards

    t = 0
    while not all(all_terminated.values()) and not all(all_truncated.values()):
        time_step = t
        actions = {}
        for agent_id in env.agent_ids:
            if agent_id in observations:
                obs_i = observations[agent_id]
                actions[agent_id] = env.agents[agent_id].devise_strategy(
                    obs_i, env.action_space, time_step
                )

        observations, rewards, all_terminated, all_truncated, info = env.step(actions)

        for agent_id, reward in rewards.items():
            total_agent_rewards[agent_id] += reward

        if t % 24 == 0:
            env.render()
        t += 1

        if all(all_terminated.values()) or all(all_truncated.values()):
            break

    total_system_profit = sum(total_agent_rewards.values())
    logger.success(
        f"Flexibility Episode Finished. Total System Profit/Loss: €{total_system_profit:.2f}"
    )

    env.plot_bid_ask_curves(4)

    # Manually calculate demand profiles
    # T = env.current_timestep
    # initial_net_demand_total = np.zeros(T)
    # actual_net_demand_total = np.zeros(T)
    #
    # for agent in env.agents:
    #     init, actual, _ = agent.get_demand_consumption()
    #     # Truncate to the number of steps actually run
    #     init = init[:T]
    #     actual = actual[:T]
    #     # Add to totals
    #     initial_net_demand_total[: len(init)] += init
    #     actual_net_demand_total[: len(actual)] += actual

    # Return the collected history

    initial_net_demand, actual_net_demand, _ = env.plot_consumption_and_costs(True)
    return {
        "prices": env.clearing_prices,
        "cumulative_cost": env.cumulative_price_paid_history,
        "price_paid_history": env.total_price_paid_history,
        "initial_net_demand": initial_net_demand,
        "actual_net_demand": actual_net_demand,
        "total_profit": total_system_profit,
        "quantity": env.clearing_quantities,
    }


# --- NEW Merged Plotting Function (with scaling) ---


def plot_comparison_results(wh_results: dict, fl_results: dict):
    """
    Generates plots comparing results from Wholesale and Flexibility markets.
    Applies a factor of 10x to Flexibility Market results for unit consistency.
    """
    logger.info("Generating comparison plots...")

    # Define the scaling factor for flexibility market results

    # Ensure data lengths match for plotting
    steps_ran = min(len(wh_results["prices"]), len(fl_results["prices"]))
    if steps_ran == 0:
        logger.error("No data to plot. Both simulations might have failed.")
        return

    timesteps = range(steps_ran)

    # --- Data Preparation with Scaling ---

    # Prices
    wh_prices = np.array(wh_results["prices"][:steps_ran])
    # Apply 10x scale
    fl_prices = np.array(fl_results["prices"][:steps_ran]) * FLEXIBILITY_SCALE_FACTOR

    fl_quantity = np.array(fl_results["quantity"][:steps_ran])

    # Cumulative Cost
    wh_cum_cost = np.array(wh_results["cumulative_cost"][:steps_ran])
    # Apply 10x scale
    fl_cum_cost = (
        np.array(fl_results["cumulative_cost"][:steps_ran]) * FLEXIBILITY_SCALE_FACTOR
    )

    wh_tot_cost = np.array(wh_results["price_paid_history"][:steps_ran])
    fl_tot_cost = (
        np.array(fl_results["price_paid_history"][:steps_ran])
        * FLEXIBILITY_SCALE_FACTOR
    )

    # Demand Profiles (no scaling needed here as it's a quantity/MWh, not a price/cost)
    wh_actual_nd = np.array(wh_results["actual_net_demand"][:steps_ran])
    fl_actual_nd = np.array(fl_results["actual_net_demand"][:steps_ran])
    initial_net_demand = np.array(wh_results["initial_net_demand"][:steps_ran])

    # --- Plot 1: Price Comparison ---
    plt.figure(figsize=(12, 6))
    # plt.plot(
    #     timesteps,
    #     wh_prices,
    #     label="Wholesale Price",
    #     color="orange",
    #     linestyle="--",
    #     alpha=0.9,
    # )
    plt.plot(
        timesteps,
        fl_prices,
        label="Local Energy Market Price",
        color="blue",
        linestyle="-",
        alpha=0.9,
    )
    plt.title("LEM Price")
    plt.xlabel("Timestep (Hour)")
    plt.ylabel("Price (€/MWh)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("market_price_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        timesteps,
        fl_quantity,
        label="Local Energy Market Clearing",
        color="blue",
        linestyle="-",
        alpha=0.9,
    )
    plt.title("LEM Clearing Quantity")
    plt.xlabel("Timestep (Hour)")
    plt.ylabel("Quantity (MWh)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("clearing_quantity.png")
    plt.close()

    # --- Plot 2: Cumulative Cost Comparison (Buyers) ---
    plt.figure(figsize=(12, 6))
    plt.plot(
        timesteps,
        wh_cum_cost,
        label="Wholesale Market (Total Cost)",
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        timesteps,
        fl_cum_cost,
        label="Local Energy Market (Total Cost)",
        color="blue",
        linestyle="-",
        linewidth=2,
    )
    # plt.plot(
    #     timesteps,
    #     fl_tot_cost,
    #     label="LEM total cost per timestep",
    #     color="green",
    #     linestyle="-",
    #     linewidth=2,
    # )
    tot = np.sum(fl_tot_cost)
    logger.debug(f"\n\n tot:{tot}, last:{fl_cum_cost[-1]}")
    plt.title("Cumulative System Cost Comparison")
    plt.xlabel("Timestep (Hour)")
    plt.ylabel("Cumulative Cost (€)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("cumulative_cost_comparison.png")
    plt.close()

    # --- Plot 3: Net Demand Profile Comparison ---
    plt.figure(figsize=(12, 6))
    plt.plot(
        timesteps,
        initial_net_demand,
        label="Initial Net Demand (Baseline)",
        color="gray",
        linestyle=":",
        linewidth=2,
    )
    plt.plot(
        timesteps,
        wh_actual_nd,
        label="Optimized Net Demand (Wholesale)",
        color="orange",
        linestyle="--",
        alpha=0.9,
    )
    plt.plot(
        timesteps,
        fl_actual_nd,
        label="Optimized Net Demand (Local Energy Market)",
        color="blue",
        linestyle="-",
        alpha=0.9,
    )
    plt.title("Net Demand Optimization Comparison")
    plt.xlabel("Timestep (Hour)")
    plt.ylabel("Total Net Demand (MWh)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("net_demand_comparison.png")
    plt.close()

    # --- Print Final Summary (with scaling) ---
    fl_total_profit_scaled = fl_results["total_profit"] * FLEXIBILITY_SCALE_FACTOR

    print("\n" + "=" * 30)
    print("--- 📊 Simulation Comparison Summary ---")
    print(f"Total Steps: {steps_ran}")
    # Note: agents_JSON is not available globally here, but we can assume n=50 for default case
    print("-" * 30)
    print("Wholesale Market:")
    print(f"  Total System Profit/Loss: €{wh_results['total_profit']:.2f}")
    print(f"  Total Buyer Cost:       €{wh_cum_cost[-1]:.2f}")
    print(f"  Average Price:          €{np.mean(wh_prices):.2f}/MWh")
    print("-" * 30)
    print(
        f"Flexibility Market (All Financials Scaled by {FLEXIBILITY_SCALE_FACTOR:.1f}x):"
    )
    print(f"  Total System Profit/Loss: €{fl_total_profit_scaled:.2f}")
    print(f"  Total Buyer Cost:       €{fl_cum_cost[-1]:.2f}")
    print(f"  Average Price:          €{np.mean(fl_prices):.2f}/MWh")
    print("=" * 30)


# --- Argument Parsing (from main_wholesale.py) ---


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a comparison between Wholesale and Flexibility Market episodes."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--generate",
        type=int,
        metavar="N",
        help="Generate N random agents and run the simulation (also saves JSON).",
    )
    group.add_argument(
        "--load-from-file",
        metavar="PATH",
        help="Load agents from a JSON file and run the simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when generating agents (default: 42).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Optional output path to save generated agents JSON (default: agents_<N>.json).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max timesteps for the environment (default: {MAX_STEPS}).",
    )
    return parser.parse_args()


# --- Main Orchestration Function ---


def main():
    args = parse_args()

    # --- 1. Get Agents ---
    if args.generate is not None:
        n = args.generate
        agents_JSON = generate_agents(n=n, seed=args.seed)
        out_path = args.out or f"agents_{n}.json"
        save_agents_to_json(agents_JSON, out_path)
    elif args.load_from_file:
        path = args.load_from_file
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Agents file not found: {path}")
        agents_JSON = load_agents_from_json(path)
    else:
        # Default behavior
        logger.info("Defaulting to generating 50 agents (seed 42).")
        agents_JSON = generate_agents(n=50, seed=42)
        save_agents_to_json(agents_JSON, "agents_50.json")

    # --- 2. Run Simulations ---

    # Run Wholesale
    buy_tariff = 0.15
    sell_tariff = 0.12
    wholesale_results = run_episode_wholesale(
        agent_configs=agents_JSON,
        max_steps=args.steps,
        buy_tariff=buy_tariff * FLEXIBILITY_SCALE_FACTOR,
        sell_tariff=sell_tariff * FLEXIBILITY_SCALE_FACTOR,
    )

    # Run Flexibility
    flexibility_results = run_episode_flexibility(
        agent_configs=agents_JSON,
        max_steps=args.steps,
        buy_tariff=buy_tariff,
        sell_tariff=sell_tariff,
    )

    # --- 3. Plot Comparison ---
    plot_comparison_results(wholesale_results, flexibility_results)


if __name__ == "__main__":
    main()
