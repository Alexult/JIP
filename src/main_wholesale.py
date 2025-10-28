import argparse
import random
import json
import os
import perlin_noise
import numpy as np
import matplotlib.pyplot as plt

# --- MODIFIED IMPORT ---
from wholesale_market import WholesaleMarketEnv

# from energymarket import (
#     DoubleAuctionEnv,
#     DoubleAuctionClearingAgent,
#     FlexibilityMarketEnv,
# )
from loguru import logger

MAX_STEPS = 96  # Increased to 4 days for a better plot
GENERATION_TYPES = ["solar", "wind", "none"]


# Function to generate agents
def generate_agents(n=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    agents = []
    for i in range(n):
        load = generate_load()
        generation_capacity = random.randint(0, 100)
        generation_type = random.choice(GENERATION_TYPES)
        agents.append(
            {
                "id": i,
                "load": load,
                "generation_capacity": generation_capacity,
                "generation_type": generation_type,
                # This marginal_price is used by ProsumerAgent for selling
                "marginal_price": random.uniform(0.05, 0.15),  # e.g., 50-150 EUR/MWh
            }
        )
    return agents


def generate_load():
    noise = perlin_noise.PerlinNoise(octaves=2, seed=random.randint(0, 1000))
    # Generate load for more than just 24 steps
    scale = random.uniform(20, 80)
    y = [
        scale * (noise(i * 0.05) * 0.5 + 0.6) + random.uniform(-5, 5)
        for i in range(MAX_STEPS * 2)
    ]
    return [max(0, val) for val in y]  # Ensure no negative load


# Save agents to JSON
def save_agents_to_json(agents, filename="agents_100.json"):
    with open(filename, "w") as f:
        json.dump(agents, f, indent=2)
    print(f"Stored {len(agents)} agents in {filename}")
    return filename


# Load agents from JSON
def load_agents_from_json(filename="agents_100.json"):
    with open(filename, "r") as f:
        return json.load(f)


# Convert JSON agents -> AGENT_CONFIGS
def convert_json_agents_configs(json_agents):
    configs = []
    for j in json_agents:
        config = {
            "load": j["load"],
            "generation_capacity": int(j["generation_capacity"]),
            "marginal_price": float(j["marginal_price"]),
            "generation_type": j.get("generation_type", "none"),
            # Add other fields required by ProsumerAgent if any
            # These are for the *other* env, but we'll add them for compatibility
            "flexible_load": j.get("flexible_load", []),
            "fixed_load": j.get("fixed_load", []),
        }
        configs.append(config)
    return configs


def run_episode(agent_configs, max_steps=MAX_STEPS):
    # --- MODIFIED ENV CREATION ---
    env = WholesaleMarketEnv(
        agent_configs=agent_configs,
        wholesale_csv_path="./data/representative_wholesale_price_2025.csv",
        max_timesteps=max_steps,
    )
    # --- END MODIFICATION ---

    logger.info(f"Starting Wholesale Market Episode Demo ({max_steps} steps)")
    observations, info = env.reset()
    all_terminated = {i: False for i in env.agent_ids}
    all_truncated = {i: False for i in env.agent_ids}

    # Use a dict to track total rewards per agent
    total_agent_rewards = {i: 0.0 for i in env.agent_ids}

    t = 0
    while not all(all_terminated.values()) and not all(all_truncated.values()):
        # The 'time_step' passed to devise_strategy tells the agent
        # which index of its load profile to use as t=0
        time_step = t

        actions = {}
        for agent_id in env.agent_ids:
            if agent_id in observations:
                obs_i = observations[agent_id]
                # Agent plans its schedule based on forecast
                # The returned bids/offers are ignored by WholesaleMarketEnv
                # but the agent's internal state (self.schedule) is updated.
                actions[agent_id] = env.agents[agent_id].devise_strategy(
                    obs_i, env.action_space, time_step
                )

        # The env's step function will read the agent's updated state
        observations, rewards, all_terminated, all_truncated, info = env.step(actions)

        for agent_id, reward in rewards.items():
            total_agent_rewards[agent_id] += reward

        env.render()
        t += 1

        # Check for 'all' manually
        if all(all_terminated.values()) or all(all_truncated.values()):
            break

    total_system_profit = sum(total_agent_rewards.values())
    print(f"\n--- Episode Finished ---")
    print(f"Total System Profit/Loss (All Agents): €{total_system_profit:.2f}")

    # Plot final results
    env.plot_results()
    env.plot_price_change_for_single_day(day=0)
    if max_steps > 24:
        env.plot_price_change_for_single_day(day=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Wholesale Market episode with generated or pre-defined agents."
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


def main():
    args = parse_args()

    # Decide source of agents
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
        # Default behavior: generate 50 agents
        logger.info(
            "No --generate or --load-from-file flag set. Defaulting to generating 50 agents."
        )
        agents_JSON = generate_agents(n=50, seed=42)
        save_agents_to_json(agents_JSON, "agents_50.json")

    agent_configs = convert_json_agents_configs(agents_JSON)
    run_episode(agent_configs, max_steps=args.steps)


if __name__ == "__main__":
    main()
