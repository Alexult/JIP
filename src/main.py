import argparse
import random
import json
import os
import perlin_noise
import numpy as np
import matplotlib.pyplot as plt

from energymarket import (
    DoubleAuctionEnv,
    DoubleAuctionClearingAgent,
    FlexibilityMarketEnv,
)
from loguru import logger

MAX_STEPS = 24
GENERATION_TYPES = ["solar", "wind", "none"]

# Function to generate agents
def generate_agents(n=100, seed=42):
    random.seed(seed)
    agents = []
    for i in range(n):
        load = generate_load()
        generation_capacity = random.randint(0, 100)
        generation_type = random.choice(GENERATION_TYPES)
        agents.append({
            "id": i,
            "load": load,
            "generation_capacity": generation_capacity,
            "generation_type": generation_type,
            "marginal_price": random.randint(550, 600) / 1000,
        })
    return agents


def generate_load():
    noise = perlin_noise.PerlinNoise(octaves=1)
    scale = random.randrange(10, 40)
    y = [scale * (noise(i * 0.1) + 1) for i in range(MAX_STEPS)]
    return y


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
        }
        gt = j.get("generation_type", None)
        if gt:
            config["generation_type"] = gt
        configs.append(config)
    return configs


def run_episode(agent_configs, max_steps=MAX_STEPS):
    env = FlexibilityMarketEnv(
        agent_configs=agent_configs,
        market_clearing_agent=DoubleAuctionClearingAgent(),
        discount=(1, 1000),
        max_timesteps=max_steps,
        buy_tariff=0.1,
        sell_tariff=0.1
    )

    logger.info(f"Starting MARL Episode Demo ({max_steps} steps)")
    observations, info = env.reset()
    all_terminated = {i: False for i in env.agent_ids}
    all_truncated = {i: False for i in env.agent_ids}
    total_reward = 0.0
    t = 1

    time_step = 0
    while not all(all_terminated.values()) and not all(all_truncated.values()):

        actions = {}
        for agent_id in env.agent_ids:
            obs_i = observations[agent_id]
            actions[agent_id] = env.agents[agent_id].devise_strategy(obs_i, env.action_space, time_step)
        time_step += 1

        observations, rewards, all_terminated, all_truncated, info = env.step(actions)
        current_step_reward = sum(rewards.values())
        total_reward += current_step_reward
        env.render()
        t += 1

    print(f"\n--- Episode Finished ---")
    print(f"Total Cumulative Profit (All Agents): {total_reward:.2f}")

    env.plot_results()
    # env.plot_consumption_and_costs()
    # env.plot_bid_ask_curves(num_plots=5)
    # env.plot_price_change_for_single_day(day=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Double Auction MARL episode with generated or pre-defined agents."
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
        # Default behavior: generate 100 with seed 42 (keeps old script's spirit)
        agents_JSON = generate_agents(n=25, seed=41)
        save_agents_to_json(agents_JSON, "agents_100.json")

    agent_configs = convert_json_agents_configs(agents_JSON)
    run_episode(agent_configs, max_steps=args.steps)


if __name__ == "__main__":
    main()
