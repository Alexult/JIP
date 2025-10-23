import argparse
import random
import json
import os
from energymarket import DoubleAuctionEnv, DoubleAuctionClearingAgent
from loguru import logger

def fixed(job, t): return 1 if job[1] == t else 0
def linear(job, t, c=0.4): return max(1 - abs(job[1] - t) * c, 0)
def free(job, t): return 1

SHAPE_MAP = {'fixed': fixed, 'linear': lambda job, t: linear(job, t, 0.4), 'free': free}
SHAPE_NAMES = list(SHAPE_MAP.keys())
MAX_STEPS = 23
AGENT_CLASSES = ["AggressiveSellerAgent", "AggressiveBuyerAgent", "ProsumerAgent"]
GENERATION_TYPES = ["solar", "wind", "none"]

# Function to generate agents
def generate_agents(n=100, seed=42):
    random.seed(seed)
    agents = []
    for i in range(n):
        agent_class = random.choice(AGENT_CLASSES)

        # Generate load patterns
        loads = []
        for _ in range(random.randint(1, 3)):
            qty = random.randint(5, 100)
            t = random.randint(0, MAX_STEPS)
            shape = random.choice(SHAPE_NAMES)
            loads.append({"qty": qty, "time": t, "shape": shape})

        flexible_load = random.randint(5, 80)
        fixed_load = random.randint(1, 30)
        generation_capacity = random.randint(0, 100)
        generation_type = random.choice(GENERATION_TYPES)
        cost_per_unit = random.randint(1,100)
        cost_per_unit/=100
        margin = random.randint(1,100)
        margin/=100

        agents.append({
            "id": i,
            "agent_class": agent_class,
            "loads": loads,
            "flexible_load": flexible_load,
            "fixed_load": fixed_load,
            "generation_capacity": generation_capacity,
            "generation_type": generation_type,
            "cost_per_unit": cost_per_unit,
            "margin": margin,
        })
    return agents

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
        loads = []
        for load in j["loads"]:
            shape_func = SHAPE_MAP[load["shape"]]
            loads.append((int(load["qty"]), int(load["time"]), shape_func))

        config = {
            "class": j["agent_class"],
            "load": loads,
            "flexible_load": int(j["flexible_load"]),
            "fixed_load": int(j["fixed_load"]),
            "generation_capacity": int(j["generation_capacity"]),
            "cost_per_unit": float(j['cost_per_unit']),
            "margin": float(j['margin']),
        }
        gt = j.get("generation_type", None)
        if gt:
            config["generation_type"] = gt
        configs.append(config)
    return configs

def run_episode(agent_configs, max_steps=MAX_STEPS):
    env = DoubleAuctionEnv(
        agent_configs=agent_configs,
        max_timesteps=max_steps,
        market_clearing_agent=DoubleAuctionClearingAgent(),
    )

    logger.info(f"Starting MARL Episode Demo ({max_steps} steps)")
    observations, info = env.reset()
    all_terminated = {i: False for i in env.agent_ids}
    all_truncated = {i: False for i in env.agent_ids}
    total_reward = 0.0

    while not all(all_terminated.values()) and not all(all_truncated.values()):
        actions = {}
        for agent_id in env.agent_ids:
            obs_i = observations[agent_id]
            actions[agent_id] = env.agents[agent_id].devise_strategy_smarter(
                obs_i, env.action_space
            )

        observations, rewards, all_terminated, all_truncated, info = env.step(actions)
        current_step_reward = sum(rewards.values())
        total_reward += current_step_reward
        env.render()

    print(f"\n--- Episode Finished ---")
    print(f"Total Cumulative Profit (All Agents): {total_reward:.2f}")

    env.plot_results()
    env.plot_consumption_and_costs()
    env.plot_bid_ask_curves(num_plots=5)
    env.plot_price_change_for_single_day(day=0)

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
        agents_JSON = generate_agents(n=100, seed=42)
        save_agents_to_json(agents_JSON, "agents_100.json")

    agent_configs = convert_json_agents_configs(agents_JSON)
    run_episode(agent_configs, max_steps=args.steps)

if __name__ == "__main__":
    main()
