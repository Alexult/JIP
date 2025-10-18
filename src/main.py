import random
import json
from energymarket import DoubleAuctionEnv, DoubleAuctionClearingAgent
from loguru import logger

if __name__ == "__main__":
    fixed = lambda job, t: 1 if job[1] == t else 0

    c = 0.4
    linear = lambda job, t: max(1 - abs(job[1] - t) * c, 0)

    free = lambda job, t: 1

    AGENT_CONFIGS = [
        {
            "class": "AggressiveSellerAgent",
            "load": [(10, 2, free)],
            "flexible_load": 1,
            "fixed_load": 2,
            "generation_capacity": 40,
            "generation_type": "solar",
        },
        {
            "class": "AggressiveBuyerAgent",
            "load": [(40, 1, linear), (30, 8, fixed)],
            "flexible_load": 50,
            "fixed_load": 10,
            "generation_capacity": 5,
        },
        {
            "class": "ProsumerAgent",
            "load": [(40, 17, free), (67, 8, fixed)],
            "flexible_load": 6,
            "fixed_load": 5,
            "generation_capacity": 50,
            "generation_type": "wind",
        },
        {
            "class": "ProsumerAgent",
            "load": [(20, 12, linear), (30, 14, fixed)],
            "flexible_load": 5,
            "fixed_load": 25,
            "generation_capacity": 60,
            "generation_type": "wind",
        },
        {
            "class": "ProsumerAgent",
            "load": [(34, 4, linear), (24, 16, fixed)],
            "flexible_load": 5,
            "fixed_load": 29,
            "generation_capacity": 11,
        },
    ]
    MAX_STEPS = 23
    AGENT_CLASSES = ["AggressiveSellerAgent", "AggressiveBuyerAgent", "ProsumerAgent"]
    GENERATION_TYPES = ["solar", "wind", "none"]

    # Function to generate 100 agents
    def generate_agents(n = 100, seed = 42):
        random.seed(seed)
        agents = []
        for i in range(n):
            agent_class = random.choice(AGENT_CLASSES)

            # Generate load patterns
            loads = []
            for _ in range(random.randint(1,3)):
                qty = random.randint(5,100)
                t = random.randint(0,MAX_STEPS)
                shape = random.choice(SHAPE_NAMES)
                loads.append({"qty": qty, "time": t, "shape": shape})

            flexible_load = random.randint(5,80)
            fixed_load = random.randint(1,30)
            generation_capacity = random.randint(0,100)
            generation_type = random.choice(GENERATION_TYPES)

            agents.append({
                "id": i,
                "agent_class": agent_class,
                "loads": loads,
                "flexible_load": flexible_load,
                "fixed_load": fixed_load,
                "generation_capacity": generation_capacity,
                "generation_type": generation_type,
            })
        return agents

    # Function to save agents to JSON
    def save_agents_to_json(agents, filename="agents_100.json"):
        with open(filename, "w") as f:
            json.dump(agents, f, indent=2)
        print(f"Stored {len(agents)} agents in {filename}")

    # Function to load agents from JSON
    def load_agents_from_json(filename="agents_100.json"):
        with open(filename, "r") as f:
            return json.load(f)

    # Function to convert JSON agents -> AGENT_CONFIGS
    def convert_json_agents_configs(json_agents):
        configs = []
        for j in json_agents:
            loads = []
            for load in j["loads"]:
                # Convert shape to callable
                shape_func = SHAPE_MAP[load["shape"]]
                loads.append((int(load["qty"]), int(load["time"]), shape_func))

            config = {
                "class": j["agent_class"],
                "load": loads,
                "flexible_load": int(j["flexible_load"]),
                "fixed_load": int(j["fixed_load"]),
                "generation_capacity": int(j["generation_capacity"]),
            }
            # Check if the generation type is "none"
            gt = j.get("generation_type",None) # pulls value from JSON agent j under "generation_type"
                                                 # and if it doesn't exist it uses the value "none"
            if gt:
                config["generation_type"] = gt

            configs.append(config)
        return configs

    agents_JSON = generate_agents(n = 100, seed = 42)
    save_agents_to_json(agents_JSON, "agents_100.json")

    AGENT_CONFIGS = convert_json_agents_configs(agents_JSON)


    env = DoubleAuctionEnv(
        agent_configs=AGENT_CONFIGS,
        max_timesteps=MAX_STEPS,
        market_clearing_agent=DoubleAuctionClearingAgent(),
        buy_tariff=0.23,
        sell_tariff=0.10,
    )

    logger.info(f"Starting MARL Episode Demo ({MAX_STEPS} steps)")

    # Initial observation received upon reset
    observations, info = env.reset()
    all_terminated = {i: False for i in env.agent_ids}
    all_truncated = {i: False for i in env.agent_ids}
    total_reward = 0.0

    while not all(all_terminated.values()) and not all(all_truncated.values()):
        actions = {}
        # Iterate over all agents to generate an action by calling the agent's internal method
        for agent_id in env.agent_ids:
            # Retrieve the agent's current 5-feature observation
            obs_i = observations[agent_id]

            # --- CALL AGENT'S DEVISE_STRATEGY METHOD (Uses ProsumerAgent or subclass method) ---
            actions[agent_id] = env.agents[agent_id].devise_strategy_smarter(
                obs_i, env.action_space
            )

        # Step the environment with the state-aware actions
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
