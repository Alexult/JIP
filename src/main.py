from energymarket import DoubleAuctionEnv, DoubleAuctionClearingAgent
from loguru import logger

if __name__ == "__main__":
    AGENT_CONFIGS = [
        {
            "class": "AggressiveSellerAgent",
            "fixed_load": 10,
            "flexible_load_max": 8,
            "generation_capacity": 40,
        },
        {
            "class": "AggressiveBuyerAgent",
            "fixed_load": 18,
            "flexible_load_max": 7,
            "generation_capacity": 5,
        },
        {
            "class": "ProsumerAgent",
            "fixed_load": 12,
            "flexible_load_max": 6,
            "generation_capacity": 30,
            "generation_type": "wind",
        },
        {
            "class": "ProsumerAgent",
            "fixed_load": 34,
            "flexible_load_max": 9,
            "generation_capacity": 18,
            "generation_type": "wind",
        },
        {
            "class": "ProsumerAgent",
            "fixed_load": 19,
            "flexible_load_max": 5,
            "generation_capacity": 11,
        },
    ]
    MAX_STEPS = 24 * 10

    env = DoubleAuctionEnv(
        agent_configs=AGENT_CONFIGS,
        max_timesteps=MAX_STEPS,
        market_clearing_agent=DoubleAuctionClearingAgent(),
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
            actions[agent_id] = env.agents[agent_id].devise_strategy(
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
    env.plot_bid_ask_curves(num_plots=5)
    env.plot_price_change_for_single_day(day=0)  # Plot prices for the first day
