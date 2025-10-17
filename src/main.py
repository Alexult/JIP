from energymarket import DoubleAuctionEnv, DoubleAuctionClearingAgent, WholesaleMarketEnv
from loguru import logger

if __name__ == "__main__":
    # --- ENVIRONMENT SELECTION ---
    # Set to True for WholesaleMarketEnv, False for DoubleAuctionEnv
    USE_WHOLESALE_MARKET = True  

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

    # --- CREATE ENVIRONMENT BASED ON SELECTION ---
    if USE_WHOLESALE_MARKET:
        env = WholesaleMarketEnv(
            agent_configs=AGENT_CONFIGS,
            wholesale_csv_path="../data/representative_wholesale_price_2025.csv",
            max_timesteps=MAX_STEPS,
        )
        env_name = "Wholesale Market"
    else:
        env = DoubleAuctionEnv(
            agent_configs=AGENT_CONFIGS,
            max_timesteps=MAX_STEPS,
            market_clearing_agent=DoubleAuctionClearingAgent(),
        )
        env_name = "Double Auction"

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
    print(f"\n--- Simulation Finished ---")
    print(f"Total Cumulative Profit (All Agents): {total_reward:.2f}")

    env.plot_results()

    # Environment-specific additional plots
    if USE_WHOLESALE_MARKET:
        env.plot_trading_pattern()
    else:
        env.plot_bid_ask_curves(num_plots=5)    

    env.plot_price_change_for_single_day(day=0)
