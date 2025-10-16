import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prosumers import ProsumerAgent  # Import to create actual prosumer instances

# import national price data
PRICE_CSV = os.path.join(os.path.dirname(__file__),
                         "..", "data", "representative_days_wholesale_price_2025.csv")

# Agent configurations matching your main.py setup
AGENT_CONFIGS = [
    {"fixed_load": 10, "flexible_load_max": 8, "generation_capacity": 40, "generation_type": "solar"},
    {"fixed_load": 18, "flexible_load_max": 7, "generation_capacity": 5,  "generation_type": "solar"},
    {"fixed_load": 12, "flexible_load_max": 6, "generation_capacity": 0,  "generation_type": "none"},  # Consumer
    {"fixed_load": 34, "flexible_load_max": 9, "generation_capacity": 18, "generation_type": "wind"},
    {"fixed_load": 19, "flexible_load_max": 5, "generation_capacity": 0,  "generation_type": "none"},  # Consumer
]

# Different response behaviors for consumers vs prosumers
CONSUMER_RESPONSE_STRENGTH = 0.6  # Pure consumers are less aggressive at consumption redcution
PROSUMER_RESPONSE_STRENGTH = 0.8  # Prosumers with generation can curtail more aggressively

# We set low/high price thresholds from each day's distribution
LOW_PCTL  = 0.30
HIGH_PCTL = 0.80


def load_four_days(csv_path: str) -> dict:
    """Return dict {day_str: DataFrame(hour, price)} for each calendar day in file."""
    df = pd.read_csv(csv_path)
    ts_col = "Datetime (Local)"
    p_col  = "Price (EUR/MWhe)"

    # Parse timestamps and split into days
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["day"] = df[ts_col].dt.date
    df["hour"] = df[ts_col].dt.hour

    days = {}
    for d, sub in df.groupby("day"):
        sub = sub.sort_values("hour")[[ "hour", p_col ]].reset_index(drop=True)
        if len(sub) != 24:
            print(f"Warning: day {d} has {len(sub)} rows (expected 24). Using what's available.")
        days[str(d)] = sub
    return days


def build_prosumers_from_configs(configs):
    """Create actual ProsumerAgent instances to get realistic demand patterns."""
    agents = []
    for i, c in enumerate(configs):
        # Create actual ProsumerAgent instance
        agent = ProsumerAgent(
            agent_id=i,
            fixed_load=float(c["fixed_load"]),
            flexible_load_max=float(c["flexible_load_max"]),
            generation_capacity=float(c.get("generation_capacity", 0.0)),
            generation_type=c.get("generation_type", "solar")
        )
        
        # Add agent type based on generation capacity
        if agent.generation_capacity > 0:
            agent.agent_type = "prosumer"
        else:
            agent.agent_type = "consumer"
        
        agents.append(agent)
    return agents


def get_hourly_demand_data(agents: list, timestep_offset: int = 0) -> dict:
    """
    Get realistic hourly demand data for all agents using their calculate_net_demand method.
    
    Args:
        agents: List of ProsumerAgent instances
        timestep_offset: Starting timestep (for different days)
    
    Returns:
        dict: Hourly data for each agent including flexible load, generation, etc.
    """
    hourly_data = {
        'agents': [],
        'total_flexible_used': [],
        'total_generation': [],
        'total_fixed': [],
        'net_demands': []
    }
    
    # Get data for 24 hours
    for hour in range(24):
        timestep = timestep_offset + hour
        hour_flexible_used = 0.0
        hour_generation = 0.0
        hour_fixed = 0.0
        hour_net_demands = []
        
        agent_hour_data = []
        
        for agent in agents:
            agent.calculate_net_demand(timestep)  # ← This calculates realistic flexible load!
            
            # Extract components
            if agent.generation_capacity > 0:
                hour_of_day = timestep % 24
                if agent.generation_type == "solar":
                    effective_generation = agent._calc_solar_generation(hour_of_day)
                else:  # wind
                    effective_generation = agent._calc_wind_generation(hour_of_day)
            else:
                effective_generation = 0.0
            
            # Now reverse calculate with UPDATED net_demand
            total_load = agent.net_demand + effective_generation
            current_flexible_load = total_load - agent.fixed_load
            
            # Store agent data for this hour
            agent_data = {
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type,
                'fixed_load': agent.fixed_load,
                'flexible_load_used': current_flexible_load,
                'generation': effective_generation,
                'net_demand': agent.net_demand
            }
            agent_hour_data.append(agent_data)
            
            # Accumulate totals
            hour_flexible_used += current_flexible_load
            hour_generation += effective_generation
            hour_fixed += agent.fixed_load
            hour_net_demands.append(agent.net_demand)
        
        hourly_data['agents'].append(agent_hour_data)
        hourly_data['total_flexible_used'].append(hour_flexible_used)
        hourly_data['total_generation'].append(hour_generation)
        hourly_data['total_fixed'].append(hour_fixed)
        hourly_data['net_demands'].append(hour_net_demands)
    
    return hourly_data


def run_baseline_for_day(day_df: pd.DataFrame, agents: list, day_index: int = 0) -> dict:
    """Compute total energy cost using realistic demand patterns with separate logic for buyers vs sellers."""
    price = day_df["Price (EUR/MWhe)"].to_numpy(dtype=float)

    # Calculate thresholds based on this day
    p_lo = np.percentile(price, LOW_PCTL * 100.0)
    p_hi = np.percentile(price, HIGH_PCTL * 100.0)

    # Get realistic hourly demand data using prosumer logic
    timestep_offset = day_index * 24  # Different starting point for each day
    hourly_demand_data = get_hourly_demand_data(agents, timestep_offset)

    total_cost = 0.0
    total_fixed = 0.0
    total_flex_used = 0.0
    total_flex_curtailed = 0.0
    total_generation = 0.0
    total_generation_curtailed = 0.0
    
    # Enhanced tracking for consumers vs prosumers
    consumer_metrics = {"cost": 0.0, "curtailed": 0.0, "used": 0.0, "count": 0}
    prosumer_metrics = {"cost": 0.0, "curtailed": 0.0, "used": 0.0, "generation": 0.0, 
                       "generation_curtailed": 0.0, "count": 0}

    # Count agents by type
    for ag in agents:
        if ag.agent_type == "consumer":
            consumer_metrics["count"] += 1
        else:
            prosumer_metrics["count"] += 1

    # For each hour, apply different logic based on net buyer/seller status
    for h in range(len(price)):
        if h >= len(hourly_demand_data['agents']):
            break
            
        p = price[h]
        hour_agent_data = hourly_demand_data['agents'][h]
        
        hour_fixed = 0.0
        hour_flex_used = 0.0
        hour_flex_curtailed = 0.0
        hour_generation = 0.0
        hour_generation_curtailed = 0.0

        for agent_data in hour_agent_data:
            agent = agents[agent_data['agent_id']]
            
            # Use the realistic flexible load from prosumer calculation
            realistic_flexible_load = agent_data['flexible_load_used']
            baseline_generation = agent_data['generation']
            
            # Determine initial net demand (before any response)
            initial_net_demand = (agent_data['fixed_load'] + realistic_flexible_load - 
                                baseline_generation)
            
            # Initialize response variables
            consumption_curtailed = 0.0
            generation_curtailed = 0.0
            
            # SEPARATE LOGIC FOR NET BUYERS VS NET SELLERS
            if initial_net_demand > 0:
                # ===== NET BUYER LOGIC =====
                # Agent needs to buy electricity from grid
                
                if agent.agent_type == "consumer":
                    response_strength = CONSUMER_RESPONSE_STRENGTH
                else:  # prosumer
                    response_strength = PROSUMER_RESPONSE_STRENGTH
                
                # High prices → reduce consumption
                if p <= p_lo:
                    r_consumption = 0.0
                elif p >= p_hi:
                    r_consumption = response_strength
                else:
                    r_consumption = response_strength * (p - p_lo) / max(1e-9, (p_hi - p_lo))
                
                consumption_curtailed = r_consumption * realistic_flexible_load
                
                # Prosumers might also increase generation slightly at high prices
                if agent.agent_type == "prosumer" and baseline_generation > 0 and p >= p_hi:
                    # Don't curtail generation when prices are high (actually produce more if possible)
                    generation_curtailed = 0.0
                
            else:
                # ===== NET SELLER LOGIC =====
                # Agent has surplus to sell to grid
                
                if agent.agent_type == "prosumer":
                    # High prices → want to sell more (reduce own consumption, maintain generation)
                    if p >= p_hi:
                        # Very high prices: reduce consumption more aggressively to sell more
                        r_consumption = PROSUMER_RESPONSE_STRENGTH * 1.2  # Even more aggressive
                        consumption_curtailed = min(r_consumption * realistic_flexible_load, 
                                                  realistic_flexible_load * 0.9)  # Max 90% reduction
                        generation_curtailed = 0.0  # Don't curtail generation at high prices
                        
                    elif p <= p_lo:
                        # Low prices → less incentive to sell
                        # Maybe curtail some generation or consume more
                        r_consumption = -0.1  # Actually increase consumption by 10%
                        consumption_curtailed = r_consumption * realistic_flexible_load  # Negative = increase
                        
                        # Curtail some generation at very low prices
                        if baseline_generation > 0:
                            generation_curtailed = 0.15 * baseline_generation  # Curtail 15% of generation
                    else:
                        # Medium prices → no special response
                        consumption_curtailed = 0.0
                        generation_curtailed = 0.0
                else:
                    # This shouldn't happen (consumers can't be net sellers with zero generation)
                    consumption_curtailed = 0.0
                    generation_curtailed = 0.0
            
            # Apply the responses
            if consumption_curtailed >= 0:
                used_flexible = realistic_flexible_load - consumption_curtailed
                curtailed_flexible = consumption_curtailed
            else:
                # Negative curtailment = increased consumption
                used_flexible = realistic_flexible_load + abs(consumption_curtailed)
                curtailed_flexible = consumption_curtailed  # Keep negative for tracking
            
            actual_generation = baseline_generation - generation_curtailed
            
            # Fixed load and totals
            hour_fixed += agent_data['fixed_load']
            hour_flex_used += used_flexible
            hour_flex_curtailed += curtailed_flexible
            hour_generation += actual_generation
            hour_generation_curtailed += generation_curtailed
            
            # Calculate final net demand and cost for this agent
            agent_net_demand = agent_data['fixed_load'] + used_flexible - actual_generation
            
            # Cost calculation
            if agent_net_demand > 0:
                # Net buyer pays for electricity
                agent_cost = agent_net_demand * p
            else:
                # Net seller receives payment (negative cost)
                agent_cost = agent_net_demand * p  # This will be negative (revenue)
            
            # Track metrics by agent type
            if agent.agent_type == "consumer":
                consumer_metrics["cost"] += agent_cost
                consumer_metrics["curtailed"] += max(0, curtailed_flexible)  # Only positive curtailment
                consumer_metrics["used"] += used_flexible
            else:
                prosumer_metrics["cost"] += agent_cost
                prosumer_metrics["curtailed"] += max(0, curtailed_flexible)  # Only positive curtailment
                prosumer_metrics["used"] += used_flexible
                prosumer_metrics["generation"] += actual_generation
                prosumer_metrics["generation_curtailed"] += generation_curtailed

        total_fixed += hour_fixed
        total_flex_used += hour_flex_used
        total_flex_curtailed += hour_flex_curtailed
        total_generation += hour_generation
        total_generation_curtailed += hour_generation_curtailed
    # Calculate total system cost (sum of all agent costs, including negative for sellers)
    total_cost = consumer_metrics["cost"] + prosumer_metrics["cost"]

    avg_price = float(np.mean(price))
    peak_price = float(np.max(price))

    return {
        "avg_price_eur_mwh": avg_price,
        "peak_price_eur_mwh": peak_price,
        "energy_fixed_MWh": total_fixed,
        "energy_flex_used_MWh": total_flex_used,
        "energy_flex_curtailed_MWh": total_flex_curtailed,
        "energy_generation_MWh": total_generation,
        "energy_generation_curtailed_MWh": total_generation_curtailed,
        "total_energy_MWh": total_fixed + total_flex_used,
        "net_energy_MWh": total_fixed + total_flex_used - total_generation,
        "total_cost_eur": total_cost,
        "curtailment_share_%": 100.0 * max(0, total_flex_curtailed) / max(1e-9, (total_fixed + total_flex_used + max(0, total_flex_curtailed))),
        "hi_threshold": p_hi,
        "lo_threshold": p_lo,
        # Enhanced metrics
        "consumer_cost_eur": consumer_metrics["cost"],
        "consumer_curtailment_MWh": consumer_metrics["curtailed"],
        "consumer_count": consumer_metrics["count"],
        "prosumer_cost_eur": prosumer_metrics["cost"],
        "prosumer_curtailment_MWh": prosumer_metrics["curtailed"],
        "prosumer_generation_MWh": prosumer_metrics["generation"],
        "prosumer_generation_curtailed_MWh": prosumer_metrics["generation_curtailed"],
        "prosumer_count": prosumer_metrics["count"],
        "hourly_demand_data": hourly_demand_data,
    }
def plot_day_comparison(days: dict, agents: list):
    """Create comprehensive comparison plots using realistic demand patterns."""
    
    # Collect data for all days
    day_names = list(days.keys())
    all_metrics = {}
    
    for day_index, (day, df) in enumerate(days.items()):
        metrics = run_baseline_for_day(df, agents, day_index)
        all_metrics[day] = metrics
    
    # Create enhanced plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Plot 1: Hourly prices
    ax1 = axes[0, 0]
    for day in day_names:
        df = days[day]
        price = df["Price (EUR/MWhe)"].to_numpy(dtype=float)
        ax1.plot(range(24), price, label=f"{day}", linewidth=2, marker='o', markersize=4)
    
    ax1.set_title('Hourly Electricity Prices', fontsize=12, pad=15)
    ax1.set_xlabel('Hour of Day', fontsize=10)
    ax1.set_ylabel('Price (€/MWh)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Realistic flexible load patterns
    ax2 = axes[0, 1]
    for day in day_names:
        hourly_data = all_metrics[day]['hourly_demand_data']
        flexible_loads = []
        for hour_data in hourly_data['agents']:
            hour_total_flexible = sum([agent['flexible_load_used'] for agent in hour_data])
            flexible_loads.append(hour_total_flexible)
        ax2.plot(range(24), flexible_loads, label=f"{day} Flexible Load", linewidth=2, marker='s')
    
    ax2.set_title('Realistic Hourly Flexible Load Patterns', fontsize=12, pad=15)
    ax2.set_xlabel('Hour of Day', fontsize=10)
    ax2.set_ylabel('Flexible Load (MWh)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Generation vs Demand
    ax3 = axes[0, 2]
    for day in day_names:
        hourly_data = all_metrics[day]['hourly_demand_data']
        total_demands = []
        total_generations = []
        for hour_data in hourly_data['agents']:
            hour_demand = sum([agent['fixed_load'] + agent['flexible_load_used'] for agent in hour_data])
            hour_gen = sum([agent['generation'] for agent in hour_data])
            total_demands.append(hour_demand)
            total_generations.append(hour_gen)
        
        ax3.plot(range(24), total_demands, label=f"{day} Demand", linewidth=2, linestyle='-')
        ax3.plot(range(24), total_generations, label=f"{day} Generation", linewidth=2, linestyle='--')
    
    ax3.set_title('Hourly Demand vs Generation', fontsize=12, pad=15)
    ax3.set_xlabel('Hour of Day', fontsize=10)
    ax3.set_ylabel('Energy (MWh)', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost breakdown by agent type
    ax4 = axes[1, 0]
    x = np.arange(len(day_names))
    width = 0.35
    
    consumer_costs = [all_metrics[day]['consumer_cost_eur'] for day in day_names]
    prosumer_costs = [all_metrics[day]['prosumer_cost_eur'] for day in day_names]
    
    ax4.bar(x - width/2, consumer_costs, width, label='Consumers', color='#ff7f0e', alpha=0.8)
    ax4.bar(x + width/2, prosumer_costs, width, label='Prosumers', color='#2ca02c', alpha=0.8)
    
    ax4.set_title('Daily Costs by Agent Type', fontsize=12, pad=15)
    ax4.set_xlabel('Representative Days', fontsize=10)
    ax4.set_ylabel('Cost (€)', fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(day_names, rotation=45, fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Energy balance
    ax5 = axes[1, 1]
    total_demand = [all_metrics[day]['total_energy_MWh'] for day in day_names]
    total_generation = [all_metrics[day]['energy_generation_MWh'] for day in day_names]
    net_energy = [all_metrics[day]['net_energy_MWh'] for day in day_names]
    
    ax5.bar(x - 0.25, total_demand, 0.25, label='Total Demand', color='#d62728', alpha=0.8)
    ax5.bar(x, total_generation, 0.25, label='Generation', color='#2ca02c', alpha=0.8)
    ax5.bar(x + 0.25, net_energy, 0.25, label='Net Demand', color='#1f77b4', alpha=0.8)
    
    ax5.set_title('Daily Energy Balance (Realistic Patterns)', fontsize=12, pad=15)
    ax5.set_xlabel('Representative Days', fontsize=10)
    ax5.set_ylabel('Energy (MWh)', fontsize=10)
    ax5.set_xticks(x)
    ax5.set_xticklabels(day_names, rotation=45, fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Response comparison
    ax6 = axes[1, 2]
    consumer_curtailment = [all_metrics[day]['consumer_curtailment_MWh'] for day in day_names]
    prosumer_curtailment = [all_metrics[day]['prosumer_curtailment_MWh'] for day in day_names]
    
    ax6.bar(x - width/2, consumer_curtailment, width, label='Consumer Load Reduction', color='#ff7f0e', alpha=0.8)
    ax6.bar(x + width/2, prosumer_curtailment, width, label='Prosumer Load Reduction', color='#2ca02c', alpha=0.8)
    
    ax6.set_title('Daily Load Reduction by Agent Type', fontsize=12, pad=15)
    ax6.set_xlabel('Representative Days', fontsize=10)
    ax6.set_ylabel('Reduced Load (MWh)', fontsize=10)
    ax6.set_xticks(x)
    ax6.set_xticklabels(day_names, rotation=45, fontsize=9)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.35, wspace=0.25)
    plt.show()
    
    # Enhanced summary table
    print("\n=== Enhanced Summary: Realistic Demand Patterns ===")
    print(f"{'Day':<12} {'Total Cost':<11} {'Consumer Cost':<13} {'Prosumer Cost':<13} {'Generation':<11} {'Net Demand':<11}")
    print(f"{'':12} {'(€)':<11} {'(€)':<13} {'(€)':<13} {'(MWh)':<11} {'(MWh)':<11}")
    print("-" * 90)
    
    for day in day_names:
        m = all_metrics[day]
        print(f"{day:<12} {m['total_cost_eur']:<11.2f} {m['consumer_cost_eur']:<13.2f} "
              f"{m['prosumer_cost_eur']:<13.2f} {m['energy_generation_MWh']:<11.2f} {m['net_energy_MWh']:<11.2f}")


def main():
    # 1) load the four days
    days = load_four_days(PRICE_CSV)
    if not days:
        raise RuntimeError("No days found in the price CSV.")

    # 2) build actual prosumer agents (with realistic demand patterns)
    agents = build_prosumers_from_configs(AGENT_CONFIGS)

    # 3) Print agent summary
    print("\n=== Agent Configuration Summary ===")
    for ag in agents:
        print(f"Agent {ag.agent_id}: {ag.agent_type}, Fixed: {ag.fixed_load} MWh, "
              f"Flexible Max: {ag.flexible_load_max} MWh, Generation: {ag.generation_capacity} MWh ({ag.generation_type})")

    # 4) run four independent scenarios with realistic demand
    for day_index, (day, df) in enumerate(days.items()):
        metrics = run_baseline_for_day(df, agents, day_index)
        print(f"\nDay {day}")
        print(f"  Total cost         : € {metrics['total_cost_eur']:.2f}")
        print(f"  Consumer cost      : € {metrics['consumer_cost_eur']:.2f} ({metrics['consumer_count']} agents)")
        print(f"  Prosumer cost      : € {metrics['prosumer_cost_eur']:.2f} ({metrics['prosumer_count']} agents)")
        print(f"  Total generation   : {metrics['energy_generation_MWh']:.2f} MWh")
        print(f"  Net demand         : {metrics['net_energy_MWh']:.2f} MWh")
        print(f"  Curtailment        : {metrics['curtailment_share_%']:.1f}%")

    # 5) Create comprehensive comparison plots
    plot_day_comparison(days, agents)


if __name__ == "__main__":
    main()