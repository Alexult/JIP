import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prosumers import ProsumerAgent  # only to mirror attributes; we won’t call market logic

# import national price data
PRICE_CSV = os.path.join(os.path.dirname(__file__),
                         "..", "data", "representative_days_wholesale_price_2025.csv")

# Same agents you already simulate in main.py
AGENT_CONFIGS = [
    {"fixed_load": 10, "flexible_load_max": 8},
    {"fixed_load": 18, "flexible_load_max": 7},
    {"fixed_load": 12, "flexible_load_max": 6},
    {"fixed_load": 34, "flexible_load_max": 9},
    {"fixed_load": 19, "flexible_load_max": 5},
]

# Simple price-response: how strongly flexible load reacts to price (0..1)
# 0.0 = no response; 1.0 = full curtailment of flexible load at/above hi threshold
RESPONSE_STRENGTH = 0.8

# We set low/high price thresholds from each day’s distribution
# (this keeps it robust across very different days)
LOW_PCTL  = 0.30
HIGH_PCTL = 0.80


def load_four_days(csv_path: str) -> dict:
    """Return dict {day_str: DataFrame(hour, price)} for each calendar day in file."""
    df = pd.read_csv(csv_path)
    # Expect columns exactly as you described:
    # 'Datetime (Local)' and 'Price(EUR/MWhe)'
    ts_col = "Datetime (Local)"
    p_col  = "Price (EUR/MWhe)"

    # Parse timestamps and split into days
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["day"] = df[ts_col].dt.date
    df["hour"] = df[ts_col].dt.hour

    days = {}
    for d, sub in df.groupby("day"):
        sub = sub.sort_values("hour")[[ "hour", p_col ]].reset_index(drop=True)
        # sanity: must have 24 hourly rows
        if len(sub) != 24:
            print(f"Warning: day {d} has {len(sub)} rows (expected 24). Using what’s available.")
        days[str(d)] = sub
    return days


def build_prosumers_from_configs(configs):
    """Create simple objects carrying fixed and flexible loads (no market behavior)."""
    pros = []
    for i, c in enumerate(configs):
        # We only need fixed/flexible loads; avoid heavy init of ProsumerAgent (no data files).
        class Simple:
            pass
        obj = Simple()
        obj.agent_id = i
        obj.fixed_load = float(c["fixed_load"])
        obj.flexible_load_max = float(c["flexible_load_max"])
        pros.append(obj)
    return pros


def run_baseline_for_day(day_df: pd.DataFrame, prosumers: list) -> dict:
    """Compute total energy cost if everyone buys at DA price,
       with a very simple flexible-load response to high prices."""
    price = day_df["Price (EUR/MWhe)"].to_numpy(dtype=float)  # length ~24

    # thresholds based on this day
    p_lo = np.percentile(price, LOW_PCTL * 100.0)
    p_hi = np.percentile(price, HIGH_PCTL * 100.0)

    total_cost = 0.0
    total_fixed = 0.0
    total_flex_used = 0.0
    total_flex_curtailed = 0.0

    # For each hour, choose how much of each agent’s flexible load is used based on price
    for h in range(len(price)):
        p = price[h]

        # price-to-response factor in [0,1]
        # below p_lo -> 0 (no curtailment); above p_hi -> RESPONSE_STRENGTH (strong curtailment)
        if p <= p_lo:
            r = 0.0
        elif p >= p_hi:
            r = RESPONSE_STRENGTH
        else:
            # linear ramp between low and high
            r = RESPONSE_STRENGTH * (p - p_lo) / max(1e-9, (p_hi - p_lo))

        hour_fixed = 0.0
        hour_flex_used = 0.0
        hour_flex_curtailed = 0.0

        for ag in prosumers:
            # fixed load is always consumed
            hour_fixed += ag.fixed_load
            # flexible: curtail a fraction r of the agent’s flexible max
            curtailed = r * ag.flexible_load_max
            used = ag.flexible_load_max - curtailed
            hour_flex_used += used
            hour_flex_curtailed += curtailed

        hourly_demand = hour_fixed + hour_flex_used  # MWh (units consistent with your loads)
        total_cost += hourly_demand * p
        total_fixed += hour_fixed
        total_flex_used += hour_flex_used
        total_flex_curtailed += hour_flex_curtailed

    avg_price = float(np.mean(price))
    peak_price = float(np.max(price))

    return {
        "avg_price_eur_mwh": avg_price,
        "peak_price_eur_mwh": peak_price,
        "energy_fixed_MWh": total_fixed,
        "energy_flex_used_MWh": total_flex_used,
        "energy_flex_curtailed_MWh": total_flex_curtailed,
        "total_energy_MWh": total_fixed + total_flex_used,
        "total_cost_eur": total_cost,
        "curtailment_share_%": 100.0 * total_flex_curtailed / max(1e-9, (total_fixed + total_flex_used + total_flex_curtailed)),
        "hi_threshold": p_hi,
        "lo_threshold": p_lo,
    }

def plot_day_comparison(days: dict, prosumers: list):
    """Create comprehensive comparison plots for the four representative days."""
    
    # Collect data for all days
    day_names = list(days.keys())
    all_metrics = {}
    hourly_data = {}
    
    for day, df in days.items():
        metrics = run_baseline_for_day(df, prosumers)
        all_metrics[day] = metrics
        
        # Store hourly prices and demands for detailed plots
        price = df["Price (EUR/MWhe)"].to_numpy(dtype=float)
        p_lo = np.percentile(price, LOW_PCTL * 100.0)
        p_hi = np.percentile(price, HIGH_PCTL * 100.0)
        
        # Calculate hourly demand profile
        hourly_demand = []
        hourly_curtailment = []
        
        for h in range(len(price)):
            p = price[h]
            if p <= p_lo:
                r = 0.0
            elif p >= p_hi:
                r = RESPONSE_STRENGTH
            else:
                r = RESPONSE_STRENGTH * (p - p_lo) / max(1e-9, (p_hi - p_lo))
            
            hour_total_demand = 0.0
            hour_curtailed = 0.0
            for ag in prosumers:
                hour_total_demand += ag.fixed_load + ag.flexible_load_max * (1 - r)
                hour_curtailed += ag.flexible_load_max * r
            
            hourly_demand.append(hour_total_demand)
            hourly_curtailment.append(hour_curtailed)
        
        hourly_data[day] = {
            'price': price,
            'demand': hourly_demand,
            'curtailment': hourly_curtailment,
            'p_lo': p_lo,
            'p_hi': p_hi
        }
    
    # Create subplot layout with better spacing
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # Increased figure size
    #fig.suptitle('Baseline Market Analysis: Four Representative Days', 
                 #fontsize=18, fontweight='bold', y=0.98)  # Moved title higher
    
    # Plot 1: Hourly prices for all days
    ax1 = axes[0, 0]
    for day in day_names:
        ax1.plot(range(24), hourly_data[day]['price'], label=f"{day}", linewidth=2, marker='o', markersize=4)
        # Add threshold lines
        ax1.axhline(y=hourly_data[day]['p_lo'], color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=hourly_data[day]['p_hi'], color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_title('Hourly Electricity Prices', fontsize=12, pad=15)  # Added padding
    ax1.set_xlabel('Hour of Day', fontsize=10)
    ax1.set_ylabel('Price (€/MWh)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hourly demand profiles
    ax2 = axes[0, 1]
    for day in day_names:
        ax2.plot(range(24), hourly_data[day]['demand'], label=f"{day}", linewidth=2, marker='s', markersize=4)
    
    ax2.set_title('Hourly Energy Demand (After Curtailment)', fontsize=12, pad=15)
    ax2.set_xlabel('Hour of Day', fontsize=10)
    ax2.set_ylabel('Demand (MWh)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Hourly curtailment
    ax3 = axes[0, 2]
    for day in day_names:
        ax3.plot(range(24), hourly_data[day]['curtailment'], label=f"{day}", linewidth=2, marker='^', markersize=4)
    
    ax3.set_title('Hourly Load Curtailment', fontsize=12, pad=15)
    ax3.set_xlabel('Hour of Day', fontsize=10)
    ax3.set_ylabel('Curtailed Load (MWh)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Daily summary metrics - Energy
    ax4 = axes[1, 0]
    metrics_names = ['energy_fixed_MWh', 'energy_flex_used_MWh', 'energy_flex_curtailed_MWh']
    metrics_labels = ['Fixed Load', 'Flexible Used', 'Curtailed']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    x = np.arange(len(day_names))
    width = 0.25
    
    for i, (metric, label, color) in enumerate(zip(metrics_names, metrics_labels, colors)):
        values = [all_metrics[day][metric] for day in day_names]
        ax4.bar(x + i*width, values, width, label=label, color=color, alpha=0.8)
    
    ax4.set_title('Daily Energy Consumption Breakdown', fontsize=12, pad=15)
    ax4.set_xlabel('Representative Days', fontsize=10)
    ax4.set_ylabel('Energy (MWh)', fontsize=10)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(day_names, rotation=45, fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Cost and price metrics
    ax5 = axes[1, 1]
    total_costs = [all_metrics[day]['total_cost_eur'] for day in day_names]
    avg_prices = [all_metrics[day]['avg_price_eur_mwh'] for day in day_names]
    
    ax5_twin = ax5.twinx()
    
    bars1 = ax5.bar(x - 0.2, total_costs, 0.4, label='Total Cost', color='#2ca02c', alpha=0.8)
    line1 = ax5_twin.plot(x, avg_prices, 'ro-', linewidth=2, markersize=8, label='Avg Price')
    
    ax5.set_title('Daily Costs vs Average Prices', fontsize=12, pad=15)
    ax5.set_xlabel('Representative Days', fontsize=10)
    ax5.set_ylabel('Total Cost (€)', color='#2ca02c', fontsize=10)
    ax5_twin.set_ylabel('Average Price (€/MWh)', color='red', fontsize=10)
    ax5.set_xticks(x)
    ax5.set_xticklabels(day_names, rotation=45, fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Curtailment percentages
    ax6 = axes[1, 2]
    curtailment_pcts = [all_metrics[day]['curtailment_share_%'] for day in day_names]
    peak_prices = [all_metrics[day]['peak_price_eur_mwh'] for day in day_names]
    
    bars2 = ax6.bar(x, curtailment_pcts, color='#ff7f0e', alpha=0.8, label='Curtailment %')
    ax6_twin = ax6.twinx()
    line2 = ax6_twin.plot(x, peak_prices, 'bs-', linewidth=2, markersize=8, label='Peak Price')
    
    ax6.set_title('Curtailment vs Peak Prices', fontsize=12, pad=15)
    ax6.set_xlabel('Representative Days', fontsize=10)
    ax6.set_ylabel('Curtailment (%)', color='#ff7f0e', fontsize=10)
    ax6_twin.set_ylabel('Peak Price (€/MWh)', color='blue', fontsize=10)
    ax6.set_xticks(x)
    ax6.set_xticklabels(day_names, rotation=45, fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Adjust spacing between subplots
    plt.subplots_adjust(
        top=0.93,       # Space for main title
        bottom=0.08,    # Space for x-axis labels
        left=0.06,      # Left margin
        right=0.95,     # Right margin
        hspace=0.35,    # Height spacing between rows
        wspace=0.25     # Width spacing between columns
    )
    
    plt.show()
    
    # Print summary table
    print("\n=== Summary Comparison Table ===")
    print(f"{'Day':<12} {'Avg Price':<10} {'Peak Price':<11} {'Total Cost':<11} {'Curtailment':<12} {'Energy Used':<12}")
    print(f"{'':12} {'(€/MWh)':<10} {'(€/MWh)':<11} {'(€)':<11} {'(%)':<12} {'(MWh)':<12}")
    print("-" * 80)
    
    for day in day_names:
        m = all_metrics[day]
        print(f"{day:<12} {m['avg_price_eur_mwh']:<10.2f} {m['peak_price_eur_mwh']:<11.2f} "
              f"{m['total_cost_eur']:<11.2f} {m['curtailment_share_%']:<12.1f} {m['total_energy_MWh']:<12.2f}")

def main():
    # 1) load the four days
    days = load_four_days(PRICE_CSV)
    if not days:
        raise RuntimeError("No days found in the price CSV.")

    # 2) build prosumers (pull fixed/flexible from the same configs as your ABM)
    prosumers = build_prosumers_from_configs(AGENT_CONFIGS)

    # 3) run four independent scenarios
    print("\n=== Baseline (No LEM, DA price only) — Four-Day Scenarios ===")
    for day, df in days.items():
        metrics = run_baseline_for_day(df, prosumers)
        print(f"\nDay {day}")
        print(f"  Avg price       : {metrics['avg_price_eur_mwh']:.2f} €/MWh")
        print(f"  Peak price      : {metrics['peak_price_eur_mwh']:.2f} €/MWh")
        print(f"  Energy fixed    : {metrics['energy_fixed_MWh']:.2f} MWh")
        print(f"  Energy flex used: {metrics['energy_flex_used_MWh']:.2f} MWh")
        print(f"  Energy curtailed: {metrics['energy_flex_curtailed_MWh']:.2f} MWh "
              f"({metrics['curtailment_share_%']:.1f}%)")
        print(f"  Total energy    : {metrics['total_energy_MWh']:.2f} MWh")
        print(f"  Total cost      : € {metrics['total_cost_eur']:.2f}")
        print(f"  Thresholds      : lo={metrics['lo_threshold']:.2f}, hi={metrics['hi_threshold']:.2f} €/MWh")

    # 4) Create comprehensive comparison plots
    plot_day_comparison(days, prosumers)



if __name__ == "__main__":
    main()



