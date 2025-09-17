import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Household Agent
# ------------------------
class Household:
    def __init__(self, id, base_demand):
        self.id = id
        self.base_demand = base_demand
        self.flexible = random.choice([True, False])  # for DP
        self.generation = random.uniform(0.0, 1.0) if random.random() < 0.5 else 0.0  # for P2P
        self.flex_cost = random.uniform(5, 20)  # €/MWh for LFM

    def step_load(self, price=None):
        """Return household net load (positive = demand, negative = export)"""
        load = self.base_demand + random.gauss(0, 0.3)

        # dynamic pricing response
        if price is not None and self.flexible and price > 0.7:
            load *= 0.7  # reduce 30%

        return load - self.generation  # include generation for P2P


# ------------------------
# Local Energy Market Model
# ------------------------
class LocalEnergyMarketModel:
    def __init__(self, n_households=20, capacity=15, strategy="baseline"):
        self.capacity = capacity
        self.strategy = strategy  # "baseline", "lfm", "p2p", "dp"

        self.households = [Household(i, random.uniform(0.5, 1.5)) for i in range(n_households)]

        # metrics
        self.history = {
            "time": [],
            "total_load": [],
            "grid_load": [],
            "price": [],
            "congestion": [],
            "flex_used": [],
            "flex_cost": [],
            "p2p_traded": [],
        }

    def step(self, t, price=None):
        """Run one timestep depending on strategy"""
        if self.strategy == "lfm":
            grid_load, flex_used, flex_cost = self._step_lfm()
            traded = 0
        elif self.strategy == "p2p":
            grid_load, traded = self._step_p2p()
            flex_used, flex_cost = 0, 0
        elif self.strategy == "dp":
            grid_load = self._step_dp(price)
            traded, flex_used, flex_cost = 0, 0, 0
        elif self.strategy == "baseline":
            grid_load = self._step_baseline()
            traded, flex_used, flex_cost = 0, 0, 0
        else:
            raise ValueError("Unknown strategy")

        # record history
        self.history["time"].append(t)
        self.history["total_load"].append(sum(h.step_load(price) for h in self.households))
        self.history["grid_load"].append(grid_load)
        self.history["price"].append(price if price is not None else 0)
        self.history["congestion"].append(1 if grid_load > self.capacity else 0)
        self.history["flex_used"].append(flex_used)
        self.history["flex_cost"].append(flex_cost)
        self.history["p2p_traded"].append(traded)

    # ---- STRATEGIES ----
    def _step_baseline(self):
        loads = [h.step_load() for h in self.households]
        return sum(loads)

    def _step_lfm(self):
        loads = [h.step_load() for h in self.households]
        total_load = sum(loads)
        flex_used, flex_cost = 0, 0
        grid_load = total_load

        if total_load > self.capacity:
            # Flexibility offers
            offers = [(h, h.base_demand * 0.3, h.flex_cost) for h in self.households]
            offers.sort(key=lambda x: x[2])  # cheapest first
            excess = total_load - self.capacity
            for h, flex, cost in offers:
                if excess <= 0: break
                used = min(flex, excess)
                grid_load -= used
                flex_used += used
                flex_cost += used * cost
                excess -= used

        return grid_load, flex_used, flex_cost

    def _step_p2p(self):
        nets = [h.step_load() for h in self.households]
        total_demand = sum(max(0, n) for n in nets)
        total_supply = -sum(min(0, n) for n in nets)
        traded = min(total_demand, total_supply)
        grid_load = total_demand - traded
        return grid_load, traded

    def _step_dp(self, price):
        loads = [h.step_load(price) for h in self.households]
        return sum(loads)

    # ---- ANALYSIS ----
    def get_results(self):
        return pd.DataFrame(self.history)


# ------------------------
# Run & Plot Example
# ------------------------
if __name__ == "__main__":
    steps = 24
    prices = np.random.rand(steps)  # hourly prices for DP

    # Run models
    models = {}
    for strategy in ["baseline", "lfm", "p2p", "dp"]:
        m = LocalEnergyMarketModel(strategy=strategy)
        for t in range(steps):
            p = prices[t] if strategy == "dp" else None
            m.step(t, price=p)
        models[strategy] = m.get_results()

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for strategy, df in models.items():
        axes[0].plot(df["time"], df["grid_load"], label=strategy.upper())
        axes[1].plot(df["time"], df["congestion"].cumsum(), label=strategy.upper())
        if strategy == "lfm":
            axes[2].plot(df["time"], df["flex_cost"].cumsum(), label="LFM Flex Cost")
        elif strategy == "p2p":
            axes[2].plot(df["time"], df["p2p_traded"].cumsum(), label="P2P Traded")
        elif strategy == "dp":
            axes[2].plot(df["time"], df["price"], label="DP Price")
        elif strategy == "baseline":
            axes[2].plot(df["time"], df["total_load"], label="Baseline Total Load")

    axes[0].set_ylabel("Grid Load (MW)")
    axes[1].set_ylabel("Cumulative Congestions")
    axes[2].set_ylabel("Strategy-specific Metric")
    axes[2].set_xlabel("Time (h)")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
