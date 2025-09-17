import random
import numpy as np

class Household:
    def __init__(self, id, base_demand):
        self.id = id
        self.base_demand = base_demand # average demand, ~0.5-1.5 kW in toy units
        self.flexible = random.choice([True, False])  # for DP (whether it can shift demand under dynamic pricing)
        self.generation = random.uniform(0.0, 1.0) if random.random() < 0.5 else 0.0  # for P2P # solar PV or not
        self.flex_cost = random.uniform(5, 20)  # €/MWh for LFM. (how expensive it is to provide flexibility in LFM)

    def step_load(self, price=None):
        """Return household net load (positive = demand, negative = export)"""
        load = self.base_demand + random.gauss(0, 0.3) # each timestep, the household's demand fluctuates

        # dynamic pricing response
        if price is not None and self.flexible and price > 0.7:
            load *= 0.7  # reduce 30%.   # under DP: if price is high and household is flexible -> reduces load

        return load - self.generation  # include generation for P2P (it can make the load negative for P2P - surplus)


class LocalEnergyMarketModel: #creates a local grid model
    def __init__(self, n_households=20, capacity=15, strategy="lfm"):
        self.capacity = capacity # is max feeder load before congestion
        self.strategy = strategy  # picks one of the 3 coordination mechanisms: "lfm", "p2p", or "dp"
        self.households = [Household(i, random.uniform(0.5, 1.5)) for i in range(n_households)]

        # metrics
        self.congestion_events = 0 # how many times feeder overloaded
        self.total_cost = 0 # DSO spending in LFM
        self.traded_energy = 0 # amount of P2P energy exchanged

    def step(self, price=None):
        """Run one timestep depending on strategy"""
        if self.strategy == "lfm":
            return self._step_lfm()
        elif self.strategy == "p2p":
            return self._step_p2p()
        elif self.strategy == "dp":
            return self._step_dp(price)
        else:
            raise ValueError("Unknown strategy")

####### Local Flexibility Market #######
    def _step_lfm(self): # Compute total load. 
        loads = [h.step_load() for h in self.households]
        total_load = sum(loads)
        if total_load > self.capacity: #If load > capacity → congestion
            self.congestion_events += 1
            # Flexibility offers. #DSO buys flexibility offers (sorted by cost).
            offers = [(h, h.base_demand * 0.3, h.flex_cost) for h in self.households]
            offers.sort(key=lambda x: x[2])  # cheapest flexibility first
            excess = total_load - self.capacity # Reduces excess demand until capacity is respected
            for h, flex, cost in offers:
                if excess <= 0: break
                used = min(flex, excess)
                total_load -= used
                excess -= used
                self.total_cost += used * cost # Accumulates system cost.
        return total_load

####### Peer-to-Peer Trading #######
    def _step_p2p(self):
        nets = [h.step_load() for h in self.households] #Compute net demand (positive = need, negative = surplus).
        total_demand = sum(max(0, n) for n in nets) 
        total_supply = -sum(min(0, n) for n in nets)
        traded = min(total_demand, total_supply) # Consumers buy from prosumers, matching supply/demand.
        self.traded_energy += traded 
        grid_load = total_demand - traded #Grid only supplies unmet demand.
        if grid_load > self.capacity: #If grid load > capacity → congestion
            self.congestion_events += 1. #Track traded energy
        return grid_load

####### Dynamic Pricing #######
    def _step_dp(self, price):
        loads = [h.step_load(price) for h in self.households] #Every household sees a price.
        total_load = sum(loads)
        if total_load > self.capacity:   #Flexible households cut demand when price > 0.7. 
            self.congestion_events += 1 #If load > capacity → congestion
        return total_load


# --- Example runs ---
if __name__ == "__main__":
    steps = 24
    prices = np.random.rand(steps)  # hourly prices for DP

    # LFM
    lfm_model = LocalEnergyMarketModel(strategy="lfm")
    for _ in range(steps):
        lfm_model.step()
    print("LFM -> Congestions:", lfm_model.congestion_events, " Cost:", round(lfm_model.total_cost, 2))

    # P2P
    p2p_model = LocalEnergyMarketModel(strategy="p2p")
    for _ in range(steps):
        p2p_model.step()
    print("P2P -> Congestions:", p2p_model.congestion_events, " Traded:", round(p2p_model.traded_energy, 2))

    # DP
    dp_model = LocalEnergyMarketModel(strategy="dp")
    for p in prices:
        dp_model.step(price=p)
    print("DP -> Congestions:", dp_model.congestion_events)
