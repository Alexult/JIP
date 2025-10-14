# Type aliases for clarity
from typing import Callable

Bid = tuple[int, float, float]  # (agent_id, price, quantity)
Offer = tuple[int, float, float]  # (agent_id, price, quantity)
MarketResult = tuple[float, float]  # (clearing_price, clearing_quantity)
Job = tuple[float, int, Callable]

"""
examples of job flexibility
fixed = lambda job, t: 1 if job[1] == t else 0

c = 0.5
linear = lambda job, t: max(1 - abs(job[1] - t) * c, 0)


a = 0.2
b = 0.1
quadratic = lambda job, t: max(abs(job[1] - t)**2 * a + abs(job[1] - t) * b, 0)

free = lambda job, t: 1
"""