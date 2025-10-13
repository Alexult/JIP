# Type aliases for clarity
from typing import Callable

Bid = tuple[int, float, float]  # (agent_id, price, quantity)
Offer = tuple[int, float, float]  # (agent_id, price, quantity)
MarketResult = tuple[float, float]  # (clearing_price, clearing_quantity)
Job = tuple[float, int, Callable]
