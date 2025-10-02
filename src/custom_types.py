# Type aliases for clarity
Bid = tuple[int, float, float]  # (agent_id, price, quantity)
Offer = tuple[int, float, float]  # (agent_id, price, quantity)
MarketResult = tuple[float, float]  # (clearing_price, clearing_quantity)
