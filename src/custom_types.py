# Type aliases for clarity
Bid = tuple[int,float, float]  # (price, quantity)
Offer = tuple[int,float, float]  # (price, quantity)
MarketResult = tuple[float, float]  # (clearing_price, clearing_quantity)
