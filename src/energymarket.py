import grid


class Basemarket:

    def __init__(self, market_price):
        self.market_price = market_price

    def match_bids(self):
        pass


class Central_market(Basemarket):
    def __init__(self, market_price):
        Basemarket.__init__(self, market_price)


class P2P(Basemarket):
    def __init__(self, market_price):
        Basemarket.__init__(self, market_price)

