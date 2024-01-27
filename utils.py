from dataclasses import dataclass

import numpy as np

@dataclass
class Quote:
    bid_price: float
    ask_price: float
    bid_amount: float
    ask_amount: float

    def __str__(self) -> str:
        return f"{self.bid_price}  {self.bid_amount} | {self.ask_price}  {self.ask_amount}"


def get_spread(config, expr, n_boot=1000):
    std = np.array([expr() for _ in range(n_boot)]).std()
    return np.random.random() * std / config['trading_session']['difficulty']


def bootstrap(expr, n=10000):
    return np.array([expr() for _ in range(n)])


def check_spread(quote: Quote, spread: float, is_absolute=True):
    return quote is not None and quote.ask_price - quote.bid_price <= spread


def get_side(expr, quote, n_boot=10):
    """
    AI chooses side to trade as the best for him as a result of n_boot trials. 
    """
    mu_hat = sum([expr() for _ in range(n_boot)]) / n_boot
    return 'ask' if quote.ask_price - mu_hat < mu_hat - quote.bid_price else 'bid'


def resolve_market(expr, quote, side):
    value = expr()
    return value, ((quote.ask_price - value) * quote.ask_amount) if side == 'ask' else \
                ((value - quote.bid_price) * quote.bid_amount)
    