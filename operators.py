from typing import Any
import numpy as np
from distributions import Distribution


class Sum(Distribution):

    def __init__(self, partial) -> None:
        super().__init__()
        self.partial = partial
        self.repr = "+"

    def __call__(self) -> Any:
        return self.partial()


class Sum(Distribution):

    def __init__(self, partial) -> None:
        super().__init__()
        self.partial = partial
        self.repr = "*"

    def __call__(self) -> Any:
        return self.partial()


class BootstrapMean(Distribution):

    def __init__(self, partial, n=10) -> None:
        super().__init__()
        self.n = n
        self.partial = partial
        self.repr = f"Bootstrap x{n}"

    def __call__(self) -> Any:
        out = []
        for i in range(self.n):
            out.append(self.partial())
        return np.array(out).mean()


def get_bootstrap_mean(left, n=10):

    def bootstrap():
        out = []
        for i in range(n):
            out.append(left())
        return np.array(out).mean()

    return Distribution(
        partial=bootstrap,
        repr=f"Bootstrap mean x{n}"
    )


def get_bootstrap_max(left, n=10):

    def bootstrap():
        out = []
        for i in range(n):
            out.append(left())
        return np.array(out).max()

    return Distribution(
        partial=bootstrap,
        repr=f"Bootstrap max x{n}"
    )


def get_sum(left, right):
    return Distribution(
        partial=lambda: left() + right(),
        repr=f"+"
    )


def get_mul(left, right):
    return Distribution(
        partial=lambda: left() * right(),
        repr=f"*"
    )


def generate_binary_op(config, left, right):
    CHOICES = [
        "add", "mul"
    ]
    items = sorted(list(config['operator_generation']['binary_op_probs'].items()),
                   key=lambda x: CHOICES.index(x[0]))
    probs = list(map(lambda x: x[1], items))
    assert len(items) == len(CHOICES)
    assert abs(sum(probs) - 1) < 1e-6, "Sum of probabilities not equal to 1"
    choice = np.random.choice(CHOICES, p=probs)

    if choice == "add":
        return get_sum(left, right)
    elif choice == "mul":
        return get_mul(left, right)
    else:
        assert False, "Unreachable"


def generate_unary_op(config, left):
    CHOICES = [
        "bootstrap_mean", "bootstrap_max"  # "normal", "geometric", "exponential",
        # "uniform", "uniform_discrete"
    ]
    items = sorted(list(config['operator_generation']['unary_op_probs'].items()),
                   key=lambda x: CHOICES.index(x[0]))
    probs = list(map(lambda x: x[1], items))
    assert len(items) == len(CHOICES)
    assert abs(sum(probs) - 1) < 1e-6, "Sum of probabilities not equal to 1"
    choice = np.random.choice(CHOICES, p=probs)

    if choice == "bootstrap_mean":
        return get_bootstrap_mean(left, n=np.random.choice([2, 10, 50]))
    elif choice == "bootstrap_max":
        return get_bootstrap_max(left, n=np.random.choice([2, 10, 50]))
    else:
        assert False, "Unreachable"
