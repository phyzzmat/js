import numpy as np
from numpy import ndarray
import scipy.stats as s
from functools import partial


class Distribution:

    def __init__(self, partial, repr, **kwargs) -> None:
        self.sampler = partial
        self.repr = repr

    def __call__(self) -> np.ndarray:
        return self.sampler()

    def __str__(self):
        if self.repr is None:
            return f"[ Distribution ]"
        else:
            return self.repr


def get_bernoulli():
    p = s.uniform.rvs(loc=0)
    bernoulli_sampler = partial(
        lambda: s.bernoulli.rvs(
            size=1,
            p=p
        )
    )
    return Distribution(
        partial=bernoulli_sampler,
        repr=f"Bern({round(p, 3)})"
    )


def get_poisson():
    lmb = s.geom.rvs(loc=0, p=0.2)
    bernoulli_sampler = partial(
        lambda: s.poisson.rvs(
            lmb,
            size=1
        )
    )
    return Distribution(
        partial=bernoulli_sampler,
        repr=f"Pois({lmb})"
    )


def get_geom():
    p = s.uniform.rvs()
    bernoulli_sampler = partial(
        lambda: s.geom.rvs(
            p,
            size=1
        )
    )
    return Distribution(
        partial=bernoulli_sampler,
        repr=f"Geom({round(p, 3)})"
    )

def get_bin():
    p = s.uniform.rvs()
    n = s.geom.rvs(p=0.1)
    bernoulli_sampler = partial(
        lambda: s.binom.rvs(
            n, p,
            size=1
        )
    )
    return Distribution(
        partial=bernoulli_sampler,
        repr=f"Bin({n}, {round(p, 3)})"
    )


def get_uniform_discrete():
    n = s.geom.rvs(loc=0, p=0.2)
    bernoulli_sampler = partial(
        lambda: s.randint.rvs(
            low=1,
            high=n + 1,
            size=1
        )
    )
    return Distribution(
        partial=bernoulli_sampler,
        repr=f"Dice({n})"
    )


def generate_distribution(config):

    CHOICES = [
        # "constant", # "normal", "geometric", "exponential",
        "bernoulli", "poisson", "uniform_discrete", "bin", "geom"
        # "uniform", "uniform_discrete"
    ]

    items = sorted(list(config['distribution_probs'].items()),
                   key=lambda x: CHOICES.index(x[0]))
    probs = list(map(lambda x: x[1], items))
    assert len(items) == len(CHOICES)
    assert abs(sum(probs) - 1) < 1e-6, "Sum of probabilities not equal to 1"
    choice = np.random.choice(CHOICES, p=probs)

    if choice == "bernoulli":
        return get_bernoulli()
    elif choice == "poisson":
        return get_poisson()
    elif choice == "uniform_discrete":
        return get_uniform_discrete()
    elif choice == "bin":
        return get_bin()
    elif choice == "geom":
        return get_geom()
    else:
        assert False, "SUCTION"
