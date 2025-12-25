import numpy as np
from typing import Callable

class MonteCarloEngine:
    """
    Generic Monte Carlo pricing engine.
    """

    def __init__(self, model, r: float):
        self.model = model
        self.r = r

    def price(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        seed: int | None = None,
    ) -> dict:
        """
        Price an option via Monte Carlo.
        """
        paths = self.model.simulate_paths(
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )

        payoffs = payoff_fn(paths)
        discounted = np.exp(-self.r * T) * payoffs

        price = discounted.mean()
        std = discounted.std(ddof=1)
        stderr = std / np.sqrt(n_paths)

        return {
            "price": price,
            "stderr": stderr,
        }

