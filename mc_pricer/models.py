
import numpy as np

class GeometricBrownianMotion:
    """
    Risk-neutral GBM model:
        dS = r S dt + sigma S dW
    """

    def __init__(self, s0: float, r: float, sigma: float):
        self.s0 = s0
        self.r = r
        self.sigma = sigma

    def simulate_paths(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths.

        Returns
        -------
        paths : ndarray of shape (n_paths, n_steps + 1)
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = self.s0

        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            z = rng.standard_normal(n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * z)

        return paths
