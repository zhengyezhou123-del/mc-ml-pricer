import numpy as np

def european_call(paths: np.ndarray, K: float) -> np.ndarray:
    """
    European call payoff: max(S_T - K, 0)
    """
    S_T = paths[:, -1]
    return np.maximum(S_T - K, 0.0)


def european_put(paths: np.ndarray, K: float) -> np.ndarray:
    """
    European put payoff: max(K - S_T, 0)
    """
    S_T = paths[:, -1]
    return np.maximum(K - S_T, 0.0)

