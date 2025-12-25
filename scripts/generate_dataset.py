import numpy as np
import pandas as pd

from mc_pricer.models import GeometricBrownianMotion
from mc_pricer.engine import MonteCarloEngine
from mc_pricer.payoffs import european_call


def sample_params(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """
    Sample option parameters for dataset rows.
    """
    S0 = rng.uniform(50, 150, size=n)
    K = rng.uniform(50, 150, size=n)
    T = rng.uniform(0.05, 2.0, size=n)
    r = rng.uniform(0.00, 0.07, size=n)
    sigma = rng.uniform(0.05, 0.80, size=n)

    df = pd.DataFrame(
        {
            "S0": S0,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "instrument": ["european_call"] * n,
        }
    )
    return df


def price_row(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int, n_paths: int, seed: int) -> tuple[float, float]:
    """
    Price 1 European call using MC and return (price, stderr).
    """
    model = GeometricBrownianMotion(s0=float(S0), r=float(r), sigma=float(sigma))
    engine = MonteCarloEngine(model, r=float(r))
    out = engine.price(
        T=float(T),
        n_steps=int(n_steps),
        n_paths=int(n_paths),
        payoff_fn=lambda paths: european_call(paths, float(K)),
        seed=int(seed),
    )
    return float(out["price"]), float(out["stderr"])


def main(
    out_path: str = "data/mc_european_call.parquet",
    n_samples: int = 2000,
    n_steps: int = 252,
    n_paths: int = 20000,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    df = sample_params(rng, n_samples)

    prices = []
    stderrs = []

    # Different seed per row for reproducibility but not identical paths
    row_seeds = rng.integers(low=0, high=2**31 - 1, size=n_samples, dtype=np.int64)

    for i, row in df.iterrows():
        price, stderr = price_row(
            S0=row["S0"],
            K=row["K"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"],
            n_steps=n_steps,
            n_paths=n_paths,
            seed=int(row_seeds[i]),
        )
        prices.append(price)
        stderrs.append(stderr)

        if (i + 1) % 200 == 0:
            print(f"Priced {i+1}/{n_samples} rows...")

    df["n_steps"] = n_steps
    df["n_paths"] = n_paths
    df["price_mc"] = prices
    df["stderr_mc"] = stderrs

    # Make sure output folder exists in Codespaces
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.to_parquet(out_path, index=False)
    print(f"\nSaved dataset to: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()

