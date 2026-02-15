"""

Functions:
- simulate_black_scholes_terminal: vectorized terminal S_T sampling (plain, antithetic)
- bs_call_price: Black-Scholes analytic price for control variate
- mc_price_european_call: plain MC price (baseline)
- mc_price_european_call_antithetic_controlvar: antithetic + control variate estimator
- qmc_price_european_call_sobol: optional QMC (Sobol + inverse CDF)
- importance_sampling_shifted: example importance sampling by shifting normal mean
- bootstrap_ci: bootstrap confidence interval for estimator
- example_main: small demo (if run as __main__)
"""

import numpy as np
from scipy.stats import norm
try:
    from scipy.stats import qmc
    _HAS_QMC = True
except Exception:
    _HAS_QMC = False

# Optional: uncomment if you want to JIT the hot function (numba must be installed)
# from numba import njit, prange

# ---------------------------
# Helper: analytic Black-Scholes (European call)
# ---------------------------
def bs_call_price(S0, K, r, sigma, T):
    """Black-Scholes price for European call."""
    if T <= 0:
        return max(S0 - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ---------------------------
# Terminal sampler (vectorized)
# ---------------------------
def simulate_black_scholes_terminal(S0, r, sigma, T, N, antithetic=False, rng=None):
    """
    Simulate N terminal prices S_T under Black-Scholes risk-neutral law.
    If antithetic=True and N is odd, N+1 samples are generated then truncated to N.
    Returns array shape (N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    if not antithetic:
        Z = rng.standard_normal(size=N)
    else:
        half = (N + 1) // 2
        Z_half = rng.standard_normal(size=half)
        Z = np.concatenate([Z_half, -Z_half])
        Z = Z[:N]

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return ST

# ---------------------------
# Plain Monte Carlo price
# ---------------------------
def mc_price_european_call(S0, K, r, sigma, T, N, rng=None):
    ST = simulate_black_scholes_terminal(S0, r, sigma, T, N, antithetic=False, rng=rng)
    payoff = np.maximum(ST - K, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    stderr = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(N)
    return price, stderr

# ---------------------------
# Antithetic + Control Variate estimator
# ---------------------------
def mc_price_european_call_antithetic_controlvar(S0, K, r, sigma, T, N, rng=None):
    """
    Use antithetic sampling and control variate with analytic call on S_T (or with S_T itself).
    Two options implemented:
     - control variable = S_T (its expectation under Q is S0 * exp(r*T))
     - or use analytic BS price as fixed control (scalar) — better: use S_T as control variable.
    Here we use control = S_T (final underlying) because E[S_T] is known.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Antithetic S_T
    ST = simulate_black_scholes_terminal(S0, r, sigma, T, N, antithetic=True, rng=rng)
    payoff = np.maximum(ST - K, 0.0)
    # control = ST (final price). Its expectation E_control = S0 * exp(r*T)
    control = ST
    E_control = S0 * np.exp(r * T)

    # compute beta via sample covariance (unbiased-ish)
    cov = np.cov(payoff, control, ddof=1)
    cov_pc = cov[0, 1]
    var_c = cov[1, 1]
    # guard against zero variance
    beta = cov_pc / var_c if var_c > 0 else 0.0

    adjusted = payoff - beta * (control - E_control)
    price = np.exp(-r * T) * adjusted.mean()
    stderr = np.exp(-r * T) * adjusted.std(ddof=1) / np.sqrt(N)
    return price, stderr, beta

# ---------------------------
# Importance sampling: change normal mean (simple shift)
# ---------------------------
def importance_sampling_shifted(S0, K, r, sigma, T, N, shift, rng=None):
    """
    Importance sampling by shifting the normal by 'shift' (i.e., sample Z ~ N(shift,1))
    Weight = exp(-shift * Z + 0.5 * shift^2)  (likelihood ratio)
    Example: for deep-OTM calls, choose positive shift to push mass to large ST.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal(size=N) + shift
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    # likelihood ratio: phi(z)/phi(z-shift) = exp(-0.5*z^2) / exp(-0.5*(z-shift)^2)
    # simplified: exp(-shift*z + 0.5*shift^2)
    lr = np.exp(-shift * (Z - shift) - 0.5 * shift**2)  # equivalent; keep numerical stable formula
    # alternative: lr = np.exp(-shift*Z + 0.5*shift**2)
    weighted_payoff = payoff * lr
    price = np.exp(-r * T) * weighted_payoff.mean()
    stderr = np.exp(-r * T) * weighted_payoff.std(ddof=1) / np.sqrt(N)
    return price, stderr

# ---------------------------
# Quasi-Monte Carlo (Sobol) for terminal price (uniform->normal)
# ---------------------------
def qmc_price_european_call_sobol(S0, K, r, sigma, T, N, scramble=True, rng_seed=None):
    """
    Use Sobol low-discrepancy points to draw normals via inverse CDF.
    NOTE: scipy.stats.qmc.Sobol requires N to be a power of two for some uses, but it works more generally.
    If scipy is not installed, raises RuntimeError.
    """
    if not _HAS_QMC:
        raise RuntimeError("scipy.stats.qmc required for QMC; install scipy.")

    dim = 1
    sob = qmc.Sobol(d=dim, scramble=scramble, seed=rng_seed)
    u = sob.random(n=N).reshape(N)  # in (0,1)
    # avoid exactly 0 or 1
    u = np.clip(u, 1e-12, 1 - 1e-12)
    Z = norm.ppf(u)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    # QMC error estimation: use randomized QMC or digital shift with multiple independent scramblings
    # Here we return the mean; user should repeat with different seeds to estimate error.
    return price

# ---------------------------
# Bootstrap CI
# ---------------------------
def bootstrap_ci(samples, discount_factor, n_boot=1000, alpha=0.05, rng=None):
    """
    Bootstrap confidence interval for discounted estimator.
    samples: array of payoffs (undiscounted)
    discount_factor: exp(-r*T)
    Returns (lower, upper) at (1-alpha) confidence, and bootstrap mean.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(samples)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = samples[idx].mean()
    lower = np.quantile(boot_means * discount_factor, alpha / 2)
    upper = np.quantile(boot_means * discount_factor, 1 - alpha / 2)
    mean = samples.mean() * discount_factor
    return mean, lower, upper

# ---------------------------
# Example / quick experiment driver
# ---------------------------
def example_main():
    import time

    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    N = 200_000
    rng = np.random.default_rng(12345)

    print("Analytic BS price:", bs_call_price(S0, K, r, sigma, T))

    t0 = time.time()
    p_plain, se_plain = mc_price_european_call(S0, K, r, sigma, T, N, rng=rng)
    t1 = time.time()
    print(f"Plain MC: price={p_plain:.6f}, stderr={se_plain:.6f}, time={t1-t0:.3f}s")

    rng = np.random.default_rng(12345)
    t0 = time.time()
    p_acv, se_acv, beta = mc_price_european_call_antithetic_controlvar(S0, K, r, sigma, T, N, rng=rng)
    t1 = time.time()
    print(f"Antithetic+ControlVar: price={p_acv:.6f}, stderr={se_acv:.6f}, beta={beta:.4f}, time={t1-t0:.3f}s")

    # importance sampling example (shift positive)
    rng = np.random.default_rng(12345)
    t0 = time.time()
    p_is, se_is = importance_sampling_shifted(S0, K, r, sigma, T, N, shift=1.0, rng=rng)
    t1 = time.time()
    print(f"Importance sampling (shift=1.0): price={p_is:.6f}, stderr={se_is:.6f}, time={t1-t0:.3f}s")

    if _HAS_QMC:
        p_qmc = qmc_price_european_call_sobol(S0, K, r, sigma, T, 2**14, scramble=True, rng_seed=123)
        print(f"QMC Sobol (single run): price={p_qmc:.6f}")

if __name__ == "__main__":
    example_main()
