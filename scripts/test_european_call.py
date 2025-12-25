from mc_pricer.models import GeometricBrownianMotion
from mc_pricer.engine import MonteCarloEngine
from mc_pricer.payoffs import european_call

if __name__ == "__main__":
    # Model parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    model = GeometricBrownianMotion(S0, r, sigma)
    engine = MonteCarloEngine(model, r)

    result = engine.price(
        T=T,
        n_steps=252,
        n_paths=100_000,
        payoff_fn=lambda paths: european_call(paths, K),
        seed=42,
    )

    print("European Call Price (MC):", result["price"])
    print("Standard Error:", result["stderr"])
