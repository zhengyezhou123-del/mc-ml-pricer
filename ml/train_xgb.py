
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a couple of standard finance features:
    - moneyness = S0 / K
    - log_moneyness = log(S0 / K)
    """
    df = df.copy()
    df["moneyness"] = df["S0"] / df["K"]
    df["log_moneyness"] = np.log(df["moneyness"])
    return df


def main(
    data_path: str = "data/mc_european_call.parquet",
    out_dir: str = "artifacts",
    test_size: float = 0.2,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(data_path)
    df = add_features(df)

    feature_cols = ["S0", "K", "T", "r", "sigma", "moneyness", "log_moneyness"]
    target_col = "price_mc"

    X = df[feature_cols].values
    y = df[target_col].values

    # Weight by inverse Monte Carlo variance (noise-aware learning)
    # Add small epsilon to avoid exploding weights when stderr is tiny.
    w = 1.0 / (df["stderr_mc"].values ** 2 + 1e-12)

    X_train, X_test, y_train, y_test, w_train, w_test, df_train, df_test = train_test_split(
        X, y, w, df, test_size=test_size, random_state=seed
    )

    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== Test Metrics ===")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean MC stderr (test): {df_test['stderr_mc'].mean():.6f}")
    print(f"RMSE / mean_stderr: {rmse / df_test['stderr_mc'].mean():.3f}")

    # Save model
    model_path = os.path.join(out_dir, "xgb_european_call.json")
    model.save_model(model_path)
    print(f"\nSaved model to: {model_path}")

    # ---- Plot 1: Pred vs True scatter ----
    plt.figure()
    plt.scatter(y_test, y_pred, s=8)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("MC price (true)")
    plt.ylabel("XGB price (pred)")
    plt.title("European Call: Prediction vs MC")
    scatter_path = os.path.join(out_dir, "pred_vs_true.png")
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {scatter_path}")

    # ---- Plot 2: Error vs moneyness (binned) ----
    df_eval = df_test.copy()
    df_eval["pred"] = y_pred
    df_eval["abs_err"] = np.abs(df_eval["pred"] - df_eval["price_mc"])

    # Bin by moneyness
    bins = np.linspace(df_eval["moneyness"].min(), df_eval["moneyness"].max(), 21)
    df_eval["m_bin"] = pd.cut(df_eval["moneyness"], bins=bins, include_lowest=True)

    bin_stats = df_eval.groupby("m_bin", observed=True)["abs_err"].mean().reset_index()
    bin_centers = []
    for interval in bin_stats["m_bin"]:
        bin_centers.append((interval.left + interval.right) / 2.0)

    plt.figure()
    plt.plot(bin_centers, bin_stats["abs_err"].values, marker="o")
    plt.xlabel("Moneyness (S0/K)")
    plt.ylabel("Mean absolute error")
    plt.title("European Call: MAE vs Moneyness (binned)")
    err_path = os.path.join(out_dir, "mae_vs_moneyness.png")
    plt.savefig(err_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {err_path}")


if __name__ == "__main__":
    main()
