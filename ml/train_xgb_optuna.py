#!/usr/bin/env python3
"""
train_xgb.py

Train final XGBoost model using Optuna best params (JSON). If best_iteration is present
in the params JSON (or in models/optuna_trials.csv), it sets n_estimators accordingly.

Usage:
    python train_xgb.py --data data/mc_european_call_with_residual.parquet --params models/best_xgb_params.json --out artifacts/xgb_final.json
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from math import sqrt

from xgboost import XGBRegressor

# -------------------------
# Helpers
# -------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["moneyness"] = df["S0"] / df["K"]
    # guard against non-positive values for log
    df["log_moneyness"] = np.log(np.clip(df["moneyness"].values, 1e-12, None))
    return df

def load_best_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Params JSON not found: {path}")
    p = json.loads(path.read_text())
    # canonicalize keys: map 'lambda' -> 'reg_lambda', 'alpha' -> 'reg_alpha', 'eta' -> 'learning_rate'
    sk = {}
    for k, v in p.items():
        if k == "lambda":
            sk["reg_lambda"] = float(v)
        elif k == "alpha":
            sk["reg_alpha"] = float(v)
        elif k == "eta":
            sk["learning_rate"] = float(v)
        elif k == "max_depth":
            sk["max_depth"] = int(v)
        elif k == "min_child_weight":
            sk["min_child_weight"] = float(v)
        elif k in ("subsample", "colsample_bytree", "gamma"):
            sk[k] = float(v)
        else:
            # ignore any other keys (best_iteration handled separately)
            sk[k] = v
    return sk, p  # sk: mapped for XGBRegressor, p: original dict (may contain best_iteration)

def prepare_X_y_w(df: pd.DataFrame):
    df = add_features(df)
    # choose target in order of preference
    if "residual_norm" in df.columns:
        y = df["residual_norm"].to_numpy(dtype=float)
        target_col = "residual_norm"
        unscale_needed = True
    elif "residual" in df.columns:
        y = df["residual"].to_numpy(dtype=float)
        target_col = "residual"
        unscale_needed = False
    elif "price_mc" in df.columns:
        y = df["price_mc"].to_numpy(dtype=float)
        target_col = "price_mc"
        unscale_needed = False
    else:
        raise ValueError("Dataset must contain 'residual_norm', 'residual', or 'price_mc' column.")

    # features: keep numeric columns excluding target and some meta cols
    drop_cols = {"residual", "residual_norm", "price_mc", "stderr_mc", "instrument", "bs_call"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].astype(float).to_numpy()
    # weights
    w = None
    if "stderr_mc" in df.columns:
        stderr = df["stderr_mc"].to_numpy(dtype=float)
        stderr = np.clip(stderr, 1e-8, None)
        w = np.clip(1.0 / (stderr ** 2), 1e-3, 1e6)
    return X, y, w, feature_cols, unscale_needed

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mc_european_call.parquet", help="input parquet dataset")
    p.add_argument("--params", default="models/best_xgb_params.json", help="Optuna best params JSON")
    p.add_argument("--out", default="artifacts/xgb_final.json", help="output model path (json)")
    p.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-best-iter-from-trials", action="store_true", help="try to read best_iteration from models/optuna_trials.csv if present")
    args = p.parse_args()

    data_path = Path(args.data)
    params_path = Path(args.params)
    out_model = Path(args.out)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    # load data
    df = pd.read_parquet(data_path)
    X, y, w, feature_cols, unscale_needed = prepare_X_y_w(df)

    # split for evaluation
    idx = np.arange(len(y))
    idx_tr, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_train, X_test = X[idx_tr], X[idx_test]
    y_train, y_test = y[idx_tr], y[idx_test]
    w_train = w[idx_tr] if w is not None else None
    w_test = w[idx_test] if w is not None else None
    df_test = df.iloc[idx_test]

    # load params (map keys)
    sk_params, raw_params = load_best_params(params_path)
    # determine n_estimators: prefer raw 'best_iteration' if present
    n_estimators = None
    if "best_iteration" in raw_params:
        try:
            n_estimators = int(raw_params["best_iteration"])
        except Exception:
            n_estimators = None

    # fallback: if trials CSV present and user asked, try to pull best_iteration column
    if n_estimators is None and args.use_best_iter_from_trials:
        trials_csv = Path("models/optuna_trials.csv")
        if trials_csv.exists():
            try:
                tdf = pd.read_csv(trials_csv)
                # look for user_attrs_best_iteration column (optuna may store it as user_attrs.best_iteration)
                col = [c for c in tdf.columns if "best_iteration" in c]
                if col:
                    # find the best (smallest) value row by 'value' metric
                    tdf_clean = tdf.dropna(subset=["value"]).sort_values("value")
                    val = tdf_clean.iloc[0][col[0]]
                    n_estimators = int(val)
            except Exception:
                pass

    if n_estimators is None:
        # default fallback
        n_estimators = 1000

    # safe mapping for XGBRegressor keys
    # ensure numeric types
    sk_params_clean = {}
    for k, v in sk_params.items():
        if isinstance(v, (int, float)):
            sk_params_clean[k] = v
        else:
            try:
                sk_params_clean[k] = float(v) if ('.' in str(v) or 'e' in str(v).lower()) else int(v)
            except Exception:
                sk_params_clean[k] = v

    # ensure mandatory keys
    sk_params_clean.setdefault("objective", "reg:squarederror")
    sk_params_clean.setdefault("n_jobs", -1)
    sk_params_clean.setdefault("random_state", int(args.seed))
    sk_params_clean["n_estimators"] = max(10, int(n_estimators))

    print("[INFO] Training XGBRegressor with params:", sk_params_clean)
    model = XGBRegressor(**sk_params_clean)

    # fit with early stopping and validation set
    eval_set = [(X_test, y_test)]
    fit_kwargs = {"eval_set": eval_set, "early_stopping_rounds": 50, "verbose": False}
    if w_train is not None:
        model.fit(X_train, y_train, sample_weight=w_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    # predictions
    y_pred = model.predict(X_test)

    # if target was residual_norm, unscale to residuals for reporting
    if unscale_needed and "stderr_mc" in df.columns:
        stderr_test = df_test["stderr_mc"].to_numpy(dtype=float)
        y_test_unscaled = y_test * stderr_test
        y_pred_unscaled = y_pred * stderr_test
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        rmse = sqrt(np.mean((y_test_unscaled - y_pred_unscaled) ** 2))
        mean_stderr = stderr_test.mean()
        ratio = rmse / mean_stderr if mean_stderr > 0 else float("inf")
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(np.mean((y_test - y_pred) ** 2))
        mean_stderr = df_test["stderr_mc"].mean() if "stderr_mc" in df_test.columns else float("nan")
        ratio = rmse / mean_stderr if mean_stderr > 0 else float("nan")

    print("\n=== Test Metrics ===")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean MC stderr (test): {mean_stderr:.6f}")
    print(f"RMSE / mean_stderr: {ratio:.3f}")

    # save model (sklearn API)
    model.save_model(str(out_model))
    print(f"\nSaved model to: {out_model}")

    # Optional: save feature columns used
    (out_model.parent / "feature_columns.txt").write_text("\n".join(feature_cols))
    print(f"Saved feature list with {len(feature_cols)} features.")

if __name__ == "__main__":
    main()
