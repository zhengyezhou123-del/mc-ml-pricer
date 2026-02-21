#!/usr/bin/env python3
"""
ml/train_xgb_optuna.py

Train an XGBoost model using Optuna-best params (JSON). This script is robust to xgboost
API differences: it first tries the sklearn XGBRegressor.fit(...) signature with early stopping;
if that fails due to an unexpected keyword argument, it falls back to xgboost.train(...).

Usage:
    python -m ml.train_xgb_optuna --data data/mc_european_call_with_residual.parquet --params models/best_xgb_params.json --out artifacts/xgb_final_optuna.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
from xgboost import XGBRegressor

# -------------------------
# Helpers
# -------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["moneyness"] = df["S0"] / df["K"]
    df["log_moneyness"] = np.log(np.clip(df["moneyness"].values, 1e-12, None))
    return df

def load_best_params(path: Path) -> tuple[dict, dict]:
    """
    Load Optuna best params JSON and return (mapped_for_sklearn, raw_params).
    Maps: eta -> learning_rate, lambda -> reg_lambda, alpha -> reg_alpha.
    """
    if not path.exists():
        raise FileNotFoundError(f"Params JSON not found: {path}")
    raw = json.loads(path.read_text())
    sk = {}
    for k, v in raw.items():
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
            # keep other keys (may include best_iteration)
            sk[k] = v
    return sk, raw

def prepare_X_y_w(df: pd.DataFrame):
    df = add_features(df)
    # choose target preference: residual_norm -> residual -> price_mc
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
        raise ValueError("Dataset must contain 'residual_norm', 'residual', or 'price_mc'.")

    # feature columns: drop meta columns
    drop_cols = {"residual", "residual_norm", "price_mc", "stderr_mc", "instrument", "bs_call"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check dataframe columns.")
    X = df[feature_cols].astype(float).to_numpy()

    # weights from stderr_mc if available
    w = None
    if "stderr_mc" in df.columns:
        stderr = df["stderr_mc"].to_numpy(dtype=float)
        stderr = np.clip(stderr, 1e-8, None)
        w = np.clip(1.0 / (stderr ** 2), 1e-3, 1e6)

    return X, y, w, feature_cols, unscale_needed

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mc_european_call_with_residual.parquet", help="input parquet dataset")
    p.add_argument("--params", default="models/best_xgb_params.json", help="Optuna best params JSON")
    p.add_argument("--out", default="artifacts/xgb_final_optuna.json", help="output model path")
    p.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--use-best-iter-from-trials", action="store_true", help="try to read best_iteration from models/optuna_trials.csv if present")
    return p.parse_args()

def main():
    args = parse_args()

    data_path = Path(args.data)
    params_path = Path(args.params)
    out_model = Path(args.out)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    # load data
    df = pd.read_parquet(data_path)
    X, y, w, feature_cols, unscale_needed = prepare_X_y_w(df)
    print(f"[INFO] Loaded data: n={len(y)}, features={len(feature_cols)}")

    # train/test split
    idx = np.arange(len(y))
    idx_tr, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_train, X_test = X[idx_tr], X[idx_test]
    y_train, y_test = y[idx_tr], y[idx_test]
    w_train = w[idx_tr] if w is not None else None
    w_test = w[idx_test] if w is not None else None
    df_test = df.iloc[idx_test]

    # load params
    sk_params, raw_params = load_best_params(params_path)

    # determine n_estimators from raw params or trials CSV
    n_estimators = None
    if "best_iteration" in raw_params:
        try:
            n_estimators = int(raw_params["best_iteration"])
        except Exception:
            n_estimators = None

    if n_estimators is None and args.use_best_iter_from_trials:
        trials_csv = Path("models/optuna_trials.csv")
        if trials_csv.exists():
            try:
                tdf = pd.read_csv(trials_csv)
                # find a column containing best_iteration if present (user_attrs)
                cand_cols = [c for c in tdf.columns if "best_iteration" in c]
                if cand_cols:
                    tdf_clean = tdf.dropna(subset=["value"]).sort_values("value")
                    n_estimators = int(tdf_clean.iloc[0][cand_cols[0]])
            except Exception:
                pass

    if n_estimators is None:
        n_estimators = 1000

    # clean sk_params to numeric types where appropriate
    sk_params_clean = {}
    for k, v in sk_params.items():
        # keep non-numeric as-is (rare)
        if isinstance(v, (int, float)):
            sk_params_clean[k] = v
        else:
            try:
                # try convert to int/float
                if isinstance(v, str) and (v.isdigit() or ("." in v) or ("e" in v.lower())):
                    sk_params_clean[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                else:
                    sk_params_clean[k] = v
            except Exception:
                sk_params_clean[k] = v

    # ensure mandatory defaults
    sk_params_clean.setdefault("objective", "reg:squarederror")
    sk_params_clean.setdefault("n_jobs", -1)
    sk_params_clean.setdefault("random_state", int(args.seed))
    sk_params_clean["n_estimators"] = max(10, int(n_estimators))

    print("[INFO] Training XGBRegressor with params:", sk_params_clean)

    model = XGBRegressor(**sk_params_clean)

    # Attempt to use sklearn API fit with early stopping. If the installed xgboost
    # version doesn't accept early_stopping_rounds in fit(), fall back to xgb.train.
    eval_set = [(X_test, y_test)]
    fit_kwargs = {"eval_set": eval_set, "early_stopping_rounds": 50, "verbose": False}

    used_sklearn_fit = False
    best_iter_reported = None

    try:
        if w_train is not None:
            model.fit(X_train, y_train, sample_weight=w_train, **fit_kwargs)
        else:
            model.fit(X_train, y_train, **fit_kwargs)

        # if fit succeeded
        used_sklearn_fit = True
        try:
            # sklearn wrapper stores best_iteration_ in some versions
            best_iter_reported = getattr(model, "best_iteration_", None) or getattr(model, "best_ntree_limit", None)
        except Exception:
            best_iter_reported = None

        # save sklearn model
        model.save_model(str(out_model))
        print(f"[INFO] Saved sklearn-format XGBRegressor to: {out_model}")

        # get predictions for diagnostics
        y_pred = model.predict(X_test)

    except TypeError:
        # Fallback to raw xgboost.train
        print("[WARN] XGBRegressor.fit() didn't accept early_stopping args — falling back to xgb.train()", file=sys.stderr)

        # construct params for xgb.train
        xgb_params = {k: v for k, v in sk_params_clean.items() if k not in ("n_estimators", "n_jobs", "random_state")}
        xgb_params.update({"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "seed": int(args.seed)})

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=(w_train if w_train is not None else None))
        dval = xgb.DMatrix(X_test, label=y_test, weight=(w_test if w_test is not None else None))

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=int(sk_params_clean.get("n_estimators", 1000)),
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # predictions and save booster
        y_pred = bst.predict(xgb.DMatrix(X_test))

        out_model_parent = out_model.parent
        out_model_parent.mkdir(parents=True, exist_ok=True)
        # Save raw booster (xgboost.Booster)
        bst.save_model(str(out_model))
        print(f"[INFO] Saved raw xgboost Booster to: {out_model}")

        best_iter_reported = getattr(bst, "best_iteration", None)

    # Compute diagnostics
    # If we trained on residual_norm, unscale predictions to price units using stderr
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
    if best_iter_reported is not None:
        print(f"[INFO] Best iteration reported by training: {best_iter_reported}")

    # save feature columns list
    (out_model.parent / "feature_columns.txt").write_text("\n".join(feature_cols))
    print(f"[INFO] Saved feature list ({len(feature_cols)} features) to: {out_model.parent / 'feature_columns.txt'}")

if __name__ == "__main__":
    main()
