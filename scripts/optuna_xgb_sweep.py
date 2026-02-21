#!/usr/bin/env python3
"""
Robust Optuna sweep for XGBoost (revised).

Key revisions:
- Prefer 'residual_norm' (if present), then 'residual', then 'price_mc' as target.
- Tighter search ranges to reduce overfitting risk (eta in [0.01,0.15], max_depth in [3,8]).
- All other features (synth fallback, pruning, sqlite storage, stderr-based weights, saving params) retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default="data/mc_european_call.parquet",
        help="Path to input parquet dataset",
    )
    p.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_xgb.db (optional)",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="xgb_residual",
        help="Optuna study name (when storage provided)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--prune", action="store_true", help="Enable XGBoost pruning callback")
    p.add_argument(
        "--allow-synth",
        action="store_true",
        help="Use small synthetic dataset if data file missing (debug)",
    )
    p.add_argument(
        "--out-params",
        type=str,
        default="models/best_xgb_params.json",
        help="Where to save best params",
    )
    return p.parse_args()


def load_data_or_synth(path: Path, allow_synth: bool = False):
    """
    Load parquet dataset. If missing and allow_synth is True, return a tiny synthetic dataset useful for debugging.
    """
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet {path}: {e}") from e
        return df
    else:
        if not allow_synth:
            raise FileNotFoundError(
                f"Data file not found: {path}. Run your dataset generator or pass --allow-synth to use a tiny synthetic dataset for debugging."
            )
        warnings.warn(
            f"{path} not found — using a small synthetic dataset for debugging (use real data for final runs)."
        )
        rng = np.random.default_rng(123)
        N = 500
        df = pd.DataFrame(
            {
                "S0": rng.uniform(50, 150, N),
                "K": rng.uniform(50, 150, N),
                "T": rng.uniform(0.05, 2.0, N),
                "r": rng.uniform(0.0, 0.07, N),
                "sigma": rng.uniform(0.05, 0.8, N),
                "price_mc": rng.normal(5.0, 1.0, N),
                "stderr_mc": np.abs(rng.normal(0.01, 0.005, N)),
            }
        )
        return df


def prepare_xyw(df: pd.DataFrame):
    """
    Create X (np.ndarray), y (np.ndarray), w (np.ndarray|None) from dataframe.

    Preference order for target:
      1) residual_norm (if present)
      2) residual
      3) price_mc (fallback)
    """
    df = df.copy()
    target_col = None
    if "residual_norm" in df.columns:
        target_col = "residual_norm"
        y = df["residual_norm"].values.astype(float)
    elif "residual" in df.columns:
        target_col = "residual"
        y = df["residual"].values.astype(float)
    elif "price_mc" in df.columns:
        warnings.warn(
            "Columns 'residual_norm' and 'residual' not found — falling back to using 'price_mc' as target. Prefer storing residual target in the dataset."
        )
        target_col = "price_mc"
        y = df["price_mc"].values.astype(float)
    else:
        raise ValueError(
            "Dataset does not contain 'residual_norm', 'residual' or 'price_mc' columns. Please add a residual or price column."
        )

    # sample weights from stderr if available
    w = None
    if "stderr_mc" in df.columns:
        stderr = df["stderr_mc"].astype(float).values
        stderr_clipped = np.clip(stderr, 1e-6, None)
        w = 1.0 / (stderr_clipped ** 2)
        w = np.clip(w, 1e-3, 1e6)

    # Drop known non-feature columns (keep bs_call if present only if desired)
    drop_cols = {"residual", "residual_norm", "price_mc", "stderr_mc", "instrument"}
    features = [c for c in df.columns if c not in drop_cols]
    if len(features) == 0:
        raise ValueError("No feature columns found after dropping target columns. Check the dataframe columns.")
    X = df[features].astype(float).values

    return X, y, w, features, target_col


def objective_factory(X_full, y_full, w_full, args, features):
    """
    Return an objective(trial) closure with captured dataset.
    """

    def objective(trial: optuna.trial.Trial):
        # sample hyperparams (tighter ranges)
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eval_metric": "rmse",
            "verbosity": 0,
            # learning rate
            "eta": trial.suggest_float("eta", 1e-2, 0.15, log=True),
            # tree depth
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            # regularization
            "lambda": trial.suggest_float("lambda", 1e-6, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True),
            # subsampling
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # other
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 20.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "seed": int(args.seed),
        }

        # create a seeded train/val split (consistent across trials)
        idx = np.arange(len(y_full))
        idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=int(args.seed))
        X_tr, X_val = X_full[idx_tr], X_full[idx_val]
        y_tr, y_val = y_full[idx_tr], y_full[idx_val]
        if w_full is not None:
            w_tr, w_val = w_full[idx_tr], w_full[idx_val]
        else:
            w_tr = w_val = None

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

        callbacks = []
        if args.prune:
            # use the eval name/metric expected by XGBoostPruningCallback
            callbacks.append(XGBoostPruningCallback(trial, "validation_0-rmse"))

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dval, "validation_0")],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=callbacks if callbacks else None,
        )

        preds = bst.predict(dval)
        # compute RMSE manually to avoid sklearn version-specific keyword args
        mse = np.mean((y_val - preds) ** 2)
        rmse = float(np.sqrt(mse))

        # record best iteration (useful when retraining on full data)
        trial.set_user_attr(
            "best_iteration",
            int(
                getattr(
                    bst,
                    "best_iteration",
                    bst.best_ntree_limit if hasattr(bst, "best_ntree_limit") else 0,
                )
            ),
        )

        return float(rmse)

    return objective


def main():
    args = parse_args()
    data_path = Path(args.data)
    out_params = Path(args.out_params)
    out_params.parent.mkdir(parents=True, exist_ok=True)

    # load dataset (or synth)
    try:
        df = load_data_or_synth(data_path, allow_synth=args.allow_synth)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    X, y, w, features, target_col = prepare_xyw(df)
    print(f"[INFO] Loaded dataset: {data_path} -> features={len(features)} cols: {features}, n={len(y)}, target='{target_col}'")

    # create study
    if args.storage:
        study = optuna.create_study(direction="minimize", storage=args.storage, study_name=args.study_name, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize")

    objective = objective_factory(X, y, w, args, features)
    print(f"[INFO] Starting Optuna sweep (trials={args.n_trials}, seed={args.seed})")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n[RESULT] Best trial:")
    print("  RMSE:", study.best_value)
    print("  Params:", study.best_params)

    # Save best params (canonicalize)
    best_params = dict(study.best_params)
    best_params.update({"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "seed": int(args.seed)})
    out_params.write_text(json.dumps(best_params, indent=2))
    print(f"[INFO] Saved best params to: {out_params.resolve()}")

    if args.storage:
        print(f"[INFO] Optuna study persisted at: {args.storage}")

    # Save trials dataframe for later analysis
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv("models/optuna_trials.csv", index=False)
        print("[INFO] Saved trials dataframe to models/optuna_trials.csv")
    except Exception:
        pass


if __name__ == "__main__":
    main()
