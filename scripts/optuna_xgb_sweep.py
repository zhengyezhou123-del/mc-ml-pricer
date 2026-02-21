import optuna
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path


# ==============================
# LOAD YOUR DATA HERE
# ==============================

def load_data():
    # replace with your real loading logic
    df = pd.read_parquet("data/train.parquet")

    y = df["residual"].values
    X = df.drop(columns=["residual"]).values

    # optional noise-aware weights
    if "weight" in df.columns:
        w = df["weight"].values
    else:
        w = None

    return X, y, w


# ==============================
# OPTUNA OBJECTIVE
# ==============================

def objective(trial):

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "verbosity": 0,

        # learning rate
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),

        # tree complexity
        "max_depth": trial.suggest_int("max_depth", 3, 10),

        # regularization
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),

        # subsampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # robustness
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }

    X, y, w = load_data()

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    rmse = mean_squared_error(y_val, preds, squared=False)

    return rmse


# ==============================
# RUN STUDY
# ==============================

if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=60)

    print("\nBest trial:")
    print("RMSE:", study.best_value)
    print("Params:", study.best_params)

    # Save best params
    Path("models").mkdir(exist_ok=True)
    pd.Series(study.best_params).to_json("models/best_xgb_params.json")
