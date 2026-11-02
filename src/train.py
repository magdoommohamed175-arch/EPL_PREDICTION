"""
Stages 6-7 - Compare algorithms, tune the strongest, and save the pipeline.

Run from the project1 folder:
    python src/train.py

Method (so the final test score is honest):
  1. Hold out 20% of players as a test set (same split as before).
  2. Compare all models and tune the top ones using 5-fold CV on the TRAIN part only.
  3. Pick the model with the best CV MAE and score it on the test set ONCE.
  4. Refit that model on all 340 players and save it with joblib.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CAT_FEATURES, NUM_FEATURES, TARGET, add_features

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

SEED = 42
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def make_pipeline(reg, scale):
    prep = ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ]
    )
    return Pipeline([("prep", prep), ("model", reg)])


def build(reg, log_y, scale):
    pipe = make_pipeline(clone(reg), scale)
    if log_y:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def param_prefix(log_y):
    return "regressor__model__" if log_y else "model__"


CANDIDATES = {
    "Ridge (log y)": dict(
        reg=RidgeCV(alphas=np.logspace(-2, 3, 30)), log=True, scale=True, grid={}
    ),
    "RandomForest (log y)": dict(
        reg=RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        log=True, scale=False,
        grid={"max_depth": [None, 6, 10], "min_samples_leaf": [1, 3, 5], "max_features": [0.3, 0.5, 1.0]},
    ),
    "RandomForest (raw y)": dict(
        reg=RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        log=False, scale=False,
        grid={"max_depth": [None, 6, 10], "min_samples_leaf": [1, 3, 5], "max_features": [0.3, 0.5, 1.0]},
    ),
    "HistGradientBoosting (log y)": dict(
        reg=HistGradientBoostingRegressor(random_state=SEED),
        log=True, scale=False,
        grid={"learning_rate": [0.03, 0.05, 0.1], "max_iter": [100, 200, 400], "max_depth": [2, 3, 4],
              "min_samples_leaf": [5, 10, 20], "l2_regularization": [0.0, 1.0, 5.0]},
    ),
    "HistGradientBoosting (raw y)": dict(
        reg=HistGradientBoostingRegressor(random_state=SEED),
        log=False, scale=False,
        grid={"learning_rate": [0.03, 0.05, 0.1], "max_iter": [100, 200, 400], "max_depth": [2, 3, 4],
              "min_samples_leaf": [5, 10, 20], "l2_regularization": [0.0, 1.0, 5.0]},
    ),
    "GradientBoosting (log y)": dict(
        reg=GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=SEED),
        log=True, scale=False,
        grid={"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4],
              "subsample": [0.7, 1.0], "min_samples_leaf": [1, 3, 5]},
    ),
}


def test_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


if __name__ == "__main__":
    df = add_features(pd.read_csv(DATA_PATH))
    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Train: {len(X_train)} players | Test: {len(X_test)} players\n")

    # ---------------------------------------------------------- 1. compare (default settings)
    print("=== Step 1: compare models (5-fold CV on training players) ===")
    rows = []
    for name, c in CANDIDATES.items():
        est = build(c["reg"], c["log"], c["scale"])
        res = cross_validate(est, X_train, y_train, cv=cv,
                             scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"))
        rows.append({
            "Model": name,
            "CV MAE": -res["test_neg_mean_absolute_error"].mean(),
            "CV RMSE": -res["test_neg_root_mean_squared_error"].mean(),
            "CV R2": res["test_r2"].mean(),
            "CV R2 std": res["test_r2"].std(),
        })
    comp = pd.DataFrame(rows).sort_values("CV MAE").reset_index(drop=True)
    print(comp.to_string(index=False, formatters={
        "CV MAE": "{:,.0f}".format, "CV RMSE": "{:,.0f}".format,
        "CV R2": "{:.3f}".format, "CV R2 std": "{:.3f}".format}))
    comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # ---------------------------------------------------------- 2. tune the top 2 tunable models
    print("\n=== Step 2: tune the two best tunable models ===")
    final_pool = {r["Model"]: (build(CANDIDATES[r["Model"]]["reg"], CANDIDATES[r["Model"]]["log"], CANDIDATES[r["Model"]]["scale"]), r["CV MAE"])
                  for _, r in comp.iterrows()}
    tunable = [m for m in comp["Model"] if CANDIDATES[m]["grid"]][:2]
    for name in tunable:
        c = CANDIDATES[name]
        prefix = param_prefix(c["log"])
        search = RandomizedSearchCV(
            build(c["reg"], c["log"], c["scale"]),
            param_distributions={prefix + k: v for k, v in c["grid"].items()},
            n_iter=20, cv=cv, scoring="neg_mean_absolute_error",
            random_state=SEED, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_mae = -search.best_score_
        print(f"{name}: tuned CV MAE {tuned_mae:,.0f}  | params: "
              f"{ {k.replace(prefix, ''): v for k, v in search.best_params_.items()} }")
        final_pool[f"{name} [tuned]"] = (search.best_estimator_, tuned_mae)

    # ---------------------------------------------------------- 3. pick best, test once
    best_name, (best_est, best_cv_mae) = min(final_pool.items(), key=lambda kv: kv[1][1])
    print(f"\n=== Step 3: selected model: {best_name} (CV MAE {best_cv_mae:,.0f}) ===")
    model = clone(best_est).fit(X_train, y_train)
    pred = model.predict(X_test)
    m = test_metrics(y_test, pred)
    print(f"Held-out test  MAE {m['MAE']:,.0f} | RMSE {m['RMSE']:,.0f} | R2 {m['R2']:.3f}")

    out = df.loc[X_test.index, ["Player", "Team", "Primary_Position", "Age", "Minutes"]].copy()
    out["Actual_EUR"] = y_test
    out["Predicted_EUR"] = pred.round(0)
    out["Error_EUR"] = out["Predicted_EUR"] - out["Actual_EUR"]
    out.sort_values("Error_EUR", key=abs, ascending=False).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    print("Biggest test misses:\n", out.reindex(out["Error_EUR"].abs().sort_values(ascending=False).index).head(8)
          [["Player", "Team", "Actual_EUR", "Predicted_EUR"]].to_string(index=False))

    # ---------------------------------------------------------- 4. refit on all data and save
    final_model = clone(best_est).fit(X, y)
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib")
    meta = {
        "model": best_name,
        "features": ALL_FEATURES,
        "target": TARGET,
        "cv_mae_train": best_cv_mae,
        "test_metrics": m,
        "note": "Test metrics come from a model fit on 80% of players; the saved model is refit on all players.",
    }
    with open(MODELS_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved pipeline to {MODELS_DIR / 'best_model.joblib'}")
