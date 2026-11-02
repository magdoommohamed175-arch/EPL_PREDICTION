"""
Stage 5 - Feature engineering.

Run from the project1 folder:
    python src/features.py

`add_features(df)` is reused later by train.py, predict.py and app.py so the
same transformations are applied everywhere.
Running this file compares the baseline against the improved features using
the SAME split as baseline.py (test_size=0.2, random_state=42), plus 5-fold CV
because a 68-player test set is noisy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"
OUT_DATA = ROOT / "data" / "ml_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "Market_Value_EUR"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_DENOM = 90  # per-90 stats use at least 90 minutes, so tiny samples don't explode

BASE_NUM = ["Age", "Minutes", "Goals", "Assists", "xG", "xA"]
NUM_FEATURES = [
    "Age", "Age_over_27", "Log_Minutes",
    "Goals", "Assists", "xG", "xA",
    "Goals_p90", "Assists_p90", "xG_p90", "xA_p90",
    "GA_p90", "xGI_p90", "Goals_minus_xG",
]
CAT_FEATURES = ["Team", "Primary_Position"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (no fitting, so no data leakage)."""
    df = df.copy()
    df["Team"] = df["Team"].str.replace("-", " ", regex=False)
    df["Primary_Position"] = df["Position"].str.split(",").str[0]

    # Age curve: value is roughly flat until ~27, then falls quickly (see age-band chart)
    df["Age_over_27"] = (df["Age"] - 27).clip(lower=0)

    df["Log_Minutes"] = np.log1p(df["Minutes"])

    denom = df["Minutes"].clip(lower=MIN_DENOM)
    for col in ["Goals", "Assists", "xG", "xA"]:
        df[f"{col}_p90"] = df[col] / denom * 90
    df["GA_p90"] = df["Goals_p90"] + df["Assists_p90"]
    df["xGI_p90"] = df["xG_p90"] + df["xA_p90"]
    df["Goals_minus_xG"] = df["Goals"] - df["xG"]  # finishing over/under-performance
    return df


def make_pipeline(num_cols, cat_cols, regressor):
    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", regressor)])


def holdout_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


if __name__ == "__main__":
    raw = pd.read_csv(DATA_PATH)
    df = add_features(raw)
    df.to_csv(OUT_DATA, index=False)
    print(f"Saved engineered dataset to {OUT_DATA}  ({df.shape[1]} columns)\n")

    y = df[TARGET]
    all_cols = NUM_FEATURES + CAT_FEATURES
    X = df[all_cols + [c for c in BASE_NUM if c not in all_cols]]

    ridge_alphas = np.logspace(-2, 3, 30)
    models = {
        "Baseline LinReg (6 numeric)": make_pipeline(BASE_NUM, [], LinearRegression()),
        "LinReg + new features": make_pipeline(NUM_FEATURES, CAT_FEATURES, LinearRegression()),
        "Ridge + new features": make_pipeline(NUM_FEATURES, CAT_FEATURES, RidgeCV(alphas=ridge_alphas)),
        "Ridge + new features (log y)": TransformedTargetRegressor(
            regressor=make_pipeline(NUM_FEATURES, CAT_FEATURES, RidgeCV(alphas=ridge_alphas)),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    print(f"{'Model':<32}{'Test MAE':>12}{'Test R2':>9}{'CV MAE':>12}{'CV R2':>16}")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, None)
        m = holdout_metrics(y_test, pred)

        cvres = cross_validate(
            model, X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error")
        )
        m["CV_R2_mean"] = float(cvres["test_r2"].mean())
        m["CV_R2_std"] = float(cvres["test_r2"].std())
        m["CV_MAE_mean"] = float(-cvres["test_neg_mean_absolute_error"].mean())
        results[name] = m

        print(
            f"{name:<32}{m['MAE']:>12,.0f}{m['R2']:>9.3f}"
            f"{m['CV_MAE_mean']:>12,.0f}{m['CV_R2_mean']:>9.3f} +/-{m['CV_R2_std']:.2f}"
        )

    with open(MODELS_DIR / "feature_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {MODELS_DIR / 'feature_comparison.json'}")
