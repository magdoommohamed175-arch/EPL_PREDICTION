"""
Step 8 - Reusable prediction function.

    from predict import predict_value
    predict_value(team="Arsenal", position="MF", age=24, minutes=300,
                  goals=1, assists=1, xg=0.4, xa=0.5)

Quick test from the project1 folder:
    python src/predict.py

The uncertainty range comes from the out-of-fold errors saved by evaluate.py
(reports/oof_predictions.csv): the 10th-90th percentile of actual/predicted.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from features import CAT_FEATURES, NUM_FEATURES, add_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
OOF_PATH = ROOT / "reports" / "oof_predictions.csv"
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
VALID_POSITIONS = ["GK", "DF", "MF", "FW"]


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _ratio_quantiles():
    """10th and 90th percentile of actual/predicted from cross-validation, or None."""
    if not OOF_PATH.exists():
        return None
    oof = pd.read_csv(OOF_PATH)
    ratio = oof["Actual_EUR"] / oof["Predicted_EUR"].clip(lower=1)
    return float(ratio.quantile(0.10)), float(ratio.quantile(0.90))


def available_teams():
    """Team names as the model knows them (for dropdowns)."""
    teams = pd.read_csv(DATA_PATH, usecols=["Team"])["Team"]
    return sorted(teams.str.replace("-", " ", regex=False).unique())


def predict_dataframe(df: pd.DataFrame) -> pd.Series:
    """df needs columns: Team, Position, Age, Minutes, Goals, Assists, xG, xA."""
    feats = add_features(df)[ALL_FEATURES]
    return pd.Series(_load_model().predict(feats), index=df.index, name="Predicted_EUR")


def predict_value(team, position, age, minutes, goals=0, assists=0, xg=0.0, xa=0.0,
                  with_interval=True) -> dict:
    """Predict one player's market value in euros."""
    if position not in VALID_POSITIONS:
        raise ValueError(f"position must be one of {VALID_POSITIONS}")
    row = pd.DataFrame([{
        "Team": team, "Position": position, "Age": float(age), "Minutes": float(minutes),
        "Goals": float(goals), "Assists": float(assists), "xG": float(xg), "xA": float(xa),
    }])
    pred = float(predict_dataframe(row).iloc[0])
    out = {"predicted_value_eur": pred}
    q = _ratio_quantiles() if with_interval else None
    if q:
        out["low_eur"], out["high_eur"] = pred * q[0], pred * q[1]
    return out


if __name__ == "__main__":
    print("Teams:", available_teams()[:3], "...")
    r = predict_value("Arsenal", "MF", age=24, minutes=300, goals=1, assists=1, xg=0.4, xa=0.5)
    print({k: f"{v:,.0f}" for k, v in r.items()})
    r = predict_value("Burnley", "DF", age=31, minutes=360)
    print({k: f"{v:,.0f}" for k, v in r.items()})
