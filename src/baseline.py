
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
 
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
 
RANDOM_STATE = 42
TEST_SIZE = 0.2
 

df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH.name}: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("Columns:", list(df.columns), "\n")
print(df.head(), "\n")
 

TARGET = None
for c in df.columns:
    if "market" in c.lower() and "value" in c.lower():
        TARGET = c
        break
if TARGET is None:
    for c in df.columns:
        if c.lower() in ("value", "value_eur", "mv", "target"):
            TARGET = c
            break
if TARGET is None:
    raise SystemExit("Could not find the target column. Set TARGET manually.")
print(f"Target column: {TARGET}")
 
# ---------------------------------------------------------------- data quality checks
print("\n--- Data quality checks ---")
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0], "\n")
print("Duplicate rows:", df.duplicated().sum())
 
for col in df.select_dtypes(include="object").columns:
    print(f"\nUnique values in '{col}': {df[col].nunique()}")
    if df[col].nunique() <= 30:
        print(df[col].value_counts().to_string())
 
# Catch glued strings like "LiverpoolGK" in categorical columns
for col in ("Position", "Pos", "position", "Team", "Squad", "team"):
    if col in df.columns:
        glued = df[col].astype(str).str.contains(r"[a-z][A-Z]{2,}$", regex=True)
        if glued.any():
            print(f"\nWARNING: possible concatenated values in '{col}':")
            print(df.loc[glued, col].unique()[:10])
 
print("\nTarget summary:\n", df[TARGET].describe())
if (df[TARGET] <= 0).any():
    print("WARNING: non-positive market values found")
 
# ---------------------------------------------------------------- baseline features
# Baseline = numeric columns only (categoricals are added in Stage 5)
numeric = df.select_dtypes(include=[np.number]).copy()
numeric = numeric.drop(columns=[TARGET])
# Drop ID-like columns if present
numeric = numeric.drop(
    columns=[c for c in numeric.columns if c.lower() in ("id", "player_id", "index", "unnamed: 0")],
    errors="ignore",
)
numeric = numeric.fillna(numeric.median())
 
X = numeric
y = df[TARGET]
print(f"\nBaseline features ({X.shape[1]}): {list(X.columns)}")
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
 
 
# ---------------------------------------------------------------- metrics helper
def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
 
 
# ---------------------------------------------------------------- model 1: raw target
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
pred = np.clip(pred, 0, None)  # market value can't be negative
raw_metrics = evaluate(y_test, pred)
 
# ---------------------------------------------------------------- model 2: log target
# Market values are heavily right-skewed, so predicting log(value) usually helps.
log_model = LinearRegression().fit(X_train, np.log1p(y_train))
log_pred = np.expm1(log_model.predict(X_test))
log_metrics = evaluate(y_test, log_pred)
 
print("\n=== Baseline results (held-out players) ===")
print(f"{'Model':<32}{'MAE':>14}{'RMSE':>14}{'R2':>8}")
print(f"{'LinearRegression (raw target)':<32}{raw_metrics['MAE']:>14,.0f}{raw_metrics['RMSE']:>14,.0f}{raw_metrics['R2']:>8.3f}")
print(f"{'LinearRegression (log target)':<32}{log_metrics['MAE']:>14,.0f}{log_metrics['RMSE']:>14,.0f}{log_metrics['R2']:>8.3f}")
 
# Naive reference: always predict the training mean
naive = evaluate(y_test, np.full(len(y_test), y_train.mean()))
print(f"{'Mean predictor (reference)':<32}{naive['MAE']:>14,.0f}{naive['RMSE']:>14,.0f}{naive['R2']:>8.3f}")
 
# ---------------------------------------------------------------- coefficients
coef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nLargest coefficients (raw-target model):\n", coef.head(10).to_string())
 
# ---------------------------------------------------------------- save
with open(MODELS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(
        {
            "target": TARGET,
            "features": list(X.columns),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "linear_regression_raw": raw_metrics,
            "linear_regression_log": log_metrics,
            "mean_predictor": naive,
        },
        f,
        indent=2,
    )
print(f"\nSaved metrics to {MODELS_DIR / 'baseline_metrics.json'}")
 

