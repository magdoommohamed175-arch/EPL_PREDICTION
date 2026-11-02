"""
Step 6 - Error analysis and feature importance for the saved model.

Run from the project1 folder (after train.py):
    python src/evaluate.py

Outputs:
  reports/oof_predictions.csv   out-of-fold prediction for every player
  reports/figures/10..14_*.png  evaluation charts
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from features import CAT_FEATURES, NUM_FEATURES, TARGET, add_features

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print(f"saved {name}")


df = add_features(pd.read_csv(DATA_PATH))
X, y = df[ALL_FEATURES], df[TARGET]
model = joblib.load(MODEL_PATH)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------------------------------------------------ out-of-fold predictions
oof = cross_val_predict(clone(model), X, y, cv=cv)
res = df[["Player", "Team", "Primary_Position", "Age", "Minutes"]].copy()
res["Actual_EUR"] = y
res["Predicted_EUR"] = oof.round(0)
res["Error_EUR"] = res["Predicted_EUR"] - res["Actual_EUR"]  # + = model says worth MORE than listed
res["Pct_Error"] = res["Error_EUR"] / res["Actual_EUR"] * 100
res["Age_Band"] = pd.cut(res["Age"], bins=[15, 21, 24, 27, 30, 33, 45],
                         labels=["<=21", "22-24", "25-27", "28-30", "31-33", "34+"])
res.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)

print("=== Out-of-fold performance (all 340 players, each predicted by a model that never saw them) ===")
print(f"MAE   : {mean_absolute_error(y, oof):,.0f}")
print(f"RMSE  : {np.sqrt(mean_squared_error(y, oof)):,.0f}")
print(f"R2    : {r2_score(y, oof):.3f}")
print(f"Median absolute % error: {res['Pct_Error'].abs().median():.0f}%\n")

# ------------------------------------------------------------------ charts
res["Actual_M"] = res["Actual_EUR"] / 1e6
res["Pred_M"] = res["Predicted_EUR"] / 1e6
res["Error_M"] = res["Error_EUR"] / 1e6

plt.figure(figsize=(6, 6))
sns.scatterplot(data=res, x="Actual_M", y="Pred_M", hue="Primary_Position", alpha=0.7)
lim = max(res["Actual_M"].max(), res["Pred_M"].max()) * 1.05
plt.plot([0, lim], [0, lim], "k--", lw=1)
plt.title("Actual vs predicted value (out-of-fold)")
plt.xlabel("Actual (M EUR)")
plt.ylabel("Predicted (M EUR)")
save("10_actual_vs_predicted.png")

plt.figure(figsize=(7, 4))
sns.histplot(res["Error_M"], bins=40)
plt.axvline(0, color="k", lw=1)
plt.title("Prediction error (predicted - actual, M EUR)")
save("11_error_distribution.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=res, x="Primary_Position", y="Error_M", ax=ax[0])
ax[0].axhline(0, color="k", lw=1)
ax[0].set_title("Error by position")
sns.boxplot(data=res, x="Age_Band", y="Error_M", ax=ax[1])
ax[1].axhline(0, color="k", lw=1)
ax[1].set_title("Error by age band")
save("12_error_by_group.png")

team_mae = res.assign(Abs=res["Error_M"].abs()).groupby("Team")["Abs"].mean().sort_values()
plt.figure(figsize=(7, 6))
team_mae.plot(kind="barh")
plt.title("Mean absolute error by team (M EUR)")
save("13_error_by_team.png")

# ------------------------------------------------------------------ permutation importance (held-out folds)
imp = pd.Series(0.0, index=ALL_FEATURES)
for tr, va in cv.split(X):
    m = clone(model).fit(X.iloc[tr], y.iloc[tr])
    r = permutation_importance(m, X.iloc[va], y.iloc[va],
                               scoring="neg_mean_absolute_error", n_repeats=10, random_state=SEED)
    imp += pd.Series(r.importances_mean, index=ALL_FEATURES) / cv.get_n_splits()
imp = imp.sort_values(ascending=False)
plt.figure(figsize=(7, 6))
imp.iloc[::-1].plot(kind="barh")
plt.title("Permutation importance (increase in MAE when shuffled, EUR)")
save("14_feature_importance.png")

# ------------------------------------------------------------------ printed summaries
print("Permutation importance (EUR of MAE lost when the feature is shuffled):")
print(imp.round(0).to_string(float_format=lambda v: f"{v:,.0f}"), "\n")

print("Error by position:")
print(res.groupby("Primary_Position")["Error_M"].agg(["count", "mean", lambda s: s.abs().mean()])
      .rename(columns={"mean": "bias_M", "<lambda_0>": "MAE_M"}).round(1).to_string(), "\n")
print("Error by age band:")
print(res.groupby("Age_Band", observed=True)["Error_M"].agg(["count", "mean", lambda s: s.abs().mean()])
      .rename(columns={"mean": "bias_M", "<lambda_0>": "MAE_M"}).round(1).to_string(), "\n")

cols = ["Player", "Team", "Primary_Position", "Age", "Minutes", "Actual_M", "Pred_M"]
print("Model says WORTH MORE than listed (candidate 'undervalued'):")
print(res.nlargest(10, "Error_M")[cols].round(1).to_string(index=False), "\n")
print("Model says worth LESS than listed (candidate 'overvalued'):")
print(res.nsmallest(10, "Error_M")[cols].round(1).to_string(index=False))
print(f"\nSaved {REPORTS_DIR / 'oof_predictions.csv'}")
