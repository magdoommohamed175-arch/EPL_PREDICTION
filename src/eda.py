
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Market_Value_EUR"
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print(f"saved {name}")


# ------------------------------------------------------------------ load + tidy
df = pd.read_csv(DATA_PATH)
df["Team"] = df["Team"].str.replace("-", " ", regex=False)  # "Leeds-United" -> "Leeds United"
df["Primary_Position"] = df["Position"].str.split(",").str[0]  # "FW,MF" -> "FW"
df["Value_M"] = df[TARGET] / 1e6  # millions of euros, easier to read
df["Log_Value"] = np.log1p(df[TARGET])

print(df.describe().T.to_string(), "\n")
print("Players under 90 minutes:", (df["Minutes"] < 90).sum())
print("Players with 0 minutes:  ", (df["Minutes"] == 0).sum(), "\n")

# ------------------------------------------------------------------ 1. target distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Value_M"], bins=30, ax=ax[0])
ax[0].set(title="Market value (EUR millions)", xlabel="Value (M EUR)")
sns.histplot(df["Log_Value"], bins=30, ax=ax[1])
ax[1].set(title="log(1 + market value)", xlabel="log value")
save("01_value_distribution.png")
print("Skewness of raw value:", round(df[TARGET].skew(), 2), "| log value:", round(df["Log_Value"].skew(), 2))

# ------------------------------------------------------------------ 2. value by position
order = df.groupby("Primary_Position")["Value_M"].median().sort_values(ascending=False).index
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Primary_Position", y="Value_M", order=order)
plt.title("Market value by primary position")
plt.ylabel("Value (M EUR)")
save("02_value_by_position.png")
print("\nMedian value by position (M EUR):\n", df.groupby("Primary_Position")["Value_M"].agg(["count", "median", "mean"]).round(1))

# ------------------------------------------------------------------ 3. value by team
order = df.groupby("Team")["Value_M"].median().sort_values(ascending=False).index
plt.figure(figsize=(9, 7))
sns.boxplot(data=df, y="Team", x="Value_M", order=order)
plt.title("Market value by team")
plt.xlabel("Value (M EUR)")
save("03_value_by_team.png")

# ------------------------------------------------------------------ 4. numeric features vs value
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ["Age", "Minutes", "Goals", "Assists", "xG", "xA"]):
    sns.scatterplot(data=df, x=col, y="Value_M", hue="Primary_Position", alpha=0.7, ax=ax, legend=(col == "Age"))
    ax.set(title=f"{col} vs value", ylabel="Value (M EUR)")
save("04_features_vs_value.png")

# ------------------------------------------------------------------ 5. correlation heatmap
num_cols = ["Age", "Minutes", "Goals", "Assists", "xG", "xA", "Value_M"]
plt.figure(figsize=(7, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
save("05_correlation.png")
print("\nCorrelation with value:\n", df[num_cols].corr()["Value_M"].drop("Value_M").sort_values(ascending=False).round(2))
print("Goals vs xG correlation:", round(df["Goals"].corr(df["xG"]), 2))

# ------------------------------------------------------------------ 6. age vs value (non-linear?)
df["Age_Band"] = pd.cut(df["Age"], bins=[15, 21, 24, 27, 30, 33, 45], labels=["<=21", "22-24", "25-27", "28-30", "31-33", "34+"])
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Age_Band", y="Value_M")
plt.title("Market value by age band")
plt.ylabel("Value (M EUR)")
save("06_value_by_age_band.png")

# ------------------------------------------------------------------ 7. outliers
q1, q3 = df["Value_M"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
cols = ["Player", "Team", "Primary_Position", "Age", "Minutes", "Goals", "Assists", "Value_M"]
print(f"\nValue outliers above {upper:.1f}M EUR (IQR rule): {(df['Value_M'] > upper).sum()}")
print("\nTop 10 most valuable players:\n", df.nlargest(10, "Value_M")[cols].to_string(index=False))
print("\nHigh value but few minutes (check these rows manually):\n",
      df[(df["Value_M"] > 30) & (df["Minutes"] < df["Minutes"].quantile(0.25))][cols].to_string(index=False))
print("\nLowest 10 values:\n", df.nsmallest(10, "Value_M")[cols].to_string(index=False))
print(f"\nCharts are in: {FIG_DIR}")
