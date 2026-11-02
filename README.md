# Premier League Transfer Market Value Predictor

A machine-learning project that estimates Premier League players' Transfermarkt market values (in euros) from age, club, position, playing time and early-season attacking statistics, with a Streamlit app for predictions and a scouting view that compares the model's estimate with each player's listed value.

![Predict tab](docs/screenshot_predict.png)
![Scouting tab](docs/screenshot_scouting.png)

## Results

Five-fold cross-validation on 340 players (mean of folds):

| Model | R² | MAE (EUR) |
|---|---|---|
| Baseline Linear Regression (6 numeric features) | 0.314 ± 0.11 | 13.2M |
| Linear Regression + engineered features | 0.446 ± 0.07 | 12.2M |
| Ridge + engineered features | 0.463 ± 0.07 | 11.9M |
| **Ridge + engineered features, log target (selected)** | **0.559 ± 0.11** | **10.0M** |

Pooling the out-of-fold predictions for all 340 players (each predicted by a model that never saw them) gives R² = 0.59, MAE = €10.0M, RMSE = €15.2M and a median absolute error of 30%. A single 80/20 split (68 test players) gave R² = 0.73 and MAE = €9.7M, but that split is small and optimistic, so the cross-validated numbers are the ones to trust.

Random Forest, HistGradientBoosting and GradientBoosting were also compared and tuned with randomized search. None beat the regularized linear model on cross-validated MAE with this amount of data.

## What drives the predictions

Permutation importance on held-out folds (increase in MAE when a feature is shuffled):

1. **Age past 27**: about €4.9M. Value is roughly flat until 27, then falls sharply.
2. **Team**: about €4.0M. There is a large club premium.
3. **Playing time (log minutes)**: about €1.8M.

Goals, assists and xG contribute very little. The performance data covers only about four games per player, which is too few to separate players reliably.

## Project structure

```
project1/
├── data/                  datasets (ml_dataset_clean.csv is the modelling input)
├── models/                best_model.joblib + metadata and metrics JSON
├── notebooks/             exploratory notebooks
├── reports/               model comparison, out-of-fold predictions, figures
├── src/
│   ├── collect_data.py    data collection
│   ├── features.py        feature engineering (shared by all scripts)
│   ├── train.py           model comparison, tuning, saving
│   ├── evaluate.py        error analysis, feature importance, out-of-fold predictions
│   └── predict.py         reusable prediction function
├── app.py                 Streamlit app
├── requirements.txt
└── README.md
```

## Setup and usage

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
pip install -r requirements.txt

python src/features.py     # compare feature sets
python src/train.py        # compare, tune and save the best model
python src/evaluate.py     # error analysis and out-of-fold predictions
streamlit run app.py       # launch the app
```

Using the model from Python:

```python
import sys; sys.path.insert(0, "src")
from predict import predict_value

predict_value(team="Arsenal", position="MF", age=24, minutes=300,
              goals=1, assists=1, xg=0.4, xa=0.5)
# {'predicted_value_eur': ..., 'low_eur': ..., 'high_eur': ...}
```

## Methodology

1. **Data**: player performance statistics matched to Transfermarkt market values by player name and club; club names standardized; 340 complete rows.
2. **EDA**: market value is strongly right-skewed; value is flat to about 27 then drops; clubs differ widely; goals and xG (0.76) and assists and xA (0.61) are highly correlated.
3. **Features**: primary position, cleaned team name, an age hinge (`max(age - 27, 0)`), log minutes, per-90 rates with a 90-minute floor so tiny samples don't explode, combined contribution stats and goals minus xG.
4. **Models**: Linear Regression, Ridge, Random Forest, HistGradientBoosting and GradientBoosting, with raw and log targets, all compared with the same folds.
5. **Selection**: models were compared and tuned using cross-validation on the training players only. The test set was scored once, on the selected model. The saved model is refit on all 340 players.
6. **Scouting view**: out-of-fold predictions are compared with listed values. A gap is marked "Strong" only if the listed value falls outside the typical 80% error band from cross-validation.

## Limitations

- **Very little performance data.** Statistics cover about four games per player, so age, club and minutes dominate. This is a prototype, not a reliable valuation tool.
- **Market values may not match the stats' date.** Values are current Transfermarkt figures, not necessarily values at the time the statistics were recorded.
- **Unseen factors.** The model can't see reputation, contract length, injuries or international record. Young stars and big-money signings are consistently under-predicted.
- **Club premium.** Players at the biggest clubs are often flagged as valued higher than listed, partly because the model learns a club effect.
- **Position labels.** Some positions in the source data look wrong (for example, a midfielder labelled as a defender).
- **Prediction ranges are approximate.** They use a fixed ratio from cross-validation errors, applied at every price level.
- **Not financial or transfer advice.** The scouting view is a screening aid only.

## Future work

- Add several full seasons with matching historical market values and use a time-aware train/test split.
- Add previous-season statistics, international caps and contract length.
- Replace the fixed-ratio ranges with quantile or conformal prediction intervals.
- Correct position labels and verify market-value dates against Transfermarkt history.

## Data sources and disclaimer

Performance statistics and market values come from FBref and Transfermarkt. Check their terms of use before redistributing any raw data. This project is independent and not affiliated with the Premier League, FBref or Transfermarkt.
