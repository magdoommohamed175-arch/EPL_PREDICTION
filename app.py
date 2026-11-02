













































"""
Steps 9-10 - Streamlit app: value prediction + scouting view.
 
Run from the project1 folder (this file lives in the project root, not src/):
    pip install streamlit
    streamlit run app.py
"""
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import streamlit as st
 
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
 
from predict import _ratio_quantiles, available_teams, predict_value  # noqa: E402
 
OOF_PATH = ROOT / "reports" / "oof_predictions.csv"
DATA_PATH = ROOT / "data" / "ml_dataset_clean.csv"
 
st.set_page_config(page_title="Premier League Market Value Predictor", page_icon="⚽", layout="wide")
 
 
def eur_m(v):
    return f"€{v / 1e6:,.1f}M"
 
 
@st.cache_data
def load_scouting():
    oof = pd.read_csv(OOF_PATH)
    stats = pd.read_csv(DATA_PATH)[["Player", "Goals", "Assists", "xG", "xA"]]
    return oof.merge(stats, on="Player", how="left")
 
 
st.title("⚽ Premier League Market Value Predictor")
st.caption(
    "Machine-learning estimate of a player's Transfermarkt market value from age, club, "
    "position, minutes and early-season attacking stats."
)
 
tab_predict, tab_scout, tab_about = st.tabs(["Predict a value", "Scouting: model vs listed value", "About & limits"])
 
# ---------------------------------------------------------------- tab 1: predict
with tab_predict:
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        team = c1.selectbox("Team", available_teams())
        position = c1.selectbox("Position", ["GK", "DF", "MF", "FW"], index=2)
        age = c2.slider("Age", 16, 40, 24)
        minutes = c2.slider("Minutes played (data covers about 4 games, max 360)", 1, 360, 270)
        goals = c3.number_input("Goals", 0, 20, 0)
        assists = c3.number_input("Assists", 0, 20, 0)
        xg = c3.number_input("xG", 0.0, 20.0, 0.0, step=0.1)
        xa = c3.number_input("xA", 0.0, 20.0, 0.0, step=0.1)
        go = st.form_submit_button("Predict market value", type="primary")
 
    if go:
        try:
            r = predict_value(team, position, age, minutes, goals, assists, xg, xa)
        except FileNotFoundError:
            st.error("Model file not found. Run `python src/train.py` first.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted value", eur_m(r["predicted_value_eur"]))
            if "low_eur" in r:
                m2.metric("Typical low", eur_m(r["low_eur"]))
                m3.metric("Typical high", eur_m(r["high_eur"]))
                st.caption(
                    "The low–high range is where about 80% of players' listed values fell relative to "
                    "the model's prediction in cross-validation. Treat the model as a tier estimate, "
                    "not an exact price."
                )
 
# ---------------------------------------------------------------- tab 2: scouting
with tab_scout:
    if not OOF_PATH.exists():
        st.error("Run `python src/evaluate.py` first to create the out-of-fold predictions.")
    else:
        df = load_scouting()
        df["Gap_M"] = (df["Predicted_EUR"] - df["Actual_EUR"]) / 1e6
        df["Gap_Pct"] = df["Gap_M"] * 1e6 / df["Actual_EUR"] * 100
        q = _ratio_quantiles()
 
        st.markdown(
            "Each prediction below comes from a model that **never saw that player**. "
            "A positive gap means the model values the player higher than the listed value."
        )
        f1, f2, f3, f4 = st.columns(4)
        direction = f1.radio("Show", ["Model higher than listed", "Model lower than listed"])
        min_minutes = f2.slider("Minimum minutes", 0, 360, 180)
        positions = f3.multiselect("Positions", ["GK", "DF", "MF", "FW"], default=["GK", "DF", "MF", "FW"])
        age_lo, age_hi = f4.slider("Age range", 15, 40, (16, 40))
        teams = st.multiselect("Teams (empty = all)", sorted(df["Team"].unique()))
        top_n = st.slider("Number of players", 5, 50, 15)
 
        d = df[
            (df["Minutes"] >= min_minutes)
            & (df["Primary_Position"].isin(positions))
            & (df["Age"].between(age_lo, age_hi))
        ]
        if teams:
            d = d[d["Team"].isin(teams)]
 
        ratio = d["Actual_EUR"] / d["Predicted_EUR"].clip(lower=1)
        if direction == "Model higher than listed":
            d = d.assign(Signal=np.where(ratio < q[0], "Strong", "Weak")).sort_values("Gap_M", ascending=False)
        else:
            d = d.assign(Signal=np.where(ratio > q[1], "Strong", "Weak")).sort_values("Gap_M", ascending=True)
 
        show = d.head(top_n)[[
            "Player", "Team", "Primary_Position", "Age", "Minutes", "Goals", "Assists", "xG", "xA",
            "Actual_EUR", "Predicted_EUR", "Gap_M", "Gap_Pct", "Signal",
        ]].rename(columns={"Primary_Position": "Pos", "Actual_EUR": "Listed (M)", "Predicted_EUR": "Model (M)",
                           "Gap_M": "Gap (M)", "Gap_Pct": "Gap %"})
        show["Listed (M)"] = (show["Listed (M)"] / 1e6).round(1)
        show["Model (M)"] = (show["Model (M)"] / 1e6).round(1)
        show["Gap (M)"] = show["Gap (M)"].round(1)
        show["Gap %"] = show["Gap %"].round(0)
 
        st.dataframe(show, hide_index=True, use_container_width=True)
        if len(show):
            st.bar_chart(show.set_index("Player")["Gap (M)"])
        st.caption(
            "Signal = 'Strong' when the listed value is outside the typical 80% error band. "
            "'Weak' gaps are within normal model error and should not be treated as findings."
        )
        st.warning(
            "This is a screening aid, not a transfer recommendation. Predictions lean heavily on club "
            "and age, and the data can't see reputation, contract length or injuries. Players from big "
            "clubs are often flagged as 'higher than listed' because of the club premium."
        )
 
with tab_about:
    st.markdown(
        "**Model:** Ridge regression on log(market value) with one-hot Team and Position, "
        "an age curve (flat to 27, then falling), playing time and per-90 attacking stats."
    )
    st.markdown(
        "**Accuracy (cross-validated, about 340 players):** R² ≈ 0.59, "
        "mean error ≈ €10M, median error ≈ 30%."
    )
    st.markdown(
        "**Limits**\n\n"
        "- Stats cover only about four games per player, so performance numbers add little; "
        "age, club and playing time drive most predictions.\n"
        "- Market values are current Transfermarkt values, not values at the time of the stats.\n"
        "- The model can't see reputation, contract status, injuries or international record.\n"
        "- Some player positions in the source data may be mislabelled."
    )