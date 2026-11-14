from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from quantlab.data import validate_matches_csv, recent_team_stats
from quantlab.probability import calculate_probabilities, estimate_expected_goals, top_correct_scores
from quantlab.market import evaluate_value, no_vig_probabilities, overround
from quantlab.storage import initialize, load_matches, save_dataset, save_prediction, list_predictions
from quantlab.risk import flat_stake

st.set_page_config(page_title="Football Quant Lab", page_icon="⚽", layout="wide")
DB_PATH = Path("data/football_quant_lab.sqlite3")
initialize(DB_PATH)

st.markdown("""
<style>
.stApp {background: #0b1220; color: #e5e7eb;}
[data-testid="stMetric"] {background: #111c2e; padding: 14px; border-radius: 10px;}
[data-testid="stSidebar"] {background: #101a2b;}
</style>
""", unsafe_allow_html=True)

st.title("⚽ FOOTBALL QUANT LAB")
st.caption("Quantitative football analysis • Probability • Market intelligence • Risk control")
page = st.sidebar.radio("NAVIGATION", ["Dashboard Overview", "Match Analysis", "Probability Explorer",
    "Odds Intelligence", "Value Bet Scanner", "Risk Management", "Prediction Tracker",
    "Model Performance", "CSV Data Management", "Settings"])

matches = load_matches(DB_PATH)

if page == "Dashboard Overview":
    st.subheader("Dashboard Overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Historical matches", len(matches))
    preds = list_predictions(DB_PATH)
    c2.metric("Saved predictions", len(preds))
    c3.metric("Settled predictions", int(preds.actual_home_goals.notna().sum()) if not preds.empty else 0)
    st.info("Import historical match data from CSV to activate data-driven team analysis. No sample or synthetic match data is bundled.")
    if not matches.empty:
        st.dataframe(matches.tail(20), use_container_width=True)

elif page == "CSV Data Management":
    st.subheader("CSV Data Management")
    st.write("Required columns: date, home_team, away_team, home_goals, away_goals. Optional: match_id, xG fields.")
    uploaded = st.file_uploader("Upload match history CSV", type=["csv"])
    if uploaded:
        raw = uploaded.getvalue()
        try:
            df, errors = validate_matches_csv(uploaded)
            if errors:
                for error in errors: st.error(error)
            else:
                st.success(f"Validated {len(df)} rows.")
                st.dataframe(df.head(10), use_container_width=True)
                if st.button("Save dataset permanently"):
                    dataset_id = save_dataset(df, uploaded.name, hashlib.sha256(raw).hexdigest(), DB_PATH)
                    st.success(f"Dataset saved. Dataset ID: {dataset_id}")
        except Exception as exc:
            st.error(f"CSV could not be read: {exc}")
    st.caption("SQLite persists imported rows and dataset fingerprints between app sessions. Keep a separate backup of the data directory.")
    if st.button("Download database backup"):
        if DB_PATH.exists():
            st.download_button("Download SQLite database", DB_PATH.read_bytes(), "football_quant_lab_backup.sqlite3")

elif page in ("Match Analysis", "Probability Explorer", "Odds Intelligence", "Value Bet Scanner"):
    st.subheader(page)
    teams = sorted(set(matches.home_team.astype(str)) | set(matches.away_team.astype(str))) if not matches.empty else []
    if len(teams) < 2:
        st.warning("Import a valid historical CSV containing at least two teams before analysis.")
    else:
        c1,c2 = st.columns(2)
        home = c1.selectbox("Home team", teams)
        away = c2.selectbox("Away team", [t for t in teams if t != home])
        if st.button("Calculate probabilities", type="primary"):
            try:
                hs, aws = recent_team_stats(matches, home), recent_team_stats(matches, away)
                league_home = float(matches.home_goals.mean())
                league_away = float(matches.away_goals.mean())
                lh, la = estimate_expected_goals(hs["scored_per_match"], hs["conceded_per_match"],
                    aws["scored_per_match"], aws["conceded_per_match"], league_home, league_away)
                p = calculate_probabilities(lh, la)
                st.session_state["last_analysis"] = {"home":home,"away":away,"lh":lh,"la":la,"p":p}
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.get("last_analysis")
        if result and result["home"] == home and result["away"] == away:
            p = result["p"]
            st.write(f"Expected goals: **{home} {result['lh']:.2f} — {result['la']:.2f} {away}**")
            st.caption("Lambdas are estimated from historical goals, not xG. Use an xG-enriched dataset and a calibrated model before treating estimates as production-grade.")
            a,b,c = st.columns(3)
            a.metric("Home win", f"{p.home_win:.1%}")
            b.metric("Draw", f"{p.draw:.1%}")
            c.metric("Away win", f"{p.away_win:.1%}")
            d,e,f = st.columns(3)
            d.metric("Over 2.5", f"{p.over_25:.1%}")
            e.metric("Under 2.5", f"{p.under_25:.1%}")
            f.metric("BTTS Yes", f"{p.btts_yes:.1%}")
            st.write("Most likely correct scores")
            st.dataframe(pd.DataFrame(top_correct_scores(p.score_matrix)), use_container_width=True)
            if page in ("Odds Intelligence", "Value Bet Scanner"):
                st.markdown("#### 1X2 bookmaker odds")
                oh,od,oa = st.columns(3)
                h_odds = oh.number_input("Home odds", min_value=1.01, value=2.0, step=0.01)
                d_odds = od.number_input("Draw odds", min_value=1.01, value=3.3, step=0.01)
                a_odds = oa.number_input("Away odds", min_value=1.01, value=4.0, step=0.01)
                odds = {"Home":h_odds,"Draw":d_odds,"Away":a_odds}
                nv = no_vig_probabilities(odds)
                st.write(f"Bookmaker overround: **{overround(odds):.2%}**")
                st.dataframe(pd.DataFrame({"Odds":odds,"No-vig probability":nv}), use_container_width=True)
                probs = {"Home":p.home_win,"Draw":p.draw,"Away":p.away_win}
                values = {k:evaluate_value(probs[k],odds[k],uncertainty=0.03) for k in odds}
                st.dataframe(pd.DataFrame(values).T, use_container_width=True)
                st.warning("Illustrative uncertainty margin only. This model is not calibrated yet; value status is provisional and should not be treated as a betting signal.")
                if st.button("Save pre-match prediction"):
                    st.error("Enter a real match ID and kickoff timestamp in the full tracker workflow before saving. This simplified screen does not create incomplete prediction records.")

elif page == "Risk Management":
    st.subheader("Risk Management")
    bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0)
    fraction = st.slider("Flat stake fraction", 0.25, 5.0, 1.0, 0.25) / 100
    st.metric("Suggested flat stake", f"{flat_stake(bankroll,fraction):,.2f}")
    st.caption("Conservative flat staking example. No martingale or loss-chasing.")

elif page == "Prediction Tracker":
    st.subheader("Prediction Tracker")
    preds = list_predictions(DB_PATH)
    st.dataframe(preds, use_container_width=True) if not preds.empty else st.info("No predictions saved yet.")

elif page == "Model Performance":
    st.subheader("Model Performance")
    st.info("Performance metrics will be activated as soon as settled predictions with stake and outcome records are available. Win rate alone is not a model-quality measure.")

elif page == "Settings":
    st.subheader("Settings")
    st.write("Model version: 1.0.0")
    st.write(f"Database: {DB_PATH}")
    st.write("Data source: user-provided CSV. No live feed or fabricated statistics are used.")
