from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from quantlab.data import validate_matches_csv, recent_team_stats
from quantlab.probability import calculate_probabilities, estimate_expected_goals, top_correct_scores
from quantlab.market import evaluate_value, no_vig_probabilities, overround
from quantlab.asian_handicap import asian_handicap_outcomes, asian_handicap_ev
from quantlab.storage import (initialize, load_matches, save_dataset, save_prediction,
                              list_predictions, settle_prediction, list_datasets)
from quantlab.risk import flat_stake
from quantlab.evaluation import brier_score, log_loss

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
preds = list_predictions(DB_PATH)


def market_probability(market: str, p) -> float:
    return {
        "1X2 Home": p.home_win, "1X2 Draw": p.draw, "1X2 Away": p.away_win,
        "Over 1.5": p.over_15, "Under 1.5": p.under_15,
        "Over 2.5": p.over_25, "Under 2.5": p.under_25,
        "Over 3.5": p.over_35, "Under 3.5": p.under_35,
        "BTTS Yes": p.btts_yes, "BTTS No": p.btts_no,
    }[market]


def market_won(market: str, hg: int, ag: int) -> bool:
    total = hg + ag
    return {
        "1X2 Home": hg > ag, "1X2 Draw": hg == ag, "1X2 Away": hg < ag,
        "Over 1.5": total >= 2, "Under 1.5": total <= 1,
        "Over 2.5": total >= 3, "Under 2.5": total <= 2,
        "Over 3.5": total >= 4, "Under 3.5": total <= 3,
        "BTTS Yes": hg > 0 and ag > 0, "BTTS No": hg == 0 or ag == 0,
    }[market]


if page == "Dashboard Overview":
    st.subheader("Dashboard Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Historical matches", len(matches))
    c2.metric("Saved predictions", len(preds))
    c3.metric("Settled predictions", int(preds.actual_home_goals.notna().sum()) if not preds.empty else 0)
    c4.metric("Datasets", len(list_datasets(DB_PATH)))
    st.info("Import real historical match data from CSV. No sample or synthetic match data is bundled.")
    if not matches.empty:
        st.dataframe(matches.tail(20), use_container_width=True)

elif page == "CSV Data Management":
    st.subheader("CSV Data Management")
    st.write("Required: date, home_team, away_team, home_goals, away_goals. Optional: match_id.")
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
    st.markdown("#### Dataset history")
    st.dataframe(list_datasets(DB_PATH), use_container_width=True)
    st.caption("Imported rows are stored in SQLite. Back up the database before redeploying or moving hosting.")
    if DB_PATH.exists():
        st.download_button("Download SQLite backup", DB_PATH.read_bytes(),
                           "football_quant_lab_backup.sqlite3", mime="application/octet-stream")

elif page in ("Match Analysis", "Probability Explorer", "Odds Intelligence", "Value Bet Scanner"):
    st.subheader(page)
    teams = sorted(set(matches.home_team.astype(str)) | set(matches.away_team.astype(str))) if not matches.empty else []
    if len(teams) < 2:
        st.warning("Import a valid historical CSV containing at least two teams before analysis.")
    else:
        c1,c2 = st.columns(2)
        home = c1.selectbox("Home team", teams)
        away = c2.selectbox("Away team", [t for t in teams if t != home])
        history_limit = st.slider("Recent matches per team", 5, 20, 10)
        if st.button("Calculate probabilities", type="primary"):
            try:
                hs, aws = recent_team_stats(matches, home, history_limit), recent_team_stats(matches, away, history_limit)
                if min(hs["matches"], aws["matches"]) < 5:
                    raise ValueError("Insufficient history: each team needs at least 5 recorded matches.")
                league_home = float(matches.home_goals.mean())
                league_away = float(matches.away_goals.mean())
                lh, la = estimate_expected_goals(hs["scored_per_match"], hs["conceded_per_match"],
                    aws["scored_per_match"], aws["conceded_per_match"], league_home, league_away)
                p = calculate_probabilities(lh, la)
                st.session_state["last_analysis"] = {"home":home,"away":away,"lh":lh,"la":la,"p":p,
                    "home_matches":hs["matches"],"away_matches":aws["matches"]}
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.get("last_analysis")
        if result and result["home"] == home and result["away"] == away:
            p = result["p"]
            st.write(f"Expected goals: **{home} {result['lh']:.2f} — {result['la']:.2f} {away}**")
            st.caption(f"Historical-goals baseline • {int(result['home_matches'])} home-team matches • {int(result['away_matches'])} away-team matches. These are goals-derived lambdas, not xG.")
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
            st.markdown("#### Asian handicap probabilities (home handicap)")
            ah_line = st.selectbox("Home handicap line", [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0], index=8)
            ah = asian_handicap_outcomes(p.score_matrix, ah_line)
            st.dataframe(pd.DataFrame([ah]), use_container_width=True)
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
                st.warning("Provisional only: uncertainty margin is a conservative placeholder, not a fitted confidence interval. Calibration/backtesting is not complete; do not treat this as a live betting signal.")

elif page == "Risk Management":
    st.subheader("Risk Management")
    bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0)
    fraction = st.slider("Flat stake fraction", 0.25, 5.0, 1.0, 0.25) / 100
    st.metric("Suggested flat stake", f"{flat_stake(bankroll,fraction):,.2f}")
    st.caption("Conservative flat staking example. No martingale or loss-chasing.")

elif page == "Prediction Tracker":
    st.subheader("Prediction Tracker")
    st.markdown("#### Save a pre-match prediction")
    if matches.empty:
        st.info("Import match history first. You can still save a prediction only when its teams and kickoff are explicitly known.")
    with st.form("save_prediction_form"):
        match_id = st.text_input("Unique Match ID")
        home_team = st.text_input("Home team")
        away_team = st.text_input("Away team")
        kickoff_local = st.datetime_input("Kickoff (Asia/Jakarta)", value=datetime.now())
        market = st.selectbox("Selected market (choose NO BET if no qualifying value)", ["NO BET","1X2 Home","1X2 Draw","1X2 Away","Over 1.5","Under 1.5","Over 2.5","Under 2.5","Over 3.5","Under 3.5","BTTS Yes","BTTS No"])
        odds_input = st.number_input("Odds (1.01 if NO BET)", min_value=1.01, value=1.01, step=0.01)
        ev_input = st.number_input("Estimated EV (decimal; 0 if NO BET)", value=0.0, step=0.01)
        stake_input = st.number_input("Stake amount (0 for NO BET)", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Lock prediction")
    if submitted:
        try:
            if market != "NO BET" and (ev_input <= 0 or stake_input <= 0):
                raise ValueError("Selected market requires positive EV and stake. Otherwise choose NO BET.")
            if market == "NO BET":
                market_value, odds_value, ev_value, stake_value = None, None, None, 0.0
            else:
                market_value, odds_value, ev_value, stake_value = market, odds_input, ev_input, stake_input
            kickoff = kickoff_local.replace(tzinfo=ZoneInfo("Asia/Jakarta")).isoformat()
            pid = save_prediction(match_id,kickoff,home_team,away_team,{"model_status":"provisional"},
                                  market_value,odds_value,ev_value,"1.0.0",stake_value,DB_PATH)
            st.success(f"Prediction locked. Record ID: {pid}")
        except Exception as exc:
            st.error(str(exc))
    st.markdown("#### Settle completed match")
    open_preds = preds[preds.actual_home_goals.isna()] if not preds.empty else pd.DataFrame()
    if not open_preds.empty:
        choices = {f"#{int(r.id)} {r.home_team} vs {r.away_team}":int(r.id) for r in open_preds.itertuples()}
        with st.form("settle_prediction_form"):
            selected = st.selectbox("Prediction record", list(choices))
            hg = st.number_input("Actual home goals", min_value=0, step=1)
            ag = st.number_input("Actual away goals", min_value=0, step=1)
            settle = st.form_submit_button("Settle prediction")
        if settle:
            try:
                row = open_preds[open_preds.id == choices[selected]].iloc[0]
                if not row.selected_market:
                    profit = 0.0
                else:
                    won = market_won(row.selected_market,int(hg),int(ag))
                    profit = float(row.stake) * (float(row.odds)-1) if won else -float(row.stake)
                settle_prediction(choices[selected],int(hg),int(ag),profit,DB_PATH)
                st.success(f"Settled. Simulated P/L: {profit:.2f}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.dataframe(list_predictions(DB_PATH), use_container_width=True)

elif page == "Model Performance":
    st.subheader("Model Performance")
    settled = preds[preds.actual_home_goals.notna()] if not preds.empty else pd.DataFrame()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total predictions",len(preds))
    c2.metric("Settled",len(settled))
    c3.metric("Win rate", "N/A" if settled.empty else f"{sum((r.simulated_profit or 0)>0 for r in settled.itertuples())/len(settled):.1%}")
    if not settled.empty:
        st.metric("Net simulated P/L",f"{settled.simulated_profit.sum():.2f}")
        st.caption("ROI/CLV and calibration metrics are withheld until full odds snapshots, explicit stake records, and model probability payloads are recorded consistently. Win rate alone does not establish model edge.")
        st.dataframe(settled,use_container_width=True)
    else:
        st.info("No settled predictions yet. Metrics are not fabricated.")

elif page == "Settings":
    st.subheader("Settings")
    st.write("Model version: 1.0.0")
    st.write(f"Database: {DB_PATH}")
    st.write("Data source: user-provided CSV. No live feed or fabricated statistics are used.")
    st.warning("For hosted deployment, configure persistent disk storage for the SQLite database.")
