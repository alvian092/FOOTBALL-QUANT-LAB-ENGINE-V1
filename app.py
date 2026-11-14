from __future__ import annotations
import hashlib
import json
import tempfile
from datetime import datetime, timezone, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from quantlab.data import validate_matches_csv, team_form_stats, venue_stats
from quantlab.backtesting import walk_forward_backtest, segmented_performance
from quantlab.ensemble import elo_ratings_from_history, poisson_component, blend_probabilities, fit_ensemble_weights
from quantlab.calibration import fit_temperature, calibration_report, apply_temperature
from quantlab.scanner import scan_1x2_market
from quantlab.market_intelligence import validate_odds_snapshots, summarize_odds_movement
from quantlab.probability import calculate_probabilities, estimate_expected_goals, top_correct_scores
from quantlab.market import evaluate_value, no_vig_probabilities, overround
from quantlab.asian_handicap import asian_handicap_outcomes, asian_handicap_ev
from quantlab.storage import (initialize, load_matches, save_dataset, save_prediction,
                              list_predictions, settle_prediction, list_datasets, record_closing_odds, backup_database, restore_database,
                              save_odds_snapshot, list_odds_snapshots)
from quantlab.risk import flat_stake
from quantlab.evaluation import brier_score, log_loss, backtest_summary, calibration_bins, bankroll_metrics

st.set_page_config(page_title="Football Quant Lab", page_icon="⚽", layout="wide")
DB_PATH = Path("data/football_quant_lab.sqlite3")
initialize(DB_PATH)

st.markdown("""
<style>
.stApp {background: #0b1220; color: #e5e7eb;}
[data-testid="stMetric"] {background: #111c2e; padding: 14px; border-radius: 10px;}
[data-testid="stSidebar"] {background: #101a2b;}
@media (max-width: 768px) {
  .block-container {padding-left: 0.85rem; padding-right: 0.85rem;}
  [data-testid="stMetric"] {padding: 9px; margin-bottom: 6px;}
  [data-testid="stDataFrame"] {overflow-x: auto;}
}
</style>
""", unsafe_allow_html=True)

st.title("⚽ FOOTBALL QUANT LAB")
st.caption("Quantitative football analysis • Probability • Market intelligence • Risk control")
page = st.sidebar.radio("NAVIGATION", ["Dashboard Overview", "Match Analysis", "Probability Explorer",
    "Odds Intelligence", "Value Bet Scanner", "Risk Management", "Prediction Tracker",
    "Model Performance", "CSV Data Management", "Walk-Forward Backtest", "Market Intelligence", "V4 Predictive Edge Scanner", "Backup & Restore", "Settings"])

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
    st.write("Required: date, home_team, away_team, home_goals, away_goals. Optional: match_id, home_xg, away_xg (provide both for xG-based analysis).")
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

elif page == "Market Intelligence":
    st.subheader("Historical Market Intelligence")
    st.write("Upload timestamped bookmaker prices. Each snapshot should contain all mutually exclusive selections for one match, market, bookmaker and timestamp.")
    st.caption("CSV fields: match_id, market, selection, odds, captured_at, optional bookmaker. Example market: 1X2; selections: Home, Draw, Away.")
    odds_upload = st.file_uploader("Upload odds history CSV", type=["csv"], key="v3_odds_history")
    if odds_upload:
        try:
            raw_odds = pd.read_csv(odds_upload)
            checked = validate_odds_snapshots(raw_odds)
            st.success(f"Validated {len(checked)} odds rows.")
            st.dataframe(checked.head(15), use_container_width=True)
            if st.button("Import immutable odds snapshots"):
                for (match_id, market, bookmaker, captured_at), part in checked.groupby(
                    ["match_id","market","bookmaker","captured_at"], dropna=False):
                    save_odds_snapshot(match_id,market,dict(zip(part.selection,part.odds)),
                        captured_at,bookmaker,source=odds_upload.name,db_path=DB_PATH)
                st.success("Odds snapshots imported. Existing timestamp/selection records are not overwritten.")
                st.rerun()
        except Exception as exc:
            st.error(f"Odds CSV rejected: {exc}")
    snapshots = list_odds_snapshots(DB_PATH)
    if not snapshots.empty:
        movement = summarize_odds_movement(snapshots.rename(columns={"captured_at":"captured_at"}))
        st.markdown("#### Opening-to-latest movement")
        if not movement.empty:
            st.metric("Tracked selection movements",len(movement))
            st.dataframe(movement,use_container_width=True)
            st.download_button("Download movement report",movement.to_csv(index=False),
                "football_quant_lab_market_movement.csv",mime="text/csv")
        else:
            st.info("Need at least two selections in each timestamped market snapshot.")
        st.markdown("#### Raw odds snapshots")
        st.dataframe(snapshots,use_container_width=True)
    else:
        st.info("No historical odds snapshots yet. Upload an odds history CSV to activate movement analysis.")


elif page == "V4 Predictive Edge Scanner":
    st.subheader("V4 — Predictive Edge & Automated Market Scanner")
    st.caption("Ensemble preview, chronological calibration workflow, and conservative market screening. No live bookmaker feed is connected.")

    st.markdown("### 1. Three-model ensemble preview")
    teams = sorted(set(matches.home_team.astype(str)) | set(matches.away_team.astype(str))) if not matches.empty else []
    if len(teams) >= 2:
        left, right = st.columns(2)
        home = left.selectbox("Home team", teams, key="v4_home")
        away = right.selectbox("Away team", [t for t in teams if t != home], key="v4_away")
        fitted_weights = st.session_state.get("v4_fitted_weights", {"poisson":0.4,"dixon_coles":0.4,"elo":0.2})
        pw = st.slider("Poisson weight", 0.0, 1.0, float(fitted_weights["poisson"]), 0.05, key="v4_pw")
        dw = st.slider("Dixon-Coles weight", 0.0, 1.0, float(fitted_weights["dixon_coles"]), 0.05, key="v4_dw")
        ew = st.slider("Elo weight", 0.0, 1.0, float(fitted_weights["elo"]), 0.05, key="v4_ew")
        if pw + dw + ew > 0 and st.button("Build ensemble forecast", type="primary"):
            try:
                hs = venue_stats(matches, home, "home", 20, 5)
                aws = venue_stats(matches, away, "away", 20, 5)
                if min(hs["matches"], aws["matches"]) < 5:
                    raise ValueError("Minimum 5 home matches and 5 away matches are required.")
                has_xg = hs["xg_for"] is not None and aws["xg_for"] is not None
                if has_xg:
                    lh, la = estimate_expected_goals(hs["xg_for"], hs["xg_against"], aws["xg_for"], aws["xg_against"], float(matches.home_xg.mean()), float(matches.away_xg.mean()))
                    source = "xG/xGA"
                else:
                    lh, la = estimate_expected_goals(hs["scored_per_match"], hs["conceded_per_match"], aws["scored_per_match"], aws["conceded_per_match"], float(matches.home_goals.mean()), float(matches.away_goals.mean()))
                    source = "Goals proxy (no xG feed)"
                components = {
                    "Poisson": poisson_component(lh, la, rho=0.0),
                    "Dixon-Coles": poisson_component(lh, la, rho=-0.08),
                    "Elo": elo_ratings_from_history(matches, home, away),
                }
                ensemble = blend_probabilities({"poisson": components["Poisson"], "dixon_coles": components["Dixon-Coles"], "elo": components["Elo"]}, {"poisson": pw, "dixon_coles": dw, "elo": ew})
                components["Ensemble"] = ensemble
                st.session_state["v4_ensemble"] = {"home": home, "away": away, "lh": lh, "la": la, "source": source, "components": components}
            except Exception as exc:
                st.error(f"Ensemble calculation failed: {exc}")
        forecast = st.session_state.get("v4_ensemble")
        if forecast:
            st.write(f"{forecast['home']} vs {forecast['away']} • Expected goals {forecast['lh']:.2f}–{forecast['la']:.2f} • Source: {forecast['source']}")
            table = [{"Model": name, "Home": p["home"], "Draw": p["draw"], "Away": p["away"]} for name, p in forecast["components"].items()]
            st.dataframe(pd.DataFrame(table).set_index("Model").style.format("{:.1%}"), use_container_width=True)
            st.caption("Weights are user-selected starting weights, not fitted or validated weights. Optimize only with out-of-fold predictions.")
    else:
        st.info("Import match history first to preview the ensemble.")

    st.markdown("#### Fit ensemble weights from out-of-fold predictions")
    st.caption("CSV needs outcome plus 9 columns: poisson_home_p/draw_p/away_p, dixon_coles_home_p/draw_p/away_p, elo_home_p/draw_p/away_p. Use predictions generated strictly out of fold.")
    weight_file = st.file_uploader("Upload OOF component predictions", type=["csv"], key="v4_weights")
    if weight_file:
        try:
            wf = pd.read_csv(weight_file)
            component_columns = {
                "poisson":["poisson_home_p","poisson_draw_p","poisson_away_p"],
                "dixon_coles":["dixon_coles_home_p","dixon_coles_draw_p","dixon_coles_away_p"],
                "elo":["elo_home_p","elo_draw_p","elo_away_p"],
            }
            required_weight = {"outcome"} | {c for cols in component_columns.values() for c in cols}
            if not required_weight.issubset(wf.columns):
                raise ValueError("Missing outcome or component probability columns.")
            if st.button("Fit non-negative ensemble weights"):
                oof = {name:wf[cols].to_numpy(dtype=float) for name,cols in component_columns.items()}
                fitted = fit_ensemble_weights(oof,wf.outcome.to_numpy(dtype=int),min_samples=100)
                st.session_state["v4_fitted_weights"] = fitted
                st.success("OOF ensemble weights fitted and saved for this dashboard session.")
            if st.session_state.get("v4_fitted_weights"):
                st.dataframe(pd.DataFrame([st.session_state["v4_fitted_weights"]],index=["Fitted weights"]),use_container_width=True)
        except Exception as exc:
            st.error(f"OOF weight fitting rejected: {exc}")

    st.divider()
    st.markdown("### 2. Chronological probability calibration")
    st.write("Upload out-of-fold 1X2 predictions. Oldest 70% fits temperature; newest 30% is held out.")
    st.caption("Required: date, outcome (0=home, 1=draw, 2=away), model_home_p, model_draw_p, model_away_p.")
    cal_file = st.file_uploader("Upload calibration CSV", type=["csv"], key="v4_calibration")
    if cal_file:
        try:
            cal = pd.read_csv(cal_file)
            needed = {"date", "outcome", "model_home_p", "model_draw_p", "model_away_p"}
            if not needed.issubset(cal.columns):
                raise ValueError("Missing required calibration columns.")
            cal["date"] = pd.to_datetime(cal.date, errors="coerce", utc=True)
            cal["outcome"] = pd.to_numeric(cal.outcome, errors="coerce")
            cal = cal.dropna(subset=["date", "outcome"]).sort_values("date").reset_index(drop=True)
            cut = int(len(cal) * 0.70)
            if cut < 50 or len(cal) - cut < 20:
                raise ValueError("Need at least 50 older calibration fixtures and 20 newer holdout fixtures.")
            cols = ["model_home_p", "model_draw_p", "model_away_p"]
            train_p, train_y = cal.loc[:cut-1, cols].to_numpy(float), cal.loc[:cut-1, "outcome"].to_numpy(int)
            test_p, test_y = cal.loc[cut:, cols].to_numpy(float), cal.loc[cut:, "outcome"].to_numpy(int)
            fitted = fit_temperature(train_p, train_y, min_samples=50)
            calibrated = apply_temperature(test_p, fitted["temperature"])
            before, after = calibration_report(test_p, test_y), calibration_report(calibrated, test_y)
            a, b, c = st.columns(3)
            a.metric("Temperature", f"{fitted['temperature']:.3f}")
            b.metric("Holdout log loss", f"{before['log_loss']:.4f} → {after['log_loss']:.4f}")
            c.metric("Holdout ECE", f"{before['ece']:.4f} → {after['ece']:.4f}")
            st.dataframe(pd.DataFrame([
                {"Metric": "Brier", "Before": before["brier"], "After": after["brier"]},
                {"Metric": "Log loss", "Before": before["log_loss"], "After": after["log_loss"]},
                {"Metric": "ECE", "Before": before["ece"], "After": after["ece"]},
            ]), use_container_width=True)
            out = cal.loc[cut:].copy()
            out[["calibrated_home_p", "calibrated_draw_p", "calibrated_away_p"]] = calibrated
            st.download_button("Download calibrated holdout", out.to_csv(index=False), "football_quant_lab_calibrated_holdout.csv", mime="text/csv")
            st.caption("Temperature is fitted on the older split only. Metrics use the untouched newer holdout. Do not tune on that holdout afterward.")
        except Exception as exc:
            st.error(f"Calibration upload rejected: {exc}")

    st.divider()
    st.markdown("### 3. Conservative 1X2 market scanner")
    st.write("Lower probability columns must be genuine lower confidence bounds from out-of-sample/bootstrap analysis. The scanner never invents uncertainty.")
    st.caption("Required: match_id, home_odds, draw_odds, away_odds, model_home_p, model_draw_p, model_away_p, lower_home_p, lower_draw_p, lower_away_p. Optional: league.")
    scan_file = st.file_uploader("Upload fixtures and bookmaker odds CSV", type=["csv"], key="v4_scan")
    if scan_file:
        try:
            scan_df = pd.read_csv(scan_file)
            edge_threshold = st.slider("Minimum edge vs no-vig market", 0.0, 0.15, 0.02, 0.005, key="v4_edge")
            ev_threshold = st.slider("Minimum conservative EV", 0.0, 0.10, 0.0, 0.005, key="v4_ev")
            max_margin = st.slider("Maximum bookmaker overround", 0.0, 0.30, 0.15, 0.01, key="v4_margin")
            if st.button("Scan value candidates", type="primary"):
                st.session_state["v4_scan_results"] = scan_1x2_market(scan_df, min_edge=edge_threshold, min_conservative_ev=ev_threshold, max_overround=max_margin)
            ranked = st.session_state.get("v4_scan_results")
            if ranked is not None:
                candidates = ranked[ranked.status == "VALUE CANDIDATE"]
                a, b = st.columns(2)
                a.metric("Fixtures scanned", ranked.match_id.nunique())
                b.metric("Qualified selections", len(candidates))
                st.dataframe(ranked, use_container_width=True)
                st.download_button("Download scanner report", ranked.to_csv(index=False), "football_quant_lab_v4_market_scan.csv", mime="text/csv")
                if candidates.empty:
                    st.warning("NO VALUE = NO BET. No selection passed all conservative filters.")
        except Exception as exc:
            st.error(f"Scanner upload rejected: {exc}")

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
        recency_half_life = st.slider("Recency half-life (matches)", 2, 10, 5)
        if st.button("Calculate probabilities", type="primary"):
            try:
                hs = venue_stats(matches, home, "home", history_limit, recency_half_life)
                aws = venue_stats(matches, away, "away", history_limit, recency_half_life)
                if min(hs["matches"], aws["matches"]) < 5:
                    raise ValueError("Insufficient venue history: home team needs 5 home matches and away team needs 5 away matches.")
                has_xg = hs["xg_for"] is not None and aws["xg_for"] is not None
                if has_xg:
                    league_home, league_away = float(matches.home_xg.mean()), float(matches.away_xg.mean())
                    home_attack, home_defence = hs["xg_for"], hs["xg_against"]
                    away_attack, away_defence = aws["xg_for"], aws["xg_against"]
                else:
                    league_home, league_away = float(matches.home_goals.mean()), float(matches.away_goals.mean())
                    home_attack, home_defence = hs["scored_per_match"], hs["conceded_per_match"]
                    away_attack, away_defence = aws["scored_per_match"], aws["conceded_per_match"]
                lh, la = estimate_expected_goals(home_attack, home_defence, away_attack, away_defence, league_home, league_away)
                p = calculate_probabilities(lh, la)
                st.session_state["last_analysis"] = {"home":home,"away":away,"lh":lh,"la":la,"p":p,
                    "home_matches":hs["matches"],"away_matches":aws["matches"],"has_xg":has_xg,
                    "home_gf":hs["scored_per_match"],"home_ga":hs["conceded_per_match"],
                    "away_gf":aws["scored_per_match"],"away_ga":aws["conceded_per_match"],
                    "home_xgf":hs["xg_for"],"home_xga":hs["xg_against"],
                    "away_xgf":aws["xg_for"],"away_xga":aws["xg_against"],"half_life":recency_half_life}
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.get("last_analysis")
        if result and result["home"] == home and result["away"] == away:
            p = result["p"]
            st.write(f"Expected goals: **{home} {result['lh']:.2f} — {result['la']:.2f} {away}**")
            st.caption(("xG/xGA venue model" if result["has_xg"] else "Goals-based venue model (xG columns absent)") + f" • {int(result['home_matches'])} home matches for {home} • {int(result['away_matches'])} away matches for {away} • recency half-life {result['half_life']} matches.")
            if result["has_xg"]:
                st.write(f"Venue xG/xGA: {home} {result['home_xgf']:.2f}/{result['home_xga']:.2f} • {away} {result['away_xgf']:.2f}/{result['away_xga']:.2f}")
            else:
                st.write(f"Venue GF/GA: {home} {result['home_gf']:.2f}/{result['home_ga']:.2f} • {away} {result['away_gf']:.2f}/{result['away_ga']:.2f}")
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
            st.markdown("#### Lock this pre-match model snapshot")
            with st.form("analysis_prediction_form"):
                record_id = st.text_input("Unique Match ID")
                kickoff_date = st.date_input("Kickoff date (Asia/Jakarta)")
                kickoff_time = st.time_input("Kickoff time (Asia/Jakarta)", value=dt_time(19, 0))
                chosen_market = st.selectbox("Market", ["NO BET","1X2 Home","1X2 Draw","1X2 Away","Over 1.5","Under 1.5","Over 2.5","Under 2.5","Over 3.5","Under 3.5","BTTS Yes","BTTS No"])
                record_odds = st.number_input("Decimal odds (used only when market selected)", min_value=1.01, value=2.0, step=0.01)
                record_stake = st.number_input("Stake (0 for NO BET)", min_value=0.0, value=0.0)
                lock = st.form_submit_button("Save immutable prediction snapshot")
            if lock:
                try:
                    kickoff = datetime.combine(kickoff_date, kickoff_time).replace(tzinfo=ZoneInfo("Asia/Jakarta")).isoformat()
                    if chosen_market == "NO BET":
                        selected, odds_value, ev_value, stake_value = None, None, None, 0.0
                    else:
                        model_p = market_probability(chosen_market, p)
                        calculated_ev = model_p * record_odds - 1.0
                        if calculated_ev <= 0 or record_stake <= 0:
                            raise ValueError("NO VALUE = NO BET. Positive raw EV and a positive stake are required to record a selected market.")
                        selected, odds_value, ev_value, stake_value = chosen_market, record_odds, calculated_ev, record_stake
                    record = save_prediction(record_id, kickoff, home, away, p.as_dict(),
                                              selected, odds_value, ev_value, "1.0.0",
                                              stake_value, DB_PATH)
                    st.success(f"Prediction locked as record #{record}.")
                except Exception as exc:
                    st.error(str(exc))
            st.caption("Snapshot stores calculated probabilities. The estimator remains provisional and uncalibrated; raw EV alone is not a production-grade betting signal.")

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

elif page == "Walk-Forward Backtest":
    st.subheader("Chronological Walk-Forward Backtest")
    st.caption("Each fixture is predicted using strictly earlier results. Same-day results are excluded. League-specific histories are used when league metadata exists.")
    if matches.empty:
        st.warning("Import historical match CSV first.")
    else:
        max_train = max(5, min(len(matches)-1, 500))
        min_train = st.slider("Minimum prior matches per league",5,max_train,min(50,max_train))
        bootstrap_n = st.selectbox("Bootstrap samples for 95% interval",[500,1000,2000],index=1)
        if st.button("Run walk-forward backtest",type="primary"):
            def historical_predictor(train, fixture):
                league_home = float(train.home_goals.mean())
                league_away = float(train.away_goals.mean())
                league_home, league_away = float(train.home_goals.mean()), float(train.away_goals.mean())
                try:
                    hstats = team_form_stats(train, str(fixture.home_team), limit=20, half_life=5)
                except ValueError:
                    hstats = {"scored_per_match":league_home,"conceded_per_match":league_away,"xg_for":None,"xg_against":None}
                try:
                    astats = team_form_stats(train, str(fixture.away_team), limit=20, half_life=5)
                except ValueError:
                    astats = {"scored_per_match":league_away,"conceded_per_match":league_home,"xg_for":None,"xg_against":None}
                has_xg = {"home_xg","away_xg"}.issubset(train.columns) and train[["home_xg","away_xg"]].notna().all().all()
                if has_xg:
                    league_home, league_away = float(train.home_xg.mean()), float(train.away_xg.mean())
                    h_for, h_against = hstats["xg_for"], hstats["xg_against"]
                    a_for, a_against = astats["xg_for"], astats["xg_against"]
                    if any(v is None for v in (h_for,h_against,a_for,a_against)):
                        has_xg = False
                if not has_xg:
                    h_for, h_against = hstats["scored_per_match"], hstats["conceded_per_match"]
                    a_for, a_against = astats["scored_per_match"], astats["conceded_per_match"]
                    league_home, league_away = float(train.home_goals.mean()), float(train.away_goals.mean())
                lh, la = estimate_expected_goals(max(.05,h_for),max(.05,h_against),
                    max(.05,a_for),max(.05,a_against),max(.05,league_home),max(.05,league_away))
                probs = calculate_probabilities(lh,la)
                return {"home":probs.home_win,"draw":probs.draw,"away":probs.away_win}
            try:
                report = walk_forward_backtest(matches,historical_predictor,min_train=min_train,
                    bootstrap_samples=bootstrap_n,group_by_league=("league" in matches.columns))
                st.session_state["v3_backtest"] = report
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
        report = st.session_state.get("v3_backtest")
        if report:
            results = report["results"]
            if results.empty:
                st.warning("No test fixtures met the minimum historical sample. Reduce the minimum or import more history.")
            else:
                st.markdown("#### Model performance")
                a,b,c = st.columns(3)
                a.metric("Out-of-sample fixtures",report["model"]["n"])
                b.metric("Model Brier",f"{report['model']['brier']:.4f}")
                c.metric("Model log loss",f"{report['model']['log_loss']:.4f}")
                st.caption(f"Model Brier 95% bootstrap CI: {report['model_brier_ci'][0]:.4f}–{report['model_brier_ci'][1]:.4f}")
                if report["market"]["n"]:
                    x,y,z = st.columns(3)
                    x.metric("Bookmaker fixtures",report["market"]["n"])
                    y.metric("Bookmaker Brier",f"{report['market']['brier']:.4f}")
                    z.metric("Bookmaker log loss",f"{report['market']['log_loss']:.4f}")
                    st.write(f"Model edge vs market — Brier improvement: **{report['model_edge_vs_market_brier']:+.4f}** • Log-loss improvement: **{report['model_edge_vs_market_log_loss']:+.4f}** (positive favours model).")
                    st.caption(f"Paired 95% bootstrap CI — Brier edge: {report['model_edge_brier_ci'][0]:+.4f} to {report['model_edge_brier_ci'][1]:+.4f}; Log-loss edge: {report['model_edge_log_loss_ci'][0]:+.4f} to {report['model_edge_log_loss_ci'][1]:+.4f}. An interval crossing zero does not establish superiority.")
                else:
                    st.info("No complete 1X2 odds columns found. Model vs bookmaker comparison is unavailable.")
                st.dataframe(results,use_container_width=True)
                st.download_button("Download out-of-sample predictions",results.to_csv(index=False),
                    "football_quant_lab_walk_forward.csv",mime="text/csv")
    settled_bets = preds[preds.actual_home_goals.notna() & preds.selected_market.notna()].copy() if not preds.empty else pd.DataFrame()
    if not settled_bets.empty:
        st.markdown("#### Settled bet segments")
        st.dataframe(segmented_performance(settled_bets,group_by="selected_market",min_bets=30),use_container_width=True)

elif page == "Risk Management":
    st.subheader("Risk Management")
    bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0)
    fraction = st.slider("Flat stake fraction", 0.25, 5.0, 1.0, 0.25) / 100
    st.metric("Suggested flat stake", f"{flat_stake(bankroll,fraction):,.2f}")
    st.caption("Conservative flat staking example. No martingale or loss-chasing.")

elif page == "Prediction Tracker":
    st.subheader("Prediction Tracker")
    st.markdown("#### Settle completed match")
    open_preds = preds[preds.actual_home_goals.isna()] if not preds.empty else pd.DataFrame()
    if not open_preds.empty:
        choices = {f"#{int(r.id)} {r.home_team} vs {r.away_team}":int(r.id) for r in open_preds.itertuples()}
        odds_choices = {k:v for k,v in choices.items() if open_preds.loc[open_preds.id == v,"selected_market"].iloc[0]}
        if odds_choices:
            with st.form("closing_odds_form"):
                closing_label = st.selectbox("Record closing odds for",list(odds_choices))
                closing_value = st.number_input("Closing decimal odds",min_value=1.01,value=2.00,step=0.01)
                save_close = st.form_submit_button("Save closing odds")
            if save_close:
                try:
                    record_closing_odds(odds_choices[closing_label],closing_value,DB_PATH)
                    st.success("Closing odds saved.")
                    st.rerun()
                except Exception as exc: st.error(str(exc))
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
    st.dataframe(list_predictions(DB_PATH), use_container_width=True) if not preds.empty else st.info("No prediction snapshots saved yet.")

elif page == "Model Performance":
    st.subheader("Model Performance — actual settled records")
    settled = preds[preds.actual_home_goals.notna()].copy() if not preds.empty else pd.DataFrame()
    bets = settled[settled.selected_market.notna()].copy() if not settled.empty else pd.DataFrame()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total predictions",len(preds))
    c2.metric("Settled",len(settled))
    c3.metric("Settled bets",len(bets))
    if not bets.empty:
        metrics = bankroll_metrics(bets.simulated_profit.astype(float).to_numpy(), bets.stake.astype(float).to_numpy())
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Net P/L",f"{metrics['net_profit']:.2f}")
        m2.metric("ROI / Yield","N/A" if metrics["roi"] is None else f"{metrics['roi']:.2%}")
        m3.metric("Max drawdown",f"{metrics['max_drawdown']:.2f}")
        m4.metric("Max drawdown %","N/A" if metrics["max_drawdown_pct"] is None else f"{metrics['max_drawdown_pct']:.1%}")
        clv_rows = bets[bets.closing_odds.notna() & bets.odds.notna()]
        if not clv_rows.empty:
            clv = bankroll_metrics(clv_rows.simulated_profit.astype(float),clv_rows.stake.astype(float),
                clv_rows.odds.astype(float),clv_rows.closing_odds.astype(float))["clv"]
            st.metric("Average CLV",f"{clv:.2%}")
        else:
            st.caption("CLV belum tersedia: catat closing odds sebelum kickoff pada Prediction Tracker.")
    else:
        st.info("Belum ada taruhan settled dengan stake aktual. ROI/yield tidak dihitung dari prediksi NO BET.")
    if not settled.empty:
        home_probs, home_outcomes = [], []
        for r in settled.itertuples():
            try:
                probs = json.loads(r.probabilities_json)
                home_probs.append(float(probs["home_win"]))
                home_outcomes.append(int(r.actual_home_goals > r.actual_away_goals))
            except (ValueError, KeyError, TypeError):
                continue
        if home_probs:
            report = backtest_summary(home_probs,home_outcomes)
            st.markdown("#### Out-of-sample home-win calibration")
            a,b,c = st.columns(3)
            a.metric("Brier score",f"{report['brier']:.4f}")
            b.metric("95% bootstrap CI",f"{report['brier_ci_low']:.4f}–{report['brier_ci_high']:.4f}")
            c.metric("Home-win hit rate",f"{report['hit_rate']:.1%}")
            st.dataframe(pd.DataFrame(calibration_bins(home_probs,home_outcomes)),use_container_width=True)
            st.caption(f"Based on {report['n']} settled prediction snapshots. Bootstrap CI is a sample-uncertainty interval, not a guarantee of future performance.")
        st.dataframe(settled,use_container_width=True)
    else:
        st.info("No settled predictions yet. Metrics are not fabricated.")

elif page == "Backup & Restore":
    st.subheader("Database Backup & Restore")
    st.warning("Restore replaces the current database. Download a backup first; never restore an untrusted file.")
    if DB_PATH.exists():
        snapshot = backup_database(DB_PATH)
        st.download_button("Download consistent SQLite backup",snapshot.read_bytes(),
            "football_quant_lab_backup.sqlite3",mime="application/octet-stream")
    restore_file = st.file_uploader("Select a Football Quant Lab SQLite backup",type=["sqlite3","sqlite","db"])
    if restore_file and st.button("Validate and restore database",type="primary"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".sqlite3",delete=False) as tmp:
                tmp.write(restore_file.getvalue()); temp_path=Path(tmp.name)
            restore_database(temp_path,DB_PATH)
            temp_path.unlink(missing_ok=True)
            st.success("Database restored successfully.")
            st.rerun()
        except Exception as exc:
            st.error(f"Restore rejected: {exc}")

elif page == "Settings":
    st.subheader("Settings & Android deployment")
    st.write("Model version: 2.0.0")
    st.write(f"Database: {DB_PATH}")
    st.write("Data source: user-provided match CSV. xG is only used when both home_xg and away_xg columns are present.")
    st.info("Deploy this repository with Streamlit Community Cloud: create an app from the GitHub repository, select branch main and app.py, then open the generated HTTPS URL in Chrome on Android. Use Chrome → Add to Home screen for app-like access.")
    st.warning("SQLite persistence on free ephemeral hosting is not guaranteed. Configure persistent storage or scheduled backups before relying on saved records.")
