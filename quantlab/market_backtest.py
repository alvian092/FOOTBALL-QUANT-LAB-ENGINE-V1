"""Evaluate model 1X2 edges against pre-kickoff and closing Pinnacle prices."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")
RESULT_MAP = {"1": 0, "X": 1, "2": 2, "H": 0, "D": 1, "A": 2}


def _normalise_team(value) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    for suffix in ("footballclub", "fc", "afc"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value


def _devig(odds: np.ndarray) -> np.ndarray:
    if odds.shape != (3,) or not np.isfinite(odds).all() or (odds <= 1.0).any():
        raise ValueError("All three decimal odds must be finite and greater than 1.0.")
    inverse = 1.0 / odds
    return inverse / inverse.sum()


def _metric(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    one_hot = np.eye(3)[outcomes]
    return {
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == outcomes).mean()),
    }


def evaluate_market(predictions: pd.DataFrame, odds: pd.DataFrame,
                    model_prefix: str = "model_adaptive",
                    min_ev: float = 0.03, max_bets_per_match: int = 1) -> tuple[pd.DataFrame, dict]:
    """Match by normalized teams and kickoff date ±1 day; never use post-match prices as entry odds."""
    if not 0 <= min_ev < 1 or max_bets_per_match != 1:
        raise ValueError("min_ev must be in [0,1) and only one selection per match is supported.")
    pred_required = {"date", "home_team", "away_team", "outcome"} | {
        f"{model_prefix}_{outcome}_p" for outcome in OUTCOMES
    }
    odds_required = {"kickoff_utc", "home", "away", "result", "void_reason",
                     "teams_reversed_after_logging"} | {
        f"pin_open_{outcome}" for outcome in OUTCOMES
    } | {f"pin_close_{outcome}" for outcome in OUTCOMES}
    if pred_required - set(predictions.columns):
        raise ValueError(f"Prediction file missing: {sorted(pred_required - set(predictions.columns))}")
    if odds_required - set(odds.columns):
        raise ValueError(f"Odds file missing: {sorted(odds_required - set(odds.columns))}")

    p = predictions.copy().reset_index(drop=True)
    o = odds.copy().copy().reset_index(drop=True)
    p["date"] = pd.to_datetime(p.date, utc=True, errors="coerce")
    o["kickoff_utc"] = pd.to_datetime(o.kickoff_utc, utc=True, errors="coerce")
    p["home_key"] = p.home_team.map(_normalise_team)
    p["away_key"] = p.away_team.map(_normalise_team)
    o["home_key"] = o.home.map(_normalise_team)
    o["away_key"] = o.away.map(_normalise_team)
    o["result_code"] = o.result.astype(str).str.upper()
    o = o[o.void_reason.fillna("").astype(str).eq("") & o.teams_reversed_after_logging.fillna(0).astype(int).eq(0)]
    o = o[o.result_code.isin(RESULT_MAP)]
    o = o.dropna(subset=["kickoff_utc"])
    o["market_date"] = o.kickoff_utc.dt.floor("D")
    p["prediction_date"] = p.date.dt.floor("D")
    candidates = p.merge(o, on=["home_key", "away_key"], how="inner", suffixes=("_model", "_market"))
    candidates["date_gap_days"] = (candidates.market_date - candidates.prediction_date).dt.days.abs()
    candidates = candidates[candidates.date_gap_days <= 1].sort_values(
        ["date_gap_days", "kickoff_utc"], kind="stable"
    )
    candidates = candidates.drop_duplicates(subset=["match_id"] if "match_id" in candidates else ["home_key", "away_key", "prediction_date"])
    rows = []
    for row in candidates.to_dict("records"):
        try:
            model_p = np.array([row[f"{model_prefix}_{x}_p"] for x in OUTCOMES], dtype=float)
            if not np.isfinite(model_p).all() or (model_p < 0).any() or (model_p > 1).any() or not np.isclose(model_p.sum(), 1.0, atol=1e-5):
                continue
            entry = np.array([row[f"pin_open_{x}"] for x in OUTCOMES], dtype=float)
            closing = np.array([row[f"pin_close_{x}"] for x in OUTCOMES], dtype=float)
            market_p = _devig(entry)
            closing_p = _devig(closing)
        except (TypeError, ValueError):
            continue
        ev = model_p * entry - 1.0
        pick = int(np.argmax(ev))
        if ev[pick] < min_ev:
            continue
        actual = RESULT_MAP[str(row["result_code"]).upper()]
        odds_taken, odds_close = float(entry[pick]), float(closing[pick])
        profit = odds_taken - 1.0 if actual == pick else -1.0
        rows.append({
            "match_id": row.get("match_id"),
            "date": str(row["kickoff_utc"]),
            "home_team": row["home"], "away_team": row["away"],
            "competition": row.get("competition", row.get("league", "")),
            "selection": OUTCOMES[pick], "model_probability": float(model_p[pick]),
            "market_probability_no_vig": float(market_p[pick]),
            "entry_odds": odds_taken, "closing_odds": odds_close,
            "expected_value": float(ev[pick]),
            "clv_pct": float((odds_taken / odds_close - 1.0) * 100.0),
            "actual_outcome": OUTCOMES[actual], "profit_units": float(profit),
            "brier_model": float(np.square(model_p - np.eye(3)[actual]).sum()),
            "brier_market": float(np.square(market_p - np.eye(3)[actual]).sum()),
            "log_loss_model": float(-np.log(max(model_p[actual], 1e-15))),
            "log_loss_market": float(-np.log(max(market_p[actual], 1e-15))),
        })
    bets = pd.DataFrame(rows)
    matched = int(len(candidates))
    if bets.empty:
        summary = {"matched_fixtures_with_prices": matched, "bets": 0, "min_ev": min_ev,
                   "note": "No qualifying bets. This is not a zero-loss result; inspect odds coverage and edge threshold."}
        return bets, summary
    bets = bets.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    bets["cumulative_profit"] = bets.profit_units.cumsum()
    bets["running_peak"] = bets.cumulative_profit.cummax().clip(lower=0)
    bets["drawdown"] = bets.running_peak - bets.cumulative_profit
    stake = float(len(bets))
    summary = {
        "matched_fixtures_with_prices": matched, "bets": int(len(bets)),
        "coverage_of_matched_predictions": float(len(bets) / matched) if matched else 0.0,
        "min_ev": min_ev, "staking": "flat 1 unit per bet; max one 1X2 selection per match",
        "total_staked_units": stake, "profit_units": float(bets.profit_units.sum()),
        "roi_yield": float(bets.profit_units.sum() / stake),
        "hit_rate": float((bets.profit_units > 0).mean()),
        "average_expected_value": float(bets.expected_value.mean()),
        "average_clv_pct": float(bets.clv_pct.mean()),
        "positive_clv_share": float((bets.clv_pct > 0).mean()),
        "max_drawdown_units": float(bets.drawdown.max()),
        "model_vs_market_brier": {
            "model": float(bets.brier_model.mean()), "market_no_vig": float(bets.brier_market.mean())
        },
        "model_vs_market_log_loss": {
            "model": float(bets.log_loss_model.mean()), "market_no_vig": float(bets.log_loss_market.mean())
        },
        "caveat": "Conditional on the provider's logged fixture sample and Pinnacle prices. This is not a complete league-market census; positive historical ROI is not proof of future profitability.",
    }
    return bets, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1X2 model value, ROI, CLV and drawdown.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--odds", required=True)
    parser.add_argument("--model-prefix", default="model_adaptive")
    parser.add_argument("--min-ev", type=float, default=0.03)
    parser.add_argument("--output-dir", default="artifacts/market_intelligence")
    args = parser.parse_args()
    bets, summary = evaluate_market(pd.read_csv(args.predictions), pd.read_csv(args.odds),
                                    args.model_prefix, args.min_ev)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bets.to_csv(output / "market_bets.csv", index=False)
    (output / "market_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
