"""Leakage-safe EPL V4 walk-forward evaluation on historical results.

This pilot is goals/results-only: it does not estimate xG or betting returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .probability import calculate_probabilities


MODEL_NAMES = ("poisson", "dixon_coles", "elo", "ensemble")
ENSEMBLE_WEIGHTS = {"poisson": 0.4, "dixon_coles": 0.4, "elo": 0.2}


def _weighted_recent(values, half_life: float = 5.0) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot calculate a rate without historical observations.")
    weights = np.exp(-np.log(2.0) * np.arange(len(values)) / half_life)
    return float(np.average(values, weights=weights))


def _team_venue_rates(history: pd.DataFrame, team: str, venue: str,
                      league_scored: float, league_conceded: float) -> tuple[float, float]:
    if venue == "home":
        rows = history.loc[history.home_team == team].sort_values("date", ascending=False).head(20)
        scored, conceded = "home_goals", "away_goals"
    else:
        rows = history.loc[history.away_team == team].sort_values("date", ascending=False).head(20)
        scored, conceded = "away_goals", "home_goals"
    if rows.empty:
        return max(0.05, league_scored), max(0.05, league_conceded)
    return (
        max(0.05, _weighted_recent(rows[scored].to_numpy())),
        max(0.05, _weighted_recent(rows[conceded].to_numpy())),
    )


def _lambdas(history: pd.DataFrame, fixture: pd.Series) -> tuple[float, float]:
    league_home = max(0.05, float(history.home_goals.mean()))
    league_away = max(0.05, float(history.away_goals.mean()))
    home_scored, home_conceded = _team_venue_rates(
        history, fixture.home_team, "home", league_home, league_away
    )
    away_scored, away_conceded = _team_venue_rates(
        history, fixture.away_team, "away", league_away, league_home
    )
    home_attack = home_scored / league_home
    home_defence = home_conceded / league_away
    away_attack = away_scored / league_away
    away_defence = away_conceded / league_home
    lam_home = league_home * home_attack * away_defence * 1.08
    lam_away = league_away * away_attack * home_defence / 1.08
    return max(0.05, float(lam_home)), max(0.05, float(lam_away))


def _elo_probabilities_round_safe(history: pd.DataFrame, home_team: str,
                                  away_team: str) -> dict[str, float]:
    """Replay historical rounds simultaneously so same-day fixture order cannot matter."""
    ratings: dict[str, float] = {}
    for _, day in history.sort_values("date").groupby("date", sort=True):
        deltas: dict[str, float] = {}
        for row in day.itertuples():
            rh = ratings.get(str(row.home_team), 1500.0)
            ra = ratings.get(str(row.away_team), 1500.0)
            expected = 1.0 / (1.0 + 10.0 ** (-((rh + 60.0) - ra) / 400.0))
            actual = 1.0 if row.home_goals > row.away_goals else 0.5 if row.home_goals == row.away_goals else 0.0
            delta = 20.0 * (actual - expected)
            deltas[str(row.home_team)] = deltas.get(str(row.home_team), 0.0) + delta
            deltas[str(row.away_team)] = deltas.get(str(row.away_team), 0.0) - delta
        for team, delta in deltas.items():
            ratings[team] = ratings.get(team, 1500.0) + delta
    rh, ra = ratings.get(home_team, 1500.0), ratings.get(away_team, 1500.0)
    expected = 1.0 / (1.0 + 10.0 ** (-((rh + 60.0) - ra) / 400.0))
    draw = 0.26
    return {"home": (1.0 - draw) * expected, "draw": draw,
            "away": (1.0 - draw) * (1.0 - expected)}


def _as_dict(probabilities) -> dict[str, float]:
    return {"home": float(probabilities.home_win), "draw": float(probabilities.draw),
            "away": float(probabilities.away_win)}


def predict_components(history: pd.DataFrame, fixture: pd.Series) -> dict[str, dict[str, float]]:
    """Build four predictions using history supplied by the chronological runner only."""
    lam_home, lam_away = _lambdas(history, fixture)
    poisson_p = _as_dict(calculate_probabilities(lam_home, lam_away, rho=0.0))
    dc_p = _as_dict(calculate_probabilities(lam_home, lam_away, rho=-0.08))
    elo_p = _elo_probabilities_round_safe(history, str(fixture.home_team), str(fixture.away_team))
    ensemble = {}
    for outcome in ("home", "draw", "away"):
        ensemble[outcome] = (
            ENSEMBLE_WEIGHTS["poisson"] * poisson_p[outcome]
            + ENSEMBLE_WEIGHTS["dixon_coles"] * dc_p[outcome]
            + ENSEMBLE_WEIGHTS["elo"] * elo_p[outcome]
        )
    return {"poisson": poisson_p, "dixon_coles": dc_p, "elo": elo_p, "ensemble": ensemble}


def _metrics(probabilities: np.ndarray, outcomes: np.ndarray,
             bootstrap_samples: int = 1000, seed: int = 42) -> dict:
    if len(outcomes) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None,
                "brier_ci_low": None, "brier_ci_high": None}
    one_hot = np.eye(3)[outcomes]
    losses = np.sum((probabilities - one_hot) ** 2, axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        boot[i] = rng.choice(losses, size=len(losses), replace=True).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n": int(len(outcomes)),
        "brier": float(losses.mean()),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-15, 1)).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == outcomes).mean()),
        "brier_ci_low": float(lo),
        "brier_ci_high": float(hi),
    }


def run_epl_backtest(matches: pd.DataFrame, min_train: int = 100,
                     bootstrap_samples: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Evaluate each fixture with strictly earlier match dates as training history."""
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if min_train < 1 or bootstrap_samples < 100:
        raise ValueError("min_train must be positive and bootstrap_samples at least 100.")
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if df.date.isna().any():
        raise ValueError("Invalid match date found.")
    if df[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Missing goal result found.")
    df = df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)

    history_rows = []
    output = []
    # Add an entire calendar day to history only after all fixtures on that day are predicted.
    for match_date, day in df.groupby("date", sort=True):
        history = pd.DataFrame(history_rows)
        if len(history) >= min_train:
            for _, fixture in day.iterrows():
                components = predict_components(history, fixture)
                hg, ag = int(fixture.home_goals), int(fixture.away_goals)
                outcome = 0 if hg > ag else 1 if hg == ag else 2
                record = {
                    "match_id": fixture.get("match_id", None),
                    "date": match_date,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "actual_home_goals": hg,
                    "actual_away_goals": ag,
                    "outcome": outcome,
                    "training_matches": int(len(history)),
                }
                for model, probs in components.items():
                    for key in ("home", "draw", "away"):
                        record[f"{model}_{key}_p"] = probs[key]
                output.append(record)
        history_rows.extend(day.to_dict("records"))

    results = pd.DataFrame(output)
    summary = {
        "competition": "EPL",
        "method": "strict-date walk-forward; same-day fixtures excluded from training",
        "min_train": min_train,
        "predicted_matches": int(len(results)),
        "mean_training_history": float(results.training_matches.mean()) if not results.empty else None,
        "excluded_before_min_train": int(len(df) - len(results)),
        "total_input_matches": int(len(df)),
        "models": {},
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "data_limitations": [
            "Results/goals only; no xG fields were supplied.",
            "No bookmaker odds; ROI, yield, CLV and value-bet profitability are not evaluated.",
            "Fixed V4 ensemble weights are provisional and are not fitted on the evaluation period.",
            "Brier confidence intervals are fixture-level bootstrap intervals and do not correct for temporal dependence.",
        ],
    }
    outcomes = results.outcome.to_numpy(dtype=int) if not results.empty else np.array([], dtype=int)
    for i, model in enumerate(MODEL_NAMES):
        cols = [f"{model}_{key}_p" for key in ("home", "draw", "away")]
        probs = results[cols].to_numpy(dtype=float) if not results.empty else np.empty((0, 3))
        summary["models"][model] = _metrics(probs, outcomes, bootstrap_samples, seed=42 + i)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EPL V4 walk-forward backtest.")
    parser.add_argument("--input", default="data/epl_pilot/epl_results_2010_2026.csv")
    parser.add_argument("--output-dir", default="artifacts/epl_v4_backtest")
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    results, summary = run_epl_backtest(data, args.min_train, args.bootstrap_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "epl_v4_oos_predictions.csv", index=False)
    (output_dir / "epl_v4_backtest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
