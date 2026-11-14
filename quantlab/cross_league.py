"""Independent, league-by-league chronological validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .epl_backtest import run_epl_backtest
from .epl_calibration import run_expanding_calibration


def _metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    one_hot = np.eye(3)[outcomes]
    return {
        "n": int(len(outcomes)),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(
            probabilities[np.arange(len(outcomes)), outcomes], 1e-15, 1.0
        )).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == outcomes).mean()),
    }


def run_cross_league(inputs: dict[str, str], min_train: int = 100,
                     min_calibration_history: int = 300,
                     bootstrap_samples: int = 20_000, seed: int = 20261114):
    prediction_frames = []
    league_summaries = {}
    for league, path in sorted(inputs.items()):
        matches = pd.read_csv(path)
        predictions, summary = run_epl_backtest(
            matches, min_train=min_train, bootstrap_samples=1000
        )
        summary["competition"] = league
        summary["input_matches"] = int(len(matches))
        summary["season_count"] = int(matches.season.nunique()) if "season" in matches else None
        if predictions.empty:
            league_summaries[league] = summary
            continue
        calibrated, parameters, calibration_summary = run_expanding_calibration(
            predictions, min_history=min_calibration_history,
            refit_every_dates=10, bootstrap_samples=bootstrap_samples, seed=seed,
        )
        for column in ("match_id", "home_team", "away_team", "training_matches"):
            calibrated[column] = predictions[column].to_numpy()
        calibrated["competition"] = league

        # Keep raw and strictly prequential-calibrated forecasts separate.
        for outcome in ("home", "draw", "away"):
            calibrated[f"model_adaptive_{outcome}_p"] = calibrated[
                f"adaptive_ensemble_raw_{outcome}_p"
            ]
            calibrated[f"model_adaptive_calibrated_{outcome}_p"] = calibrated[
                f"adaptive_ensemble_cal_{outcome}_p"
            ]
            calibrated[f"model_elo_{outcome}_p"] = calibrated[f"elo_cal_{outcome}_p"]

        prediction_frames.append(calibrated)
        league_summaries[league] = {
            **summary,
            "season_count": int(matches.season.nunique()) if "season" in matches else None,
            "calibration": calibration_summary["metrics"],
            "paired_comparisons": calibration_summary["paired_comparisons"],
            "calibration_parameter_dates": int(len(parameters)),
            "calibration_min_history": min_calibration_history,
        }

    combined = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    overall = {}
    if not combined.empty:
        for model in ("model_adaptive", "model_adaptive_calibrated", "model_elo"):
            columns = [f"{model}_{outcome}_p" for outcome in ("home", "draw", "away")]
            overall[model] = _metrics(combined[columns].to_numpy(dtype=float),
                                      combined.outcome.to_numpy(dtype=int))
    return combined, {
        "method": "independent strict-date walk-forward and per-league expanding calibration",
        "leagues": league_summaries,
        "pooled_descriptive_metrics": overall,
        "market_validation": {
            "raw_model_prefix": "model_adaptive",
            "calibrated_model_prefix": "model_adaptive_calibrated",
            "calibrated_forecasts": "expanding-window temperature scaling and OOF ensemble weights; each date uses earlier OOS history only",
            "instruction": "Evaluate raw and calibrated forecasts separately against identical odds and EV thresholds; do not select whichever has better retrospective ROI without a new holdout.",
        },
        "warning": "Pooled metrics are descriptive only; inspect league-specific results. Market metrics are conditional on the matched odds sample and do not establish future profitability.",
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-league walk-forward validation.")
    parser.add_argument("--input", action="append", required=True, metavar="LEAGUE=CSV")
    parser.add_argument("--output-dir", default="artifacts/cross_league")
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--min-calibration-history", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    inputs = {}
    for item in args.input:
        if "=" not in item:
            parser.error("--input must use LEAGUE=CSV")
        league, path = item.split("=", 1)
        inputs[league] = path
    predictions, summary = run_cross_league(
        inputs, args.min_train, args.min_calibration_history,
        args.bootstrap_samples, args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "cross_league_oos_predictions.csv", index=False)
    (output / "cross_league_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
