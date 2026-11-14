"""Segmented performance, prequential temperature calibration and report export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

OUTCOMES = ("home", "draw", "away")


def _probability_metrics(frame: pd.DataFrame, columns: tuple[str, str, str]) -> dict:
    if frame.empty:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None, "ece": None}
    probs = frame[list(columns)].to_numpy(dtype=float)
    y = frame.outcome.to_numpy(dtype=int)
    one_hot = np.eye(3)[y]
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y).astype(float)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "n": int(len(frame)),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)).mean()),
        "accuracy": float((predicted == y).mean()),
        "ece": float(ece),
    }


def _fit_temperature(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Fit one scalar temperature on past OOS predictions only."""
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    def objective(temperature: float) -> float:
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        calibrated = exp / exp.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(calibrated[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)).mean())
    result = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded")
    return float(result.x) if result.success else 1.0


def _calibrate_prequential(frame: pd.DataFrame, min_history: int = 300,
                           refit_every_dates: int = 10) -> pd.DataFrame:
    if min_history < 1 or refit_every_dates < 1:
        raise ValueError("min_history and refit_every_dates must be positive.")
    out = frame.copy()
    out["date"] = pd.to_datetime(out.date, utc=True, errors="coerce")
    if out.date.isna().any():
        raise ValueError("Invalid prediction date.")
    raw_cols = tuple(f"home_p" if x == "home" else f"draw_p" if x == "draw" else "away_p" for x in OUTCOMES)
    raw = out[list(raw_cols)].to_numpy(dtype=float)
    y = out.outcome.to_numpy(dtype=int)
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError("Raw probabilities must be finite and within [0,1].")
    if not np.allclose(raw.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Raw probabilities must sum to one.")
    out["model_intl_calibrated_home_p"] = np.nan
    out["model_intl_calibrated_draw_p"] = np.nan
    out["model_intl_calibrated_away_p"] = np.nan
    out["calibration_temperature"] = np.nan
    date_groups = out.groupby("date", sort=True).indices
    ordered_dates = list(date_groups.keys())
    history_indices: list[int] = []
    temperature = 1.0
    last_fit_date_index = None
    for date_index, day in enumerate(ordered_dates):
        day_indices = np.asarray(date_groups[day], dtype=int)
        if len(history_indices) >= min_history:
            if last_fit_date_index is None or date_index - last_fit_date_index >= refit_every_dates:
                temperature = _fit_temperature(raw[history_indices], y[history_indices])
                last_fit_date_index = date_index
            logits = np.log(np.clip(raw[day_indices], 1e-12, 1.0)) / temperature
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            calibrated = exp / exp.sum(axis=1, keepdims=True)
            out.loc[day_indices, "model_intl_calibrated_home_p"] = calibrated[:, 0]
            out.loc[day_indices, "model_intl_calibrated_draw_p"] = calibrated[:, 1]
            out.loc[day_indices, "model_intl_calibrated_away_p"] = calibrated[:, 2]
            out.loc[day_indices, "calibration_temperature"] = temperature
        history_indices.extend(day_indices.tolist())
    return out


def _competition_class(value: str) -> str:
    name = str(value).lower()
    if "friendly" in name:
        return "friendly"
    if "qualification" in name or "qualifier" in name:
        return "qualification"
    return "tournament"


def evaluate_international_predictions(predictions: pd.DataFrame,
                                       min_calibration_history: int = 300,
                                       refit_every_dates: int = 10) -> tuple[pd.DataFrame, dict]:
    required = {"date", "home_team", "away_team", "competition", "outcome",
                "home_p", "draw_p", "away_p"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file missing: {sorted(missing)}")
    frame = predictions.copy().reset_index(drop=True)
    frame["competition_class"] = frame.competition.map(_competition_class)
    if "match_type" not in frame:
        frame["match_type"] = frame.competition_class
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="coerce")
    frame = frame.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    calibrated = _calibrate_prequential(frame, min_calibration_history, refit_every_dates)
    raw_cols = ("home_p", "draw_p", "away_p")
    calibrated_cols = tuple(f"model_intl_calibrated_{x}_p" for x in OUTCOMES)
    evaluated = calibrated.dropna(subset=list(calibrated_cols)).copy()
    segment_reports = {}
    for key in ("match_type", "competition_class", "competition"):
        groups = {}
        for value, group in evaluated.groupby(key, dropna=False):
            groups[str(value)] = {
                "raw": _probability_metrics(group, raw_cols),
                "calibrated": _probability_metrics(group, calibrated_cols),
            }
        segment_reports[key] = groups
    summary = {
        "calibration_method": "expanding prequential multiclass temperature scaling",
        "calibration_fit_policy": "each date uses only earlier out-of-sample predictions and outcomes; same-day fixtures share one pre-date temperature",
        "min_calibration_history": min_calibration_history,
        "refit_every_dates": refit_every_dates,
        "input_oos_predictions": int(len(frame)),
        "calibrated_predictions": int(len(evaluated)),
        "overall": {
            "raw": _probability_metrics(evaluated, raw_cols),
            "calibrated": _probability_metrics(evaluated, calibrated_cols),
        },
        "segments": segment_reports,
        "market_testing": {
            "expected_model_prefix": "model_intl_calibrated",
            "market_test_command": "python -m quantlab.market_backtest --predictions <calibrated_csv> --odds <odds_csv> --model-prefix model_intl_calibrated",
            "caveat": "ROI, yield and CLV are only computed for fixtures matched to eligible pre-kickoff and closing odds. No matched bets is not zero ROI.",
        },
        "limitations": [
            "Calibration is fitted only from prior OOS predictions; early rows before minimum history are excluded from calibrated evaluation.",
            "Temperature scaling changes probability sharpness, not model ranking.",
            "Segment results can be noisy for small sample sizes; inspect n and uncertainty before interpretation.",
            "Market-ledger matches are a selected sample, not a full international fixture census.",
        ],
    }
    return calibrated, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and calibrate international OOS predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts/international_v1")
    parser.add_argument("--min-calibration-history", type=int, default=300)
    parser.add_argument("--refit-every-dates", type=int, default=10)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    evaluated, summary = evaluate_international_predictions(
        frame, args.min_calibration_history, args.refit_every_dates
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output / "international_calibrated_predictions.csv", index=False)
    (output / "international_segmented_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
