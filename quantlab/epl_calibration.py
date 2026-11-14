"""Leakage-safe expanding-window calibration and OOF ensemble weighting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .epl_robustness import paired_cluster_bootstrap

BASE_MODELS = ("poisson", "dixon_coles", "elo")
OUTCOMES = ("home", "draw", "away")
DEFAULT_WEIGHTS = np.array([0.4, 0.4, 0.2], dtype=float)


def _softmax_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    if len(outcomes) < 2:
        return 1.0

    def objective(log_temperature: float) -> float:
        calibrated = _softmax_temperature(probabilities, np.exp(log_temperature))
        return float(-np.log(np.clip(
            calibrated[np.arange(len(outcomes)), outcomes], 1e-15, 1.0
        )).mean())

    fit = minimize_scalar(
        objective, bounds=(-2.302585093, 2.302585093), method="bounded",
        options={"xatol": 1e-5},
    )
    return float(np.exp(fit.x)) if fit.success else 1.0


def _fit_weights(base_probabilities: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    if len(outcomes) < 3:
        return DEFAULT_WEIGHTS.copy()

    def objective(weights: np.ndarray) -> float:
        mixture = np.einsum("m,nmk->nk", weights, base_probabilities)
        return float(-np.log(np.clip(
            mixture[np.arange(len(outcomes)), outcomes], 1e-15, 1.0
        )).mean())

    fit = minimize(
        objective, DEFAULT_WEIGHTS.copy(), method="SLSQP",
        bounds=[(0.0, 1.0)] * len(BASE_MODELS),
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"maxiter": 100, "ftol": 1e-10},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        return DEFAULT_WEIGHTS.copy()
    weights = np.clip(fit.x, 0.0, 1.0)
    return weights / weights.sum()


def _metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    one_hot = np.eye(3)[outcomes]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    log_loss = -np.log(np.clip(
        probabilities[np.arange(len(outcomes)), outcomes], 1e-15, 1.0
    )).mean()
    accuracy = (probabilities.argmax(axis=1) == outcomes).mean()
    classwise_ece = []
    for outcome_index in range(3):
        ece = 0.0
        for lower in np.linspace(0.0, 1.0, 11)[:-1]:
            upper = lower + 0.1
            in_bin = (probabilities[:, outcome_index] >= lower) & (
                (probabilities[:, outcome_index] < upper)
                if upper < 1.0 else (probabilities[:, outcome_index] <= upper)
            )
            if in_bin.any():
                ece += in_bin.mean() * abs(
                    probabilities[in_bin, outcome_index].mean()
                    - one_hot[in_bin, outcome_index].mean()
                )
        classwise_ece.append(ece)
    return {
        "n": int(len(outcomes)),
        "brier": float(brier.mean()),
        "log_loss": float(log_loss),
        "accuracy": float(accuracy),
        "classwise_ece_mean": float(np.mean(classwise_ece)),
    }


def run_expanding_calibration(
    predictions: pd.DataFrame,
    min_history: int = 300,
    refit_every_dates: int = 10,
    bootstrap_samples: int = 20_000,
    seed: int = 20261114,
):
    """Fit all calibration/weight parameters using strictly earlier OOS dates only."""
    required = {"date", "outcome"}
    for model in (*BASE_MODELS, "ensemble"):
        required.update(f"{model}_{outcome}_p" for outcome in OUTCOMES)
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if min_history < 1 or refit_every_dates < 1:
        raise ValueError("min_history and refit_every_dates must be positive.")

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="coerce")
    if frame.date.isna().any() or not frame.outcome.isin([0, 1, 2]).all():
        raise ValueError("Invalid date or outcome encoding.")
    frame = frame.sort_values(["date"], kind="stable").reset_index(drop=True)
    outcomes = frame.outcome.to_numpy(dtype=int)

    raw = {
        model: frame[[f"{model}_{outcome}_p" for outcome in OUTCOMES]].to_numpy(float)
        for model in BASE_MODELS
    }
    for model, probabilities in raw.items():
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)):
            raise ValueError(f"{model} probabilities must be valid and sum to one.")

    fixed = frame[[f"ensemble_{outcome}_p" for outcome in OUTCOMES]].to_numpy(float)
    if (not np.isfinite(fixed).all() or (fixed < 0).any() or (fixed > 1).any()
            or not np.allclose(fixed.sum(axis=1), 1.0, atol=1e-6)):
        raise ValueError("Fixed ensemble probabilities must be valid and sum to one.")

    calibrated = {model: np.empty_like(raw[model]) for model in BASE_MODELS}
    adaptive_raw = np.empty((len(frame), 3))
    adaptive_calibrated = np.empty((len(frame), 3))
    date_groups = list(frame.groupby("date", sort=True).indices.values())
    temperatures = {model: 1.0 for model in BASE_MODELS}
    ensemble_temperature = 1.0
    weights = DEFAULT_WEIGHTS.copy()
    history_fixtures = 0

    for date_index, indices in enumerate(date_groups):
        first_row = int(indices[0])
        # No outcomes from this date or any later date are visible to fitting.
        if history_fixtures >= min_history and date_index % refit_every_dates == 0:
            for model in BASE_MODELS:
                temperatures[model] = _fit_temperature(raw[model][:first_row], outcomes[:first_row])
            historical_calibrated = np.stack(
                [calibrated[model][:first_row] for model in BASE_MODELS], axis=1
            )
            weights = _fit_weights(historical_calibrated, outcomes[:first_row])
            ensemble_temperature = _fit_temperature(
                adaptive_raw[:first_row], outcomes[:first_row]
            )

        for model in BASE_MODELS:
            calibrated[model][indices] = _softmax_temperature(
                raw[model][indices], temperatures[model]
            )
        current_stack = np.stack(
            [calibrated[model][indices] for model in BASE_MODELS], axis=1
        )
        adaptive_raw[indices] = np.einsum("m,nmk->nk", weights, current_stack)
        adaptive_calibrated[indices] = _softmax_temperature(
            adaptive_raw[indices], ensemble_temperature
        )
        history_fixtures = int(indices[-1]) + 1

    output = frame[["date", "outcome"]].copy()
    for model in BASE_MODELS:
        output[[f"{model}_cal_{outcome}_p" for outcome in OUTCOMES]] = calibrated[model]
    output[[f"adaptive_ensemble_raw_{outcome}_p" for outcome in OUTCOMES]] = adaptive_raw
    output[[f"adaptive_ensemble_cal_{outcome}_p" for outcome in OUTCOMES]] = adaptive_calibrated

    # Record the as-of parameters for each date; these rows use only prior-date outcomes.
    parameter_rows = []
    temperatures = {model: 1.0 for model in BASE_MODELS}
    ensemble_temperature = 1.0
    weights = DEFAULT_WEIGHTS.copy()
    history_fixtures = 0
    for date_index, indices in enumerate(date_groups):
        first_row = int(indices[0])
        if history_fixtures >= min_history and date_index % refit_every_dates == 0:
            for model in BASE_MODELS:
                temperatures[model] = _fit_temperature(raw[model][:first_row], outcomes[:first_row])
            historical_calibrated = np.stack(
                [calibrated[model][:first_row] for model in BASE_MODELS], axis=1
            )
            weights = _fit_weights(historical_calibrated, outcomes[:first_row])
            ensemble_temperature = _fit_temperature(
                adaptive_raw[:first_row], outcomes[:first_row]
            )
        parameter_rows.append({
            "date": str(frame.date.iloc[first_row]),
            "history_fixtures": first_row,
            **{f"temperature_{model}": temperatures[model] for model in BASE_MODELS},
            "ensemble_temperature": ensemble_temperature,
            **{f"weight_{model}": float(weights[i]) for i, model in enumerate(BASE_MODELS)},
        })
        history_fixtures = int(indices[-1]) + 1
    parameters = pd.DataFrame(parameter_rows)

    metrics = {
        **{f"raw_{model}": _metrics(raw[model], outcomes) for model in BASE_MODELS},
        **{f"calibrated_{model}": _metrics(calibrated[model], outcomes) for model in BASE_MODELS},
        "fixed_ensemble": _metrics(fixed, outcomes),
        "adaptive_ensemble_raw": _metrics(adaptive_raw, outcomes),
        "adaptive_ensemble_calibrated": _metrics(adaptive_calibrated, outcomes),
    }

    def loss_pair(probabilities: np.ndarray):
        one_hot = np.eye(3)[outcomes]
        return (
            np.square(probabilities - one_hot).sum(axis=1),
            -np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)),
        )

    comparisons = {
        "adaptive_raw_vs_raw_elo": ("Adaptive OOF Ensemble (raw)", adaptive_raw, "Raw Elo", raw["elo"]),
        "calibrated_elo_vs_raw_elo": ("Calibrated Elo", calibrated["elo"], "Raw Elo", raw["elo"]),
        "adaptive_calibrated_vs_raw_elo": (
            "Adaptive OOF Ensemble (calibrated)", adaptive_calibrated, "Raw Elo", raw["elo"]
        ),
        "adaptive_raw_vs_fixed_ensemble": (
            "Adaptive OOF Ensemble (raw)", adaptive_raw, "Fixed V4 Ensemble", fixed
        ),
    }
    paired = {}
    for name, (left_name, left, right_name, right) in comparisons.items():
        paired[name] = paired_cluster_bootstrap(
            loss_pair(left), loss_pair(right), frame.date,
            samples=bootstrap_samples, seed=seed,
            left_name=left_name, right_name=right_name,
        )
    return output, parameters, {"metrics": metrics, "paired_comparisons": paired}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe expanding-window EPL calibration and OOF weights"
    )
    parser.add_argument("--input", default="artifacts/epl_v4_backtest/epl_v4_oos_predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/epl_v4_calibration")
    parser.add_argument("--min-history", type=int, default=300)
    parser.add_argument("--refit-every-dates", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()

    predictions = pd.read_csv(args.input)
    output, parameters, summary = run_expanding_calibration(
        predictions, args.min_history, args.refit_every_dates,
        args.bootstrap_samples, args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "epl_v4_calibrated_oos_predictions.csv", index=False)
    parameters.to_csv(output_dir / "epl_v4_calibration_parameters.csv", index=False)
    summary.update({
        "min_history": args.min_history,
        "refit_every_dates": args.refit_every_dates,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    })
    (output_dir / "epl_v4_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
