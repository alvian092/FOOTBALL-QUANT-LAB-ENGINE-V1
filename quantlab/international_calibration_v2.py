"""Category-aware, strictly prequential international calibration experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

CLASSES = ("friendly", "qualification", "tournament")
RAW = ("home_p", "draw_p", "away_p")
CAL = ("model_intl_v2_home_p", "model_intl_v2_draw_p", "model_intl_v2_away_p")


def competition_class(value: object) -> str:
    name = str(value).lower()
    if "friendly" in name:
        return "friendly"
    if "qualification" in name or "qualifier" in name:
        return "qualification"
    return "tournament"


def fit_temperature(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))

    def loss(temp: float) -> float:
        z = logits / temp
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(p[np.arange(len(outcomes)), outcomes], 1e-15, 1)).mean())

    result = minimize_scalar(loss, bounds=(0.5, 3.0), method="bounded")
    return float(result.x) if result.success else 1.0


def _apply_temperature(p: np.ndarray, temp: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / temp
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    onehot = np.eye(3)[y]
    return {
        "n": int(len(y)),
        "brier": float(np.square(p - onehot).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


def _date_cluster_ci(frame: pd.DataFrame, probs: np.ndarray, samples: int, seed: int) -> dict:
    if frame.empty or samples < 1:
        return {"brier": None, "log_loss": None}
    work = frame[["date", "outcome"]].copy()
    work["brier"] = np.square(probs - np.eye(3)[work.outcome.to_numpy(dtype=int)]).sum(axis=1)
    work["log_loss"] = -np.log(np.clip(probs[np.arange(len(work)), work.outcome.to_numpy(dtype=int)], 1e-15, 1))
    daily = work.groupby("date")[["brier", "log_loss"]].mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(daily), size=(samples, len(daily)))
    result = {}
    for j, name in enumerate(("brier", "log_loss")):
        boot = daily[draws, j].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        result[name] = [float(lo), float(hi)]
    return result


def _prequential(frame: pd.DataFrame, min_history: int, min_category_history: int,
                 shrinkage_prior: float, refit_every_dates: int) -> tuple[np.ndarray, np.ndarray]:
    raw = frame[list(RAW)].to_numpy(dtype=float)
    y = frame.outcome.to_numpy(dtype=int)
    categories = frame.competition_class.to_numpy()
    dates = frame.date.to_numpy()
    output = np.full_like(raw, np.nan, dtype=float)
    temperatures = {key: 1.0 for key in CLASSES}
    global_temp = 1.0
    last_fit = -10**9
    history: list[int] = []
    by_category = {key: [] for key in CLASSES}
    unique_dates = pd.Index(dates).drop_duplicates().sort_values()
    for date_idx, day in enumerate(unique_dates):
        idx = np.flatnonzero(dates == day)
        if len(history) >= min_history and date_idx - last_fit >= refit_every_dates:
            global_temp = fit_temperature(raw[history], y[history])
            for key in CLASSES:
                cat_history = by_category[key]
                if len(cat_history) >= min_category_history:
                    fitted = fit_temperature(raw[cat_history], y[cat_history])
                    weight = len(cat_history) / (len(cat_history) + shrinkage_prior)
                    temperatures[key] = float(np.exp(weight * np.log(fitted) + (1 - weight) * np.log(global_temp)))
                else:
                    temperatures[key] = global_temp
            last_fit = date_idx
        if len(history) >= min_history:
            for key in CLASSES:
                mask = categories[idx] == key
                if mask.any():
                    output[idx[mask]] = _apply_temperature(raw[idx[mask]], temperatures[key])
        # Update only after every fixture on this date has been predicted.
        history.extend(idx.tolist())
        for key in CLASSES:
            by_category[key].extend(idx[categories[idx] == key].tolist())
    return output, np.array([temperatures[c] for c in categories], dtype=float)


def run_experiments(predictions: pd.DataFrame, bootstrap_samples: int = 2000) -> tuple[pd.DataFrame, dict]:
    required = {"date", "competition", "outcome", *RAW}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    frame = predictions.copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="coerce")
    if frame.date.isna().any():
        raise ValueError("Invalid prediction date.")
    frame["competition_class"] = frame.competition.map(competition_class)
    frame = frame.sort_values(["date"], kind="stable").reset_index(drop=True)
    raw = frame[list(RAW)].to_numpy(dtype=float)
    y = frame.outcome.to_numpy(dtype=int)
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any() or not np.allclose(raw.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must be finite, in [0,1], and sum to one.")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Outcome must be encoded as 0, 1, or 2.")

    configs = [(300, 100, 100), (300, 300, 300), (500, 100, 100), (500, 300, 300)]
    reports = {}
    outputs = {}
    for min_hist, min_cat, shrink in configs:
        key = f"global_min{min_hist}_category_min{min_cat}_shrink{shrink}"
        calibrated, _ = _prequential(frame, min_hist, min_cat, shrink, 30)
        outputs[key] = calibrated
        valid = np.isfinite(calibrated).all(axis=1)
        reports[key] = {"settings": {"min_global_history": min_hist, "min_category_history": min_cat, "shrinkage_prior": shrink},
                        "n": int(valid.sum()), "overall": _metrics(calibrated[valid], y[valid])}

    # The conservative, pre-declared default is not selected on the evaluation outcomes.
    default_key = "global_min500_category_min300_shrink300"
    chosen = outputs[default_key]
    frame[list(CAL)] = chosen
    valid = np.isfinite(chosen).all(axis=1)
    eval_frame = frame.loc[valid].copy()
    eval_probs = chosen[valid]
    eval_y = y[valid]
    temporal = {}
    for year, group in eval_frame.groupby(eval_frame.date.dt.year):
        positions = np.flatnonzero(eval_frame.date.dt.year.to_numpy() == year)
        temporal[str(year)] = _metrics(eval_probs[positions], eval_y[positions])
    segments = {}
    for category in CLASSES:
        mask = eval_frame.competition_class.to_numpy() == category
        segments[category] = {
            "raw": _metrics(raw[valid][mask], eval_y[mask]),
            "calibrated": _metrics(eval_probs[mask], eval_y[mask]),
            "n": int(mask.sum()),
        }
    summary = {
        "method": "strict-date prequential category temperature with log-space shrinkage to global temperature",
        "default_configuration": {"min_global_history": 500, "min_category_history": 300, "shrinkage_prior": 300, "refit_every_dates": 30},
        "default_configuration_predeclared": True,
        "calibrated_rows": int(valid.sum()),
        "segments": segments,
        "overall": {"raw": _metrics(raw[valid], eval_y), "calibrated": _metrics(eval_probs, eval_y),
                    "calibrated_cluster_bootstrap_ci_95": _date_cluster_ci(eval_frame, eval_probs, bootstrap_samples, 20261115)},
        "by_calendar_year": temporal,
        "sensitivity_grid": reports,
        "caveats": [
            "All calibration fits use only outcomes from strictly earlier dates; same-day fixtures share pre-date parameters.",
            "The default is pre-declared, not chosen by minimizing metrics on this evaluation period.",
            "Sensitivity-grid metrics are diagnostic only and must not be used to select and report an in-sample winner.",
            "Date-cluster intervals account for same-day fixture clustering but not all serial dependence or model-selection uncertainty."
        ]
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="International category calibration sensitivity and temporal validation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts/international_v2")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    predictions, summary = run_experiments(pd.read_csv(args.input), args.bootstrap_samples)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "international_v2_calibrated_predictions.csv", index=False)
    (out / "international_v2_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
