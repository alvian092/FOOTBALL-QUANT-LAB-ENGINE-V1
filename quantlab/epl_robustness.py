"""Paired matchday-cluster bootstrap and season robustness for EPL OOS predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")


def _losses(probabilities: np.ndarray, outcomes: np.ndarray):
    if probabilities.shape != (len(outcomes), 3):
        raise ValueError("Each model must provide an Nx3 probability matrix.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    one_hot = np.eye(3)[outcomes]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    log_loss = -np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-15, 1.0))
    accuracy = (probabilities.argmax(axis=1) == outcomes).astype(float)
    return brier, log_loss, accuracy


def paired_cluster_bootstrap(
    left_losses,
    right_losses,
    dates: pd.Series,
    samples: int = 20_000,
    seed: int = 20261114,
    left_name: str = "Elo",
    right_name: str = "Ensemble",
) -> dict:
    """Bootstrap paired left-minus-right loss differences by match date."""
    if samples < 1000:
        raise ValueError("Use at least 1,000 bootstrap samples.")
    if (len(left_losses[0]) != len(right_losses[0])
            or len(left_losses[1]) != len(right_losses[1])
            or len(left_losses[0]) != len(dates) or not len(dates)):
        raise ValueError("Loss arrays and dates must have equal, non-zero lengths.")
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates, utc=True).dt.strftime("%Y-%m-%d"),
        "brier_diff": np.asarray(left_losses[0]) - np.asarray(right_losses[0]),
        "log_loss_diff": np.asarray(left_losses[1]) - np.asarray(right_losses[1]),
    })
    grouped = frame.groupby("date", sort=True).agg(
        brier_sum=("brier_diff", "sum"),
        log_loss_sum=("log_loss_diff", "sum"),
        n=("brier_diff", "size"),
    )
    sums = grouped[["brier_sum", "log_loss_sum"]].to_numpy(dtype=float)
    counts = grouped.n.to_numpy(dtype=float)
    observed = sums.sum(axis=0) / counts.sum()
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, 2), dtype=float)
    for i in range(samples):
        selected = rng.integers(0, len(counts), size=len(counts))
        boot[i] = sums[selected].sum(axis=0) / counts[selected].sum()

    metrics = {}
    for j, metric_name in enumerate(("brier", "log_loss")):
        low, high = np.quantile(boot[:, j], [0.025, 0.975])
        probability_nonnegative = (np.count_nonzero(boot[:, j] >= 0) + 1) / (samples + 1)
        metrics[metric_name] = {
            "left_minus_right": float(observed[j]),
            "ci_95_low": float(low),
            "ci_95_high": float(high),
            "bootstrap_probability_nonnegative": float(probability_nonnegative),
            "left_better": bool(high < 0),
        }
    return {
        "method": f"paired percentile bootstrap; calendar-date clusters; {left_name} minus {right_name}",
        "left_model": left_name,
        "right_model": right_name,
        "samples": int(samples),
        "seed": int(seed),
        "clusters": int(len(counts)),
        "fixtures": int(counts.sum()),
        "metrics": metrics,
        "interpretation": f"Negative differences favor {left_name}. This is a cluster-bootstrap uncertainty interval, not a formal null-hypothesis p-value or evidence of betting profitability.",
    }

def evaluate_predictions(predictions: pd.DataFrame, samples: int = 20_000,
                         seed: int = 20261114):
    required = {"date", "outcome"}
    for model in ("elo", "ensemble"):
        required.update(f"{model}_{outcome}_p" for outcome in OUTCOMES)
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = predictions.copy()
    df["date"] = pd.to_datetime(df.date, errors="coerce", utc=True)
    if df.date.isna().any():
        raise ValueError("Invalid prediction date found.")
    if not df.outcome.isin([0, 1, 2]).all():
        raise ValueError("Outcome must be encoded as 0=home, 1=draw, 2=away.")
    outcomes = df.outcome.to_numpy(dtype=int)
    elo_p = df[[f"elo_{x}_p" for x in OUTCOMES]].to_numpy(dtype=float)
    ensemble_p = df[[f"ensemble_{x}_p" for x in OUTCOMES]].to_numpy(dtype=float)
    elo_losses = _losses(elo_p, outcomes)
    ensemble_losses = _losses(ensemble_p, outcomes)
    paired = paired_cluster_bootstrap(elo_losses[:2], ensemble_losses[:2], df.date,
                                      samples=samples, seed=seed)

    years = np.where(df.date.dt.month >= 7, df.date.dt.year, df.date.dt.year - 1)
    df["season"] = years.astype(str) + "-" + (years + 1).astype(str)
    rows = []
    for season, group in df.groupby("season", sort=True):
        y = group.outcome.to_numpy(dtype=int)
        for model in ("elo", "ensemble"):
            p = group[[f"{model}_{x}_p" for x in OUTCOMES]].to_numpy(dtype=float)
            brier, logloss, accuracy = _losses(p, y)
            rows.append({
                "season": season, "model": model, "matches": len(group),
                "brier": float(brier.mean()), "log_loss": float(logloss.mean()),
                "accuracy": float(accuracy.mean()),
            })
    season_metrics = pd.DataFrame(rows)
    elo = season_metrics[season_metrics.model == "elo"].set_index("season")
    ensemble = season_metrics[season_metrics.model == "ensemble"].set_index("season")
    brier_wins = int((elo.brier < ensemble.brier).sum())
    logloss_wins = int((elo.log_loss < ensemble.log_loss).sum())
    accuracy_wins = int((elo.accuracy > ensemble.accuracy).sum())
    n_seasons = int(df.season.nunique())
    summary = {
        "fixtures": int(len(df)),
        "seasons": n_seasons,
        "season_range": [str(df.season.min()), str(df.season.max())],
        "paired_bootstrap": paired,
        "season_robustness": {
            "elo_brier_wins": brier_wins,
            "elo_log_loss_wins": logloss_wins,
            "elo_accuracy_wins": accuracy_wins,
            "seasons_compared": n_seasons,
            "brier_win_share": float(brier_wins / n_seasons),
            "note": "Season comparisons are descriptive; no multiple-testing adjustment is applied.",
        },
    }
    return season_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired Elo/Ensemble robustness on EPL OOS predictions.")
    parser.add_argument("--input", default="artifacts/epl_v4_backtest/epl_v4_oos_predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/epl_v4_robustness")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    predictions = pd.read_csv(args.input)
    season_metrics, summary = evaluate_predictions(predictions, args.bootstrap_samples, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    season_metrics.to_csv(output / "epl_v4_season_robustness.csv", index=False)
    (output / "epl_v4_paired_bootstrap.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
