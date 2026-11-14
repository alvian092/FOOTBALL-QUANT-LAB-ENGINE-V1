"""Out-of-sample prediction and bankroll performance metrics."""
from __future__ import annotations
import numpy as np


def _paired(probabilities, outcomes):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("Probabilities and outcomes must have equal non-empty shapes.")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite.")
    if ((p < 0) | (p > 1)).any() or ((y != 0) & (y != 1)).any():
        raise ValueError("Probabilities must be in [0,1] and outcomes must be binary.")
    return p, y


def brier_score(probabilities, outcomes) -> float:
    p, y = _paired(probabilities, outcomes)
    return float(np.mean((p-y)**2))


def log_loss(probabilities, outcomes, eps: float = 1e-15) -> float:
    p, y = _paired(probabilities, outcomes)
    p = np.clip(p, eps, 1-eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def multiclass_brier_score(probability_rows, outcome_indices) -> float:
    p = np.asarray(probability_rows, dtype=float)
    y = np.asarray(outcome_indices, dtype=int)
    if p.ndim != 2 or p.shape[0] != y.size or p.shape[0] == 0:
        raise ValueError("Expected an N x K probability matrix and N outcome indices.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1].")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each multiclass probability row must sum to one.")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Outcome index is outside the class range.")
    one_hot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p-one_hot)**2, axis=1)))


def calibration_bins(probabilities, outcomes, bins: int = 10) -> list[dict[str, float | int]]:
    p, y = _paired(probabilities, outcomes)
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    edges = np.linspace(0, 1, bins+1)
    result = []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1]))
        if mask.any():
            result.append({"count": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                           "observed_rate": float(y[mask].mean()),
                           "lower_bound": float(edges[i]), "upper_bound": float(edges[i+1])})
    return result


def bankroll_metrics(profits, stakes=None) -> dict[str, float | int | None]:
    pnl = np.asarray(profits, dtype=float)
    if pnl.ndim != 1 or not np.isfinite(pnl).all():
        raise ValueError("profits must be a finite one-dimensional sequence.")
    if pnl.size == 0:
        return {"bets": 0, "net_profit": 0.0, "roi": None, "yield": None, "max_drawdown": 0.0}
    cumulative = np.cumsum(pnl)
    equity = np.r_[0.0, cumulative]
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    if stakes is None:
        roi = yield_value = None
    else:
        staked = np.asarray(stakes, dtype=float)
        if staked.shape != pnl.shape or not np.isfinite(staked).all() or (staked < 0).any():
            raise ValueError("stakes must be finite, non-negative, and match profits.")
        total_staked = float(staked.sum())
        roi = float(pnl.sum()/total_staked) if total_staked else None
        yield_value = roi
    return {"bets": int(pnl.size), "net_profit": float(pnl.sum()), "roi": roi,
            "yield": yield_value, "max_drawdown": float(drawdown.max())}
