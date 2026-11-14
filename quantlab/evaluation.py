"""Prediction and bankroll performance metrics."""
from __future__ import annotations
import numpy as np


def brier_score(probabilities, outcomes) -> float:
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("Probabilities and outcomes must have equal non-empty shapes.")
    return float(np.mean((p-y)**2))


def log_loss(probabilities, outcomes, eps: float = 1e-15) -> float:
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("Probabilities and outcomes must have equal non-empty shapes.")
    p = np.clip(p, eps, 1-eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def bankroll_metrics(profits) -> dict[str, float]:
    values = np.asarray(profits, dtype=float)
    if values.size == 0:
        return {"bets": 0, "roi": 0.0, "yield": 0.0, "max_drawdown": 0.0}
    cumulative = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    drawdown = peaks - cumulative
    total_staked = float(np.sum(np.abs(values[values < 0])))  # conservative proxy; use explicit stakes for true yield
    return {"bets": int(values.size), "net_profit": float(values.sum()),
            "roi": float(values.sum()/total_staked) if total_staked else 0.0,
            "yield": float(values.mean()), "max_drawdown": float(drawdown.max())}
