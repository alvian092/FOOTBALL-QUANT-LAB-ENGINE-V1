"""Out-of-sample calibration and realized bankroll performance metrics."""
from __future__ import annotations
import numpy as np


def _paired(probabilities, outcomes):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0: raise ValueError("Probabilities and outcomes must have equal non-empty shapes.")
    if not np.isfinite(p).all() or not np.isfinite(y).all(): raise ValueError("Inputs must be finite.")
    if ((p < 0) | (p > 1)).any() or ((y != 0) & (y != 1)).any():
        raise ValueError("Probabilities must be in [0,1] and outcomes must be binary.")
    return p, y


def brier_score(probabilities, outcomes):
    p, y = _paired(probabilities, outcomes)
    return float(np.mean((p-y)**2))


def log_loss(probabilities, outcomes, eps=1e-15):
    p, y = _paired(probabilities, outcomes)
    p = np.clip(p, eps, 1-eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def multiclass_brier_score(probability_rows, outcome_indices):
    p, y = np.asarray(probability_rows, dtype=float), np.asarray(outcome_indices, dtype=int)
    if p.ndim != 2 or p.shape[0] != y.size or not p.shape[0]: raise ValueError("Expected an N x K matrix and N outcome indices.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise ValueError("Probabilities must be finite and in [0,1].")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6): raise ValueError("Each probability row must sum to one.")
    if (y < 0).any() or (y >= p.shape[1]).any(): raise ValueError("Outcome index is outside the class range.")
    return float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2, axis=1)))


def calibration_bins(probabilities, outcomes, bins=10):
    p, y = _paired(probabilities, outcomes)
    if bins < 2: raise ValueError("bins must be at least 2.")
    edges, out = np.linspace(0, 1, bins+1), []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1]))
        if mask.any():
            out.append({"count": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                        "observed_rate": float(y[mask].mean()), "lower_bound": float(edges[i]),
                        "upper_bound": float(edges[i+1])})
    return out


def _wilson_interval(successes, n, z=1.959963984540054):
    if not n: return (None, None)
    p = successes/n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return float(max(0, center-half)), float(min(1, center+half))


def backtest_summary(probabilities, outcomes, bootstrap_samples=2000, seed=42):
    """Bootstrap percentile CI for Brier score plus Wilson interval for hit rate."""
    p, y = _paired(probabilities, outcomes)
    if bootstrap_samples < 100: raise ValueError("bootstrap_samples must be at least 100.")
    rng, scores = np.random.default_rng(seed), np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, len(p), len(p))
        scores[i] = np.mean((p[idx]-y[idx])**2)
    lo, hi = np.quantile(scores, [0.025, 0.975])
    hit_lo, hit_hi = _wilson_interval(int(y.sum()), len(y))
    return {"n": int(len(y)), "brier": brier_score(p, y), "brier_ci_low": float(lo),
            "brier_ci_high": float(hi), "log_loss": log_loss(p, y),
            "hit_rate": float(y.mean()), "hit_rate_ci_low": hit_lo, "hit_rate_ci_high": hit_hi}


def bankroll_metrics(profits, stakes=None, entry_odds=None, closing_odds=None):
    pnl = np.asarray(profits, dtype=float)
    if pnl.ndim != 1 or not np.isfinite(pnl).all(): raise ValueError("profits must be a finite one-dimensional sequence.")
    if not pnl.size:
        return {"bets": 0, "net_profit": 0.0, "roi": None, "yield": None,
                "max_drawdown": 0.0, "max_drawdown_pct": None, "clv": None}
    equity = np.r_[0.0, np.cumsum(pnl)]
    peaks = np.maximum.accumulate(equity)
    dd = peaks - equity
    dd_pct = np.divide(dd, peaks, out=np.zeros_like(dd), where=peaks > 0)
    if stakes is None: roi = yield_value = None
    else:
        staked = np.asarray(stakes, dtype=float)
        if staked.shape != pnl.shape or not np.isfinite(staked).all() or (staked < 0).any():
            raise ValueError("stakes must be finite, non-negative, and match profits.")
        total = float(staked.sum())
        roi = float(pnl.sum()/total) if total else None
        yield_value = roi
    clv = None
    if entry_odds is not None and closing_odds is not None:
        entry, close = np.asarray(entry_odds, dtype=float), np.asarray(closing_odds, dtype=float)
        if entry.shape != pnl.shape or close.shape != pnl.shape or (entry <= 1).any() or (close <= 1).any():
            raise ValueError("Entry and closing odds must match bets and be greater than 1.")
        clv = float(np.mean(entry/close - 1.0))
    return {"bets": int(pnl.size), "net_profit": float(pnl.sum()), "roi": roi, "yield": yield_value,
            "max_drawdown": float(dd.max()), "max_drawdown_pct": float(dd_pct.max()), "clv": clv}
