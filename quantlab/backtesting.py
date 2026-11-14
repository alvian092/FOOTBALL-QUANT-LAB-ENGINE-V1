"""Leakage-safe chronological backtesting and model-vs-market evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .market import no_vig_probabilities

CLASS_ORDER = ("home", "draw", "away")
ODDS_COLUMNS = ("home_odds", "draw_odds", "away_odds")


def _probability_vector(value):
    if not isinstance(value, dict) or set(CLASS_ORDER) - set(value):
        raise ValueError("Predictor must return home, draw and away probabilities.")
    p = np.asarray([value[k] for k in CLASS_ORDER], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1, atol=1e-6):
        raise ValueError("Predicted probabilities must be finite, in [0,1], and sum to one.")
    return p


def _scores(prob_rows, outcomes):
    p = np.asarray(prob_rows, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if not len(y):
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    onehot = np.eye(3)[y]
    return {
        "n": int(len(y)),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)))),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
    }


def _bootstrap_mean_ci(values, samples=2000, seed=42):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return (None, None)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def walk_forward_backtest(matches: pd.DataFrame, predictor, min_train=100,
                          test_size=1, max_train=None, bootstrap_samples=2000):
    """Call predictor(train_history, fixture) using only strictly earlier dates.

    Required match fields: date, home_team, away_team, home_goals, away_goals.
    Optional bookmaker benchmark: home_odds, draw_odds, away_odds.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {', '.join(sorted(missing))}")
    if min_train < 1 or test_size < 1 or (max_train is not None and max_train < min_train):
        raise ValueError("Invalid train/test window settings.")
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if df["date"].isna().any():
        raise ValueError("All backtest dates must be valid.")
    df = df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    records = []
    for _, fixture in df.iterrows():
        # Strictly earlier dates: same-day results cannot leak into one another.
        history = df[df.date < fixture.date]
        if max_train is not None:
            history = history.tail(max_train)
        if len(history) < min_train:
            continue
        model_p = _probability_vector(predictor(history.copy(), fixture.copy()))
        hg, ag = int(fixture.home_goals), int(fixture.away_goals)
        outcome = 0 if hg > ag else 1 if hg == ag else 2
        row = {
            "date": fixture.date, "match_id": fixture.get("match_id", None),
            "league": fixture.get("league", "Unknown"),
            "home_team": fixture.home_team, "away_team": fixture.away_team,
            "actual_home_goals": hg, "actual_away_goals": ag, "outcome": outcome,
            "model_home_p": model_p[0], "model_draw_p": model_p[1], "model_away_p": model_p[2],
        }
        if set(ODDS_COLUMNS).issubset(df.columns):
            odds = [fixture[c] for c in ODDS_COLUMNS]
            try:
                if all(pd.notna(x) and float(x) > 1 for x in odds):
                    market = no_vig_probabilities(dict(zip(CLASS_ORDER, map(float, odds))))
                    row.update({f"market_{k}_p": market[k] for k in CLASS_ORDER})
            except (TypeError, ValueError):
                pass
        records.append(row)
    results = pd.DataFrame(records)
    if results.empty:
        return {"results": results, "model": _scores([], []), "market": _scores([], []),
                "model_brier_ci": (None, None), "market_brier_ci": (None, None),
                "model_edge_vs_market_brier": None, "model_edge_vs_market_log_loss": None}
    model_p = results[[f"model_{k}_p" for k in CLASS_ORDER]].to_numpy()
    outcomes = results.outcome.to_numpy(dtype=int)
    model = _scores(model_p, outcomes)
    model_loss = np.sum((model_p - np.eye(3)[outcomes])**2, axis=1)
    model_ci = _bootstrap_mean_ci(model_loss, bootstrap_samples)
    market_cols = [f"market_{k}_p" for k in CLASS_ORDER]
    market_rows = results[market_cols].notna().all(axis=1) if set(market_cols).issubset(results.columns) else pd.Series(False, index=results.index)
    if market_rows.any():
        market_p = results.loc[market_rows, market_cols].to_numpy()
        market_y = results.loc[market_rows, "outcome"].to_numpy(dtype=int)
        market = _scores(market_p, market_y)
        market_loss = np.sum((market_p - np.eye(3)[market_y])**2, axis=1)
        market_ci = _bootstrap_mean_ci(market_loss, bootstrap_samples, seed=43)
        model_common = results.loc[market_rows, [f"model_{k}_p" for k in CLASS_ORDER]].to_numpy()
        model_common_scores = _scores(model_common, market_y)
        model_common_brier = model_common_scores["brier"]
        model_common_ll = model_common_scores["log_loss"]
        edge_brier = market["brier"] - model_common_brier
        edge_ll = market["log_loss"] - model_common_ll
    else:
        market, market_ci, edge_brier, edge_ll = _scores([], []), (None, None), None, None
    return {"results": results, "model": model, "market": market,
            "model_brier_ci": model_ci, "market_brier_ci": market_ci,
            "model_edge_vs_market_brier": edge_brier,
            "model_edge_vs_market_log_loss": edge_ll}


def segmented_performance(records: pd.DataFrame, group_by=("league", "selected_market"),
                          min_bets=30):
    """Summarize settled bet records by segment; small samples are flagged, not hidden."""
    required = {"simulated_profit", "stake"}
    if not required.issubset(records.columns):
        raise ValueError("Records need simulated_profit and stake columns.")
    groups = [group_by] if isinstance(group_by, str) else list(group_by)
    if not groups or not set(groups).issubset(records.columns):
        raise ValueError("All segment columns must exist.")
    rows = []
    for key, part in records.groupby(groups, dropna=False):
        pnl = pd.to_numeric(part.simulated_profit, errors="coerce").to_numpy(dtype=float)
        stakes = pd.to_numeric(part.stake, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(pnl) & np.isfinite(stakes) & (stakes >= 0)
        pnl, stakes = pnl[valid], stakes[valid]
        total_stake = float(stakes.sum())
        item = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        item.update({"bets": int(len(pnl)), "net_profit": float(pnl.sum()),
                     "total_stake": total_stake,
                     "roi": float(pnl.sum()/total_stake) if total_stake else None,
                     "max_drawdown": float(np.max(np.maximum.accumulate(np.r_[0, np.cumsum(pnl)]) - np.r_[0, np.cumsum(pnl)])) if len(pnl) else 0.0,
                     "sample_status": "OK" if len(pnl) >= min_bets else "SMALL SAMPLE"})
        if {"odds", "closing_odds"}.issubset(part.columns):
            odds = pd.to_numeric(part.odds, errors="coerce")
            close = pd.to_numeric(part.closing_odds, errors="coerce")
            valid_clv = odds.notna() & close.notna() & (odds > 1) & (close > 1)
            item["avg_clv"] = float((odds[valid_clv]/close[valid_clv]-1).mean()) if valid_clv.any() else None
        rows.append(item)
    return pd.DataFrame(rows).sort_values("net_profit", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
