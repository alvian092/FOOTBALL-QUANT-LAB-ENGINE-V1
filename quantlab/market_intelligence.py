"""Historical bookmaker odds snapshots and market movement intelligence."""
from __future__ import annotations
import pandas as pd
from .market import no_vig_probabilities

REQUIRED_ODDS_COLUMNS = {"match_id", "market", "selection", "odds", "captured_at"}


def validate_odds_snapshots(data: pd.DataFrame):
    missing = REQUIRED_ODDS_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing odds snapshot fields: {', '.join(sorted(missing))}")
    df = data.copy()
    df["odds"] = pd.to_numeric(df.odds, errors="coerce")
    df["captured_at"] = pd.to_datetime(df.captured_at, errors="coerce", utc=True)
    if df.odds.isna().any() or (df.odds <= 1).any():
        raise ValueError("All decimal odds must be numeric and greater than 1.")
    if df.captured_at.isna().any():
        raise ValueError("All captured_at values must be valid timestamps.")
    for col in ("match_id", "market", "selection"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"{col} cannot be empty.")
        df[col] = df[col].astype(str).str.strip()
    if "bookmaker" not in df.columns:
        df["bookmaker"] = "Unknown"
    df["bookmaker"] = df.bookmaker.fillna("Unknown").astype(str)
    return df


def summarize_odds_movement(snapshots: pd.DataFrame):
    """Compare first and last complete snapshots per match/market/bookmaker."""
    df = validate_odds_snapshots(snapshots)
    grouping = ["match_id", "market", "bookmaker", "captured_at"]
    complete = []
    for _, snap in df.groupby(grouping, dropna=False):
        odds = dict(zip(snap.selection, snap.odds.astype(float)))
        if len(odds) < 2:
            continue
        try:
            no_vig = no_vig_probabilities(odds)
        except ValueError:
            continue
        for selection, price in odds.items():
            complete.append({"match_id": snap.match_id.iloc[0], "market": snap.market.iloc[0],
                "bookmaker": snap.bookmaker.iloc[0], "captured_at": snap.captured_at.iloc[0],
                "selection": selection, "odds": price, "no_vig_probability": no_vig[selection],
                "overround": sum(1.0/v for v in odds.values())-1.0})
    if not complete:
        return pd.DataFrame(columns=["match_id","market","bookmaker","selection","opening_odds","closing_odds","odds_change_pct","opening_no_vig_probability","closing_no_vig_probability","probability_shift","opening_overround","closing_overround"])
    priced = pd.DataFrame(complete).sort_values("captured_at")
    rows = []
    keys = ["match_id", "market", "bookmaker", "selection"]
    for key, part in priced.groupby(keys, dropna=False):
        first, last = part.iloc[0], part.iloc[-1]
        rows.append(dict(zip(keys, key)) | {
            "opening_odds": float(first.odds), "closing_odds": float(last.odds),
            "odds_change_pct": float(last.odds/first.odds-1),
            "opening_no_vig_probability": float(first.no_vig_probability),
            "closing_no_vig_probability": float(last.no_vig_probability),
            "probability_shift": float(last.no_vig_probability-first.no_vig_probability),
            "opening_overround": float(first.overround), "closing_overround": float(last.overround),
            "snapshots": int(len(part)),
            "movement": "SHORTENED" if last.odds < first.odds else "DRIFTED" if last.odds > first.odds else "UNCHANGED",
        })
    return pd.DataFrame(rows).sort_values(["match_id","market","selection"]).reset_index(drop=True)
