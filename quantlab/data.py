"""Validated match ingestion and venue/recency-aware team features."""
from __future__ import annotations
import math
import pandas as pd

REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}
XG_COLUMNS = {"home_xg", "away_xg"}


def validate_matches_csv(file_or_buffer) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(file_or_buffer)
    errors = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return df, [f"Missing required columns: {', '.join(sorted(missing))}"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if df["date"].isna().any(): errors.append("date contains invalid values.")
    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if (df[col].isna() | (df[col] < 0) | (df[col] % 1 != 0)).any():
            errors.append(f"{col} must contain non-negative whole-number goal counts.")
    for col in ("home_team", "away_team"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            errors.append(f"{col} cannot contain empty team names.")
        df[col] = df[col].astype("string").str.strip()
    if not errors and (df.home_team == df.away_team).any():
        errors.append("A team cannot play itself.")
    if XG_COLUMNS.intersection(df.columns) and not XG_COLUMNS.issubset(df.columns):
        errors.append("Provide both home_xg and away_xg, or neither.")
    if XG_COLUMNS.issubset(df.columns):
        for col in XG_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any() or (df[col] < 0).any():
                errors.append(f"{col} must contain non-negative numeric xG values.")
    if "match_id" not in df.columns:
        df["match_id"] = df.date.astype(str) + "|" + df.home_team.astype(str) + "|" + df.away_team.astype(str)
    df["match_id"] = df.match_id.astype("string").str.strip()
    if df.match_id.isna().any() or (df.match_id == "").any(): errors.append("match_id cannot be empty.")
    if df.match_id.duplicated().any(): errors.append("Duplicate match_id values found in uploaded file.")
    return df, errors


def _weighted_mean(values, half_life: float) -> float:
    if half_life <= 0: raise ValueError("half_life must be positive.")
    vals = [float(v) for v in values]
    weights = [math.exp(-math.log(2) * age / half_life) for age in range(len(vals))]
    return float(sum(v*w for v, w in zip(vals, weights)) / sum(weights))


def team_form_stats(matches: pd.DataFrame, team: str, limit: int = 10,
                    half_life: float = 5.0) -> dict:
    """Latest N matches; recency weights use an exponential half-life in matches."""
    if not 1 <= limit <= 20: raise ValueError("limit must be between 1 and 20.")
    rows = matches[(matches.home_team == team) | (matches.away_team == team)].copy()
    rows = rows.sort_values("date", ascending=False).head(limit)
    if rows.empty: raise ValueError(f"No historical matches found for {team}.")
    gf, ga, xgf, xga = [], [], [], []
    has_xg = XG_COLUMNS.issubset(rows.columns)
    for r in rows.itertuples():
        is_home = r.home_team == team
        gf.append(r.home_goals if is_home else r.away_goals)
        ga.append(r.away_goals if is_home else r.home_goals)
        if has_xg:
            xgf.append(r.home_xg if is_home else r.away_xg)
            xga.append(r.away_xg if is_home else r.home_xg)
    return {"matches": len(rows), "scored_per_match": _weighted_mean(gf, half_life),
            "conceded_per_match": _weighted_mean(ga, half_life),
            "home_matches": int((rows.home_team == team).sum()),
            "away_matches": int((rows.away_team == team).sum()),
            "xg_for": _weighted_mean(xgf, half_life) if has_xg else None,
            "xg_against": _weighted_mean(xga, half_life) if has_xg else None}


def venue_stats(matches: pd.DataFrame, team: str, venue: str, limit: int = 20,
                half_life: float = 5.0) -> dict:
    """Weighted team stats restricted to home or away fixtures."""
    if venue not in {"home", "away"}: raise ValueError("venue must be 'home' or 'away'.")
    team_col = "home_team" if venue == "home" else "away_team"
    rows = matches[matches[team_col] == team].sort_values("date", ascending=False).head(limit)
    if rows.empty: raise ValueError(f"No {venue}-venue history found for {team}.")
    gf, ga = ("home_goals", "away_goals") if venue == "home" else ("away_goals", "home_goals")
    xgf, xga = ("home_xg", "away_xg") if venue == "home" else ("away_xg", "home_xg")
    has_xg = XG_COLUMNS.issubset(rows.columns)
    return {"matches": len(rows), "scored_per_match": _weighted_mean(rows[gf], half_life),
            "conceded_per_match": _weighted_mean(rows[ga], half_life),
            "xg_for": _weighted_mean(rows[xgf], half_life) if has_xg else None,
            "xg_against": _weighted_mean(rows[xga], half_life) if has_xg else None}


def recent_team_stats(matches: pd.DataFrame, team: str, limit: int = 20) -> dict[str, float]:
    """Backward-compatible unweighted recent scoring/conceding rates."""
    s = team_form_stats(matches, team, limit, half_life=1e12)
    return {"matches": float(s["matches"]), "scored_per_match": float(s["scored_per_match"]),
            "conceded_per_match": float(s["conceded_per_match"])}
