"""CSV ingestion and validation helpers."""
from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}


def validate_matches_csv(file_or_buffer) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(file_or_buffer)
    errors = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return df, [f"Missing required columns: {', '.join(sorted(missing))}"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if df["date"].isna().any():
        errors.append("date contains invalid values.")
    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].isna() | (df[col] < 0) | (df[col] % 1 != 0)
        if invalid.any():
            errors.append(f"{col} must contain non-negative whole-number goal counts.")
    for col in ("home_team", "away_team"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            errors.append(f"{col} cannot contain empty team names.")
        df[col] = df[col].astype("string").str.strip()
    if not errors and (df["home_team"] == df["away_team"]).any():
        errors.append("A team cannot play itself.")
    if "match_id" not in df.columns:
        df["match_id"] = (
            df["date"].astype(str) + "|" + df["home_team"].astype(str) + "|" + df["away_team"].astype(str)
        )
    df["match_id"] = df["match_id"].astype("string").str.strip()
    if df["match_id"].isna().any() or (df["match_id"] == "").any():
        errors.append("match_id cannot be empty.")
    if df["match_id"].duplicated().any():
        errors.append("Duplicate match_id values found in uploaded file.")
    return df, errors


def recent_team_stats(matches: pd.DataFrame, team: str, limit: int = 20) -> dict[str, float]:
    """Return recent per-match scoring/conceding rates from completed matches."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    rows = matches[(matches.home_team == team) | (matches.away_team == team)].copy()
    rows = rows.sort_values("date", ascending=False).head(limit)
    if rows.empty:
        raise ValueError(f"No historical matches found for {team}.")
    scored, conceded = [], []
    for row in rows.itertuples():
        if row.home_team == team:
            scored.append(float(row.home_goals)); conceded.append(float(row.away_goals))
        else:
            scored.append(float(row.away_goals)); conceded.append(float(row.home_goals))
    return {"matches": float(len(rows)), "scored_per_match": sum(scored)/len(scored),
            "conceded_per_match": sum(conceded)/len(conceded)}
