"""Senior men's international results pipeline and leakage-safe Elo baseline.

Source: openfootball/internationals (CC0-1.0 mirror of historical men's full
international results). This is community-maintained data, not an official FIFA
feed. Friendly rows are treated with lower Elo K, not discarded.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SOURCE = "Mon Sport Info / martj42 international results"
SOURCE_URL = "https://www.mon-sport-info.com/en/catalogue/football-results-intl"
DOWNLOAD_URL = "https://www.mon-sport-info.com/en/catalogue/football-results-intl/download/csv"
FALLBACK_SOURCE = "openfootball/internationals"
FALLBACK_SOURCE_URL = "https://github.com/openfootball/internationals"
LICENSE = "CC0-1.0"
SOURCE_UPDATED = "2026-07-20"
# Deliberately excludes Olympic, youth, B-team, and invitational/select-team folders.
COMPETITION_DIRS = {
    "friendly", "fifa_world_cup", "fifa_world_cup_qualification",
    "uefa_euro", "uefa_euro_qualification", "uefa_nations_league",
    "afc_asian_cup", "afc_asian_cup_qualification",
    "african_cup_of_nations", "african_cup_of_nations_qualification",
    "copa_america", "concacaf_gold_cup", "concacaf_gold_cup_qualification",
    "ofc_nations_cup",
}
FRIENDLY_K_MULTIPLIER = 0.55
COMPETITION_K_MULTIPLIERS = {
    "fifa_world_cup": 1.25,
    "fifa_world_cup_qualification": 1.0,
    "uefa_euro": 1.2,
    "uefa_euro_qualification": 1.0,
    "uefa_nations_league": 1.0,
    "afc_asian_cup": 1.2,
    "afc_asian_cup_qualification": 1.0,
    "african_cup_of_nations": 1.2,
    "african_cup_of_nations_qualification": 1.0,
    "copa_america": 1.2,
    "concacaf_gold_cup": 1.2,
    "concacaf_gold_cup_qualification": 1.0,
    "ofc_nations_cup": 1.2,
}
MATCH_RE = re.compile(r"^\s{2}(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)(?:\s+@\s+(.+))?$")


def _read_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Football-Quant-Lab/6.0"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8-sig")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not retrieve {url}: {exc}") from exc


def parse_football_txt(content: str, filename: str, competition: str) -> list[dict]:
    """Parse settled fixtures from one openfootball Football.TXT file."""
    match = re.match(r"(\d{4})_", Path(filename).name)
    if not match:
        raise ValueError(f"Cannot infer year from filename: {filename}")
    year = int(match.group(1))
    current_date = None
    rows = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("=", "#", "(")):
            continue
        if re.match(r"^[A-Z][a-z]{2} [A-Z][a-z]{2} \d{1,2}$", stripped):
            current_date = datetime.strptime(f"{stripped} {year}", "%a %b %d %Y").date().isoformat()
            continue
        parsed = MATCH_RE.match(line)
        if not parsed or current_date is None:
            continue
        home, hg, ag, away, venue = parsed.groups()
        home, away = re.sub(r"\s+", " ", home).strip(), re.sub(r"\s+", " ", away).strip()
        if not home or not away or home == away:
            continue
        rows.append({
            "match_id": f"{competition}-{current_date}-{home}-{away}",
            "date": current_date, "home_team": home, "away_team": away,
            "home_goals": int(hg), "away_goals": int(ag),
            "competition": competition,
            "match_type": "friendly" if competition == "friendly" else "competitive",
            "venue": (venue or "").strip(),
        })
    return rows


def _tree_files(tree_text: str, start_year: int, end_year: int) -> list[tuple[str, str]]:
    current_dir = None
    selected = []
    for line in tree_text.splitlines():
        folder = re.search(r"[├└]───([^/]+)$", line)
        if folder:
            current_dir = folder.group(1).strip()
            continue
        file_match = re.search(r"([0-9]{4}_[a-z0-9_]+\.txt)$", line.strip())
        if current_dir in COMPETITION_DIRS and file_match:
            filename = file_match.group(1)
            year = int(filename[:4])
            if start_year <= year <= end_year:
                selected.append((current_dir, filename))
    return selected


def _fetch_monsport_results(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch CC0 international results snapshot; keep senior men's full internationals only."""
    content = _read_url(DOWNLOAD_URL)
    frame = pd.read_csv(StringIO(content))
    required = {"category", "date", "home", "away", "home_score", "away_score",
                "competition", "city", "country", "neutral"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Mon Sport Info CSV missing columns: {sorted(missing)}")
    frame = frame[frame.category.astype(str).str.strip().str.lower().isin({"men", "male"})].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame = frame[
        frame.date.notna()
        & frame.home_score.notna()
        & frame.away_score.notna()
        & frame.date.dt.year.between(start_year, end_year)
    ].copy()
    frame["neutral"] = frame.neutral.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    frame["home_team"] = frame.home.astype(str).str.strip()
    frame["away_team"] = frame.away.astype(str).str.strip()
    frame["home_goals"] = pd.to_numeric(frame.home_score, errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame.away_score, errors="coerce")
    frame["competition"] = frame.competition.astype(str).str.strip()
    frame["match_type"] = np.where(
        frame.competition.str.contains("friendly", case=False, na=False),
        "friendly", "competitive"
    )
    frame["venue"] = (
        frame.city.fillna("").astype(str).str.strip()
        + ", "
        + frame.country.fillna("").astype(str).str.strip()
    ).str.strip(", ")
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["home_goals", "away_goals"])
    frame["home_goals"] = frame.home_goals.astype(int)
    frame["away_goals"] = frame.away_goals.astype(int)
    frame["match_id"] = (
        frame.date + "-" + frame.home_team + "-" + frame.away_team
        + "-" + frame.home_goals.astype(str) + "-" + frame.away_goals.astype(str)
    )
    return frame[[
        "match_id", "date", "home_team", "away_team", "home_goals",
        "away_goals", "competition", "match_type", "venue", "neutral"
    ]].drop_duplicates("match_id")


def fetch_international_results(output_csv: str | Path, start_year: int = 2000,
                                end_year: int = 2026, pause_seconds: float = 0.0) -> dict:
    """Fetch CC0 men's international results; fall back to openfootball if needed."""
    if start_year < 1872 or end_year < start_year:
        raise ValueError("Invalid international results year range.")
    source_used = SOURCE
    source_url = DOWNLOAD_URL
    source_updated = SOURCE_UPDATED
    try:
        frame = _fetch_monsport_results(start_year, end_year)
        if frame.empty:
            raise RuntimeError("Mon Sport Info returned no settled men's matches in the requested date range.")
    except (RuntimeError, ValueError, pd.errors.ParserError) as primary_error:
        # A transparent fallback preserves pipeline availability, but has an older data cutoff.
        source_used = FALLBACK_SOURCE
        source_url = FALLBACK_SOURCE_URL
        source_updated = None
        tree = _read_url(f"{FALLBACK_SOURCE_URL}/raw/refs/heads/master/TREE.md")
        files = _tree_files(tree, start_year, end_year)
        if not files:
            raise RuntimeError(
                f"Primary source failed ({primary_error}); no fallback files found."
            ) from primary_error

        def fetch_one(item: tuple[str, str]) -> tuple[str, str, list[dict], str | None]:
            competition, filename = item
            url = f"{FALLBACK_SOURCE_URL}/raw/refs/heads/master/{competition}/{filename}"
            try:
                parsed_rows = parse_football_txt(_read_url(url), filename, competition)
                return competition, filename, parsed_rows, None
            except (RuntimeError, ValueError) as exc:
                return competition, filename, [], str(exc)

        rows, missing = [], []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for competition, filename, parsed_rows, error in executor.map(fetch_one, files):
                rows.extend(parsed_rows)
                if error:
                    missing.append({"file": f"{competition}/{filename}", "reason": error})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("No settled international fixtures parsed from fallback source.")
        frame["neutral"] = False
    else:
        missing = []
        files = []

    frame = frame.drop_duplicates("match_id").sort_values(
        ["date", "home_team", "away_team"], kind="stable"
    )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {
        "source": source_used, "source_url": source_url, "license": LICENSE,
        "source_snapshot_updated": source_updated,
        "scope": "senior men's full internationals; women filtered out",
        "start_year": start_year, "end_year": end_year,
        "files_discovered": len(files), "matches": int(len(frame)),
        "first_date": str(frame.date.min()), "last_date": str(frame.date.max()),
        "competitions": {str(k): int(v) for k, v in frame.competition.value_counts().items()},
        "missing_files": missing, "output": str(output),
        "neutral_venue_matches": int(frame.neutral.astype(bool).sum()),
        "limitations": [
            "Community-maintained CC0 historical dataset derived from martj42; not an official FIFA feed.",
            "Men's full-international classification follows source data; FIFA A-match status is not independently verified for every friendly.",
            "The Mon Sport Info snapshot is dated 2026-07-20; later results may be absent.",
            "No lineups, player availability, travel, or rest features are supplied.",
            "Women's, youth, Olympic, B-team, and select-team matches are excluded.",
        ],
    }


def _probabilities(home_rating: float, away_rating: float,
                   home_advantage: float = 45.0, draw_probability: float = 0.265) -> np.ndarray:
    expected_home = 1.0 / (1.0 + 10.0 ** (-((home_rating + home_advantage) - away_rating) / 400.0))
    return np.array([(1.0 - draw_probability) * expected_home,
                     draw_probability,
                     (1.0 - draw_probability) * (1.0 - expected_home)])


def _k_multiplier(competition: str) -> float:
    name = str(competition).strip().lower().replace(" ", "_")
    if "friendly" in name:
        return FRIENDLY_K_MULTIPLIER
    if name in COMPETITION_K_MULTIPLIERS:
        return COMPETITION_K_MULTIPLIERS[name]
    if "world_cup" in name and "qualification" not in name:
        return 1.25
    if any(token in name for token in ("euro", "asian_cup", "african_cup", "copa_america", "gold_cup")) and "qualification" not in name:
        return 1.2
    return 1.0


def run_international_backtest(matches: pd.DataFrame, min_train: int = 50,
                               base_k: float = 20.0, home_advantage: float = 45.0,
                               draw_probability: float = 0.265) -> tuple[pd.DataFrame, dict]:
    """Strict-date national-team Elo replay; same-day results update only after predictions."""
    required = {"date", "home_team", "away_team", "home_goals", "away_goals", "competition"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if min_train < 1 or base_k <= 0 or not 0 < draw_probability < 1:
        raise ValueError("Invalid min_train, base_k, or draw_probability.")
    df = matches.copy()
    df["date"] = pd.to_datetime(df.date, errors="coerce", utc=True)
    if df.date.isna().any() or df[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Invalid date or missing settled score.")
    if (df[["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError("Goals cannot be negative.")
    df = df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    ratings: dict[str, float] = {}
    history_count = 0
    output = []
    for day_date, day in df.groupby("date", sort=True):
        day_updates: dict[str, float] = {}
        for row in day.itertuples():
            home, away = str(row.home_team), str(row.away_team)
            rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
            neutral_value = getattr(row, "neutral", False)
            is_neutral = (
                bool(neutral_value) if isinstance(neutral_value, (bool, np.bool_))
                else str(neutral_value).strip().lower() in {"true", "1", "yes", "y"}
            )
            match_home_advantage = 0.0 if is_neutral else home_advantage
            p = _probabilities(rh, ra, match_home_advantage, draw_probability)
            hg, ag = int(row.home_goals), int(row.away_goals)
            outcome = 0 if hg > ag else 1 if hg == ag else 2
            if history_count >= min_train:
                output.append({
                    "match_id": getattr(row, "match_id", None), "date": day_date,
                    "home_team": home, "away_team": away,
                    "competition": str(row.competition),
                    "match_type": getattr(row, "match_type", "friendly" if row.competition == "friendly" else "competitive"),
                    "actual_home_goals": hg, "actual_away_goals": ag, "outcome": outcome,
                    "home_p": float(p[0]), "draw_p": float(p[1]), "away_p": float(p[2]),
                    "home_elo_pre": rh, "away_elo_pre": ra, "training_matches": history_count,
                    "neutral": is_neutral,
                })
            expected_home = p[0] + 0.5 * p[1]
            actual_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = base_k * _k_multiplier(str(row.competition)) * (actual_home - expected_home)
            day_updates[home] = day_updates.get(home, 0.0) + delta
            day_updates[away] = day_updates.get(away, 0.0) - delta
        for team, delta in day_updates.items():
            ratings[team] = ratings.get(team, 1500.0) + delta
        history_count += len(day)
    predictions = pd.DataFrame(output)
    if predictions.empty:
        metrics = {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    else:
        probs = predictions[["home_p", "draw_p", "away_p"]].to_numpy(dtype=float)
        outcomes = predictions.outcome.to_numpy(dtype=int)
        one_hot = np.eye(3)[outcomes]
        losses = np.sum((probs - one_hot) ** 2, axis=1)
        metrics = {
            "n": int(len(predictions)), "brier": float(losses.mean()),
            "log_loss": float(-np.log(np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-15, 1)).mean()),
            "accuracy": float((probs.argmax(axis=1) == outcomes).mean()),
        }
    summary = {
        "competition_scope": "selected senior men's internationals",
        "method": "strict-date walk-forward Elo; same-day fixtures predicted before ratings update",
        "min_train": min_train, "base_k": base_k, "friendly_k_multiplier": FRIENDLY_K_MULTIPLIER,
        "home_advantage_elo_points": home_advantage, "neutral_venue_home_advantage": 0.0,
        "draw_probability": draw_probability,
        "input_matches": int(len(df)), "predicted_matches": int(len(predictions)),
        "metrics": metrics,
        "limitations": [
            "Initial ratings are equal; early team ratings are uncertain.",
            "Friendly K reduction is a transparent heuristic, not fitted or proven optimal.",
            "No explicit lineup, squad rotation, travel, rest, or neutral-venue feature.",
            "Results-only probabilities require calibration and independent forward validation before market use.",
            "No odds supplied: ROI, yield, CLV, and drawdown are not evaluated.",
        ],
    }
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="International senior men's results and Elo backtest.")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--output", default="data/international/senior_men.csv")
    fetch.add_argument("--start-year", type=int, default=2000)
    fetch.add_argument("--end-year", type=int, default=2026)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--input", default="data/international/senior_men.csv")
    backtest.add_argument("--output-dir", default="artifacts/international_v1")
    backtest.add_argument("--min-train", type=int, default=50)
    args = parser.parse_args()
    if args.command == "fetch":
        result = fetch_international_results(args.output, args.start_year, args.end_year)
        print(json.dumps(result, indent=2))
        return
    predictions, summary = run_international_backtest(pd.read_csv(args.input), min_train=args.min_train)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "international_oos_predictions.csv", index=False)
    (output / "international_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
