"""Fetch CC0 cross-league results from openfootball/football.json."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

LEAGUES = {
    "bundesliga": ("de.1", "Germany Bundesliga"),
    "laliga": ("es.1", "Spain La Liga"),
    "seriea": ("it.1", "Italy Serie A"),
    "ligue1": ("fr.1", "France Ligue 1"),
    "eredivisie": ("nl.1", "Netherlands Eredivisie"),
}


def _team_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().removesuffix(" FC").removesuffix(" AFC").strip()


def _score(match: dict):
    score = match.get("score")
    if isinstance(score, dict):
        score = score.get("ft")
    if not isinstance(score, list) or len(score) != 2:
        return None
    try:
        home, away = int(score[0]), int(score[1])
    except (TypeError, ValueError):
        return None
    if home < 0 or away < 0:
        return None
    return home, away


def fetch_cross_leagues(output_dir: str | Path, start_year: int = 2010,
                        end_year: int = 2025, pause_seconds: float = 0.05) -> dict:
    """Fetch completed season files; missing league-season files are recorded, not fabricated."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for league_key, (code, league_name) in LEAGUES.items():
        rows = []
        missing_seasons = []
        for year in range(start_year, end_year + 1):
            season = f"{year}-{str(year + 1)[-2:]}"
            url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
            request = Request(url, headers={"User-Agent": "Football-Quant-Lab/5.0"})
            try:
                with urlopen(request, timeout=20) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 404:
                    missing_seasons.append(season)
                    continue
                raise
            except (URLError, TimeoutError) as exc:
                raise RuntimeError(f"Could not fetch {url}: {exc}") from exc
            for match in payload.get("matches", []):
                result = _score(match)
                if result is None or not match.get("date") or not match.get("team1") or not match.get("team2"):
                    continue
                home_goals, away_goals = result
                date = str(match["date"])[:10]
                home, away = _team_name(match["team1"]), _team_name(match["team2"])
                rows.append({
                    "match_id": f"{league_key}-{season}-{date}-{home}-{away}",
                    "date": date, "home_team": home, "away_team": away,
                    "home_goals": home_goals, "away_goals": away_goals,
                    "league": league_key, "league_name": league_name, "season": season,
                })
            time.sleep(pause_seconds)
        frame = pd.DataFrame(rows)
        if frame.empty:
            summaries[league_key] = {"matches": 0, "seasons": 0, "missing_seasons": missing_seasons}
            continue
        frame = frame.drop_duplicates("match_id").sort_values(["date", "home_team", "away_team"])
        frame.to_csv(output / f"{league_key}.csv", index=False)
        summaries[league_key] = {
            "matches": int(len(frame)), "seasons": int(frame.season.nunique()),
            "first_season": str(frame.season.min()), "last_season": str(frame.season.max()),
            "missing_seasons": missing_seasons,
            "path": str(output / f"{league_key}.csv"),
        }
    usable = sum(item["matches"] >= 500 for item in summaries.values())
    if usable < 3:
        raise RuntimeError(f"Only {usable} leagues have >=500 settled matches; cross-league pilot requires at least 3.")
    return {"source": "openfootball/football.json", "license": "CC0-1.0",
            "leagues": summaries, "usable_leagues": usable}


def main():
    parser = argparse.ArgumentParser(description="Fetch CC0 multi-league football results.")
    parser.add_argument("--output-dir", default="data/cross_league")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    result = fetch_cross_leagues(args.output_dir, args.start_year, args.end_year)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
