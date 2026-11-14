import pandas as pd
import pytest

from quantlab.international import (
    _tree_files,
    parse_football_txt,
    run_international_backtest,
)


def test_parse_friendly_football_txt():
    content = """= Friendly 2026

# Date       Sun Jan 18 - Sun Jan 18 2026 (1d)

Sun Jan 18
  Bolivia                1-1 Panama                   @ Tarija, Bolivia
  Mexico                 2-0 Iceland                  @ Querétaro, Mexico
     (Player 12')
"""
    rows = parse_football_txt(content, "2026_friendly.txt", "friendly")
    assert len(rows) == 2
    assert rows[0]["date"] == "2026-01-18"
    assert rows[0]["home_team"] == "Bolivia"
    assert rows[0]["away_team"] == "Panama"
    assert rows[0]["home_goals"] == 1
    assert rows[0]["away_goals"] == 1
    assert rows[0]["match_type"] == "friendly"
    assert rows[0]["venue"] == "Tarija, Bolivia"


def test_tree_discovery_only_whitelisted_competitions_and_years():
    tree = """├───friendly
│       1999_friendly.txt
│       2025_friendly.txt
├───fifa_world_cup
│       2026_fifa_world_cup.txt
├───more
│       2025_invitational.txt
"""
    assert _tree_files(tree, 2000, 2025) == [
        ("friendly", "2025_friendly.txt")
    ]


def test_walk_forward_excludes_same_day_and_applies_friendly_discount():
    matches = pd.DataFrame([
        {"date": "2020-01-01", "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 0, "competition": "friendly", "match_type": "friendly"},
        {"date": "2020-01-01", "home_team": "C", "away_team": "D",
         "home_goals": 0, "away_goals": 2, "competition": "friendly", "match_type": "friendly"},
        {"date": "2020-01-02", "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 0, "competition": "friendly", "match_type": "friendly"},
    ])
    predictions, summary = run_international_backtest(matches, min_train=1)
    assert len(predictions) == 1
    assert predictions.iloc[0]["date"] == pd.Timestamp("2020-01-02", tz="UTC")
    assert predictions.iloc[0]["training_matches"] == 2
    assert predictions.iloc[0]["home_elo_pre"] > 1500
    assert predictions.iloc[0]["away_elo_pre"] < 1500
    assert summary["metrics"]["n"] == 1
    assert summary["friendly_k_multiplier"] < 1.0


def test_walk_forward_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        run_international_backtest(pd.DataFrame([{"date": "2020-01-01"}]))
