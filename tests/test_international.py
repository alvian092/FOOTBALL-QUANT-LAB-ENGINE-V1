import pandas as pd
import pytest

from quantlab.international import (
    _tree_files,
    _fetch_monsport_results,
    _k_multiplier,
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



def test_monsport_feed_filters_men_and_preserves_neutral_venue(monkeypatch):
    csv_text = """category,date,home,away,home_score,away_score,competition,city,country,neutral
men,2026-06-11,Team A,Team B,2,1,FIFA World Cup,City,Country,True
women,2026-06-12,Team C,Team D,1,0,Friendly,City,Country,False
men,2026-06-13,Team E,Team F,0,0,Friendly,City,Country,False
men,2027-01-01,Team G,Team H,1,0,Friendly,City,Country,False
"""
    monkeypatch.setattr("quantlab.international._read_url", lambda url: csv_text)
    frame = _fetch_monsport_results(2000, 2026)
    assert len(frame) == 2
    assert set(frame.home_team) == {"Team A", "Team E"}
    assert bool(frame.loc[frame.home_team == "Team A", "neutral"].iloc[0]) is True
    assert frame.loc[frame.home_team == "Team E", "match_type"].iloc[0] == "friendly"


def test_competition_multiplier_supports_source_labels():
    assert _k_multiplier("Friendly") == 0.55
    assert _k_multiplier("FIFA World Cup") == 1.25
    assert _k_multiplier("UEFA EURO") == 1.2
    assert _k_multiplier("FIFA World Cup qualification") == 1.0
