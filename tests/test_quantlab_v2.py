import pandas as pd
import pytest
from quantlab.data import team_form_stats, venue_stats, validate_matches_csv
from quantlab.evaluation import backtest_summary, bankroll_metrics
from quantlab.storage import backup_database, restore_database, initialize, save_dataset, load_matches


def history():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01","2026-01-02","2026-01-03","2026-01-04"], utc=True),
        "home_team":["A","B","A","C"], "away_team":["B","A","C","A"],
        "home_goals":[0,3,2,1], "away_goals":[1,0,0,2],
        "home_xg":[0.4,2.2,1.8,0.9], "away_xg":[1.2,0.2,0.3,1.7],
    })


def test_recency_and_venue_features():
    df=history()
    form=team_form_stats(df,"A",limit=4,half_life=1)
    home=venue_stats(df,"A","home",limit=10)
    assert form["matches"] == 4 and form["xg_for"] is not None
    assert home["matches"] == 2
    assert home["scored_per_match"] == pytest.approx(1.0)
    assert home["xg_for"] == pytest.approx(1.1)


def test_backtest_confidence_interval():
    result=backtest_summary([.8,.2,.7,.1],[1,0,1,0],bootstrap_samples=200)
    assert result["n"] == 4
    assert result["brier_ci_low"] <= result["brier"] <= result["brier_ci_high"]


def test_clv_and_drawdown_metrics():
    result=bankroll_metrics([10,-15,5],[10,10,10],[2.1,2.0,2.2],[1.9,2.1,2.0])
    assert result["roi"] == pytest.approx(0)
    assert result["max_drawdown"] == pytest.approx(15)
    assert result["clv"] > 0


def test_database_backup_restore(tmp_path):
    db=tmp_path/"source.sqlite"
    backup=tmp_path/"backup.sqlite"
    initialize(db)
    backup_database(db,backup)
    restored=tmp_path/"restored.sqlite"
    restore_database(backup,restored)
    assert load_matches(restored).empty


def test_xg_validation_requires_pair():
    csv= "date,home_team,away_team,home_goals,away_goals,home_xg\n2026-01-01,A,B,1,0,0.8\n"
    _, errors=validate_matches_csv(__import__("io").StringIO(csv))
    assert any("both home_xg and away_xg" in e for e in errors)
