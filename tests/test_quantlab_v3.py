import pandas as pd
import pytest

from quantlab.backtesting import walk_forward_backtest, segmented_performance
from quantlab.market_intelligence import summarize_odds_movement, validate_odds_snapshots
from quantlab.storage import initialize, save_odds_snapshot, list_odds_snapshots


def fixture_data():
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC"),
        "match_id": [f"M{i}" for i in range(8)],
        "league": ["League A"] * 8,
        "home_team": ["A","B","C","A","B","C","A","B"],
        "away_team": ["B","C","A","C","A","B","B","C"],
        "home_goals": [1,0,2,1,3,0,2,1],
        "away_goals": [0,0,1,1,1,2,1,0],
        "home_odds": [2.1] * 8, "draw_odds": [3.2] * 8, "away_odds": [3.6] * 8,
    })


def test_walk_forward_uses_only_strictly_prior_dates_and_benchmarks_market():
    df = fixture_data()
    seen = []
    def predictor(train, fixture):
        assert train.date.max() < fixture.date
        seen.append((len(train), fixture.match_id))
        return {"home": 0.5, "draw": 0.25, "away": 0.25}
    result = walk_forward_backtest(df, predictor, min_train=4, bootstrap_samples=200)
    assert len(result["results"]) == 4
    assert result["model"]["n"] == 4
    assert result["market"]["n"] == 4
    assert result["model_brier_ci"][0] <= result["model"]["brier"] <= result["model_brier_ci"][1]
    assert len(seen) == 4


def test_walk_forward_does_not_use_same_day_results():
    df = fixture_data()
    df.loc[1, "date"] = df.loc[0, "date"]
    def predictor(train, fixture):
        assert not (train.date == fixture.date).any()
        return {"home": 1/3, "draw": 1/3, "away": 1/3}
    result = walk_forward_backtest(df, predictor, min_train=2, bootstrap_samples=100)
    assert not result["results"].empty


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        walk_forward_backtest(fixture_data(), lambda *_: {"home": .6, "draw": .3, "away": .3}, min_train=2)


def test_odds_movement_normalizes_each_snapshot():
    rows = []
    for captured, prices in [
        ("2026-01-01T10:00:00Z", {"Home":2.2,"Draw":3.2,"Away":3.4}),
        ("2026-01-01T18:00:00Z", {"Home":1.9,"Draw":3.4,"Away":4.0}),
    ]:
        for selection, odds in prices.items():
            rows.append({"match_id":"M1","market":"1X2","selection":selection,
                         "odds":odds,"captured_at":captured,"bookmaker":"Book A"})
    result = summarize_odds_movement(pd.DataFrame(rows))
    home = result[result.selection == "Home"].iloc[0]
    assert home.opening_odds == pytest.approx(2.2)
    assert home.closing_odds == pytest.approx(1.9)
    assert home.movement == "SHORTENED"
    assert home.probability_shift > 0


def test_odds_snapshot_validation():
    bad = pd.DataFrame([{"match_id":"M","market":"1X2","selection":"H","odds":1,
                         "captured_at":"2026-01-01T00:00:00Z"}])
    with pytest.raises(ValueError):
        validate_odds_snapshots(bad)


def test_segmented_performance_flags_small_samples_and_clv():
    records = pd.DataFrame({
        "league":["A","A","B"], "selected_market":["1X2 Home"]*3,
        "simulated_profit":[10,-5,2], "stake":[10,10,10],
        "odds":[2.1,2.0,2.2], "closing_odds":[1.9,2.1,2.0],
    })
    result = segmented_performance(records, group_by=("league","selected_market"), min_bets=2)
    a = result[result.league == "A"].iloc[0]
    b = result[result.league == "B"].iloc[0]
    assert a.roi == pytest.approx(.25)
    assert a.sample_status == "OK"
    assert b.sample_status == "SMALL SAMPLE"
    assert a.avg_clv > 0


def test_odds_snapshots_persist_immutably(tmp_path):
    db = tmp_path / "quant.sqlite"
    initialize(db)
    stamp = "2026-01-01T10:00:00+00:00"
    save_odds_snapshot("M1","1X2",{"Home":2.1,"Draw":3.2,"Away":3.5},stamp,db_path=db)
    save_odds_snapshot("M1","1X2",{"Home":2.0,"Draw":3.3,"Away":3.8},stamp,db_path=db)
    saved = list_odds_snapshots(db)
    assert len(saved) == 3
    assert set(saved.selection) == {"Home","Draw","Away"}
