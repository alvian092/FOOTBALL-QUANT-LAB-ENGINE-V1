import numpy as np
import pandas as pd

from quantlab.epl_backtest import run_epl_backtest


def _matches():
    rows = []
    dates = ["2020-08-01", "2020-08-08", "2020-08-15", "2020-08-22"]
    fixtures = [
        ("A", "B", 2, 0), ("C", "D", 1, 1),
        ("A", "C", 0, 1), ("B", "D", 2, 2),
        ("A", "D", 1, 0), ("B", "C", 0, 0),
        ("C", "A", 1, 2), ("D", "B", 0, 1),
    ]
    for i, (home, away, hg, ag) in enumerate(fixtures):
        rows.append({
            "match_id": f"M{i}", "date": dates[i // 2],
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "league": "EPL",
        })
    return pd.DataFrame(rows)


def test_epl_backtest_excludes_same_day_fixtures_from_training():
    results, summary = run_epl_backtest(_matches(), min_train=1, bootstrap_samples=100)
    assert len(results) == 6
    # Both fixtures on the second date only see the two fixtures from the first date.
    second_day = results[results.date == pd.Timestamp("2020-08-08", tz="UTC")]
    assert len(second_day) == 2
    assert second_day.training_matches.eq(2).all()
    assert summary["excluded_before_min_train"] == 2


def test_epl_backtest_emits_valid_probabilities_for_all_components():
    results, summary = run_epl_backtest(_matches(), min_train=1, bootstrap_samples=100)
    for model in ("poisson", "dixon_coles", "elo", "ensemble"):
        cols = [f"{model}_{outcome}_p" for outcome in ("home", "draw", "away")]
        values = results[cols].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0).all()
        assert (values <= 1).all()
        assert np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
        assert summary["models"][model]["n"] == len(results)


def test_epl_backtest_respects_minimum_training_threshold():
    results, summary = run_epl_backtest(_matches(), min_train=3, bootstrap_samples=100)
    assert len(results) == 4
    assert results.training_matches.ge(3).all()
    assert summary["excluded_before_min_train"] == 4
