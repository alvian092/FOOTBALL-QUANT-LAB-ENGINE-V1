import numpy as np
import pandas as pd
import pytest

from quantlab.market_backtest import _devig, evaluate_market


def _inputs():
    predictions = pd.DataFrame([{
        "match_id": "m1", "date": "2026-08-01", "home_team": "Alpha FC",
        "away_team": "Beta", "outcome": 0,
        "model_adaptive_home_p": 0.60, "model_adaptive_draw_p": 0.25,
        "model_adaptive_away_p": 0.15,
    }])
    odds = pd.DataFrame([{
        "kickoff_utc": "2026-08-01T18:00:00Z", "home": "Alpha", "away": "Beta",
        "result": "1", "void_reason": "", "teams_reversed_after_logging": 0,
        "pin_open_home": 2.0, "pin_open_draw": 3.5, "pin_open_away": 4.5,
        "pin_close_home": 1.8, "pin_close_draw": 3.7, "pin_close_away": 5.0,
    }])
    return predictions, odds


def test_devig_proportional_probabilities_sum_to_one():
    probabilities = _devig(np.array([2.0, 3.0, 4.0]))
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_market_backtest_reports_positive_ev_roi_clv_and_drawdown():
    predictions, odds = _inputs()
    bets, summary = evaluate_market(predictions, odds, min_ev=0.03)
    assert len(bets) == 1
    assert summary["bets"] == 1
    assert summary["roi_yield"] == 1.0
    assert summary["average_clv_pct"] > 0
    assert summary["max_drawdown_units"] == 0


def test_market_backtest_filters_reversed_and_void_rows():
    predictions, odds = _inputs()
    odds.loc[0, "teams_reversed_after_logging"] = 1
    bets, summary = evaluate_market(predictions, odds)
    assert bets.empty
    assert summary["bets"] == 0


def test_market_backtest_rejects_missing_columns():
    predictions, odds = _inputs()
    with pytest.raises(ValueError, match="Odds file missing"):
        evaluate_market(predictions, odds.drop(columns=["pin_close_away"]))
