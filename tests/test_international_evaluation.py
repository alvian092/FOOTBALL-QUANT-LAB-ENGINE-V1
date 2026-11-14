import numpy as np
import pandas as pd
import pytest

from quantlab.international_evaluation import (
    _competition_class,
    _fit_temperature,
    evaluate_international_predictions,
)


def _predictions(n=40):
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for i, date in enumerate(dates):
        competition = ("friendly", "fifa_world_cup_qualification", "fifa_world_cup")[i % 3]
        outcome = i % 3
        # Slightly overconfident but valid probability triplet.
        p = [0.70, 0.20, 0.10]
        rows.append({
            "match_id": f"m{i}", "date": date, "home_team": f"H{i}",
            "away_team": f"A{i}", "competition": competition,
            "match_type": "friendly" if competition == "friendly" else "competitive",
            "outcome": outcome, "home_p": p[0], "draw_p": p[1], "away_p": p[2],
        })
    return pd.DataFrame(rows)


def test_competition_grouping():
    assert _competition_class("friendly") == "friendly"
    assert _competition_class("fifa_world_cup_qualification") == "qualification"
    assert _competition_class("uefa_euro") == "tournament"


def test_temperature_fit_returns_bounded_positive_value():
    probs = np.tile([0.70, 0.20, 0.10], (30, 1))
    outcomes = np.array([0, 1, 2] * 10)
    temperature = _fit_temperature(probs, outcomes)
    assert 0.5 <= temperature <= 3.0


def test_prequential_calibration_does_not_calibrate_before_warmup():
    predictions, summary = evaluate_international_predictions(
        _predictions(40), min_calibration_history=10, refit_every_dates=2
    )
    assert predictions["model_intl_calibrated_home_p"].notna().all()
    assert predictions["calibration_temperature"].notna().all()
    assert summary["calibrated_predictions"] == 40
    assert "friendly" in summary["segments"]["competition_class"]
    assert "qualification" in summary["segments"]["competition_class"]
    assert "tournament" in summary["segments"]["competition_class"]


def test_calibration_rejects_missing_prediction_fields():
    with pytest.raises(ValueError, match="Prediction file missing"):
        evaluate_international_predictions(pd.DataFrame([{"date": "2020-01-01"}]))
