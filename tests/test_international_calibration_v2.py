import numpy as np
import pandas as pd
import pytest

from quantlab.international_calibration_v2 import (
    _prequential,
    competition_class,
    run_experiments,
)


def _frame(n=36):
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for i, day in enumerate(dates):
        rows.append({
            "date": day,
            "competition": ("friendly", "world_cup_qualification", "world_cup")[i % 3],
            "outcome": i % 3,
            "home_p": 0.60,
            "draw_p": 0.25,
            "away_p": 0.15,
        })
    return pd.DataFrame(rows)


def test_competition_classes():
    assert competition_class("International Friendly") == "friendly"
    assert competition_class("FIFA World Cup qualification") == "qualification"
    assert competition_class("UEFA EURO") == "tournament"


def test_prequential_category_temperature_only_uses_prior_dates():
    frame = _frame(12)
    frame["competition_class"] = frame.competition.map(competition_class)
    calibrated, _ = _prequential(frame, min_history=3, min_category_history=1,
                                  shrinkage_prior=2, refit_every_dates=1)
    assert np.isnan(calibrated[:3]).all()
    assert np.isfinite(calibrated[3:]).all()
    assert np.allclose(calibrated[3:].sum(axis=1), 1.0)


def test_same_day_predictions_are_not_in_each_others_training_history():
    frame = _frame(6)
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame["competition_class"] = frame.competition.map(competition_class)
    calibrated, _ = _prequential(frame, min_history=2, min_category_history=1,
                                  shrinkage_prior=1, refit_every_dates=1)
    # Both fixtures on day one are warm-up rows; day two is the first eligible date.
    assert np.isnan(calibrated[:2]).all()
    assert np.isfinite(calibrated[2:]).all()


def test_experiment_report_has_temporal_and_segment_metrics():
    frame = _frame(40)
    _, report = run_experiments(frame, bootstrap_samples=100)
    assert report["default_configuration_predeclared"] is True
    assert set(report["segments"]) == {"friendly", "qualification", "tournament"}
    assert "paired_date_cluster_bootstrap_ci_95" in report["overall"]
    assert "raw" in report["by_calendar_year"].get("2020", {})
    assert "calibrated" in report["by_calendar_year"].get("2020", {})
    assert report["sensitivity_grid"]
