import pytest
from quantlab.evaluation import brier_score, log_loss, multiclass_brier_score, calibration_bins, bankroll_metrics


def test_binary_scores():
    assert brier_score([0.8, 0.2], [1, 0]) == pytest.approx(0.04)
    assert log_loss([0.8, 0.2], [1, 0]) > 0


def test_multiclass_brier_and_calibration():
    assert multiclass_brier_score([[0.7,0.2,0.1],[0.1,0.2,0.7]],[0,2]) == pytest.approx(0.14)
    bins = calibration_bins([0.8,0.2],[1,0],bins=2)
    assert sum(x["count"] for x in bins) == 2


def test_roi_requires_explicit_stakes():
    assert bankroll_metrics([5,-2])["roi"] is None
    assert bankroll_metrics([5,-2],[10,10])["roi"] == pytest.approx(0.15)
