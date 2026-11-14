import numpy as np
import pytest
from quantlab.probability import calculate_probabilities, goal_probability_matrix
from quantlab.market import no_vig_probabilities, evaluate_value
from quantlab.risk import flat_stake, fractional_kelly


def test_probability_matrix_normalizes():
    matrix = goal_probability_matrix(1.4, 1.1)
    assert matrix.sum() == pytest.approx(1.0)
    assert np.all(matrix >= 0)


def test_outcome_probabilities_sum_to_one():
    p = calculate_probabilities(1.5, 1.0)
    assert p.home_win + p.draw + p.away_win == pytest.approx(1.0)


def test_market_no_vig_normalizes():
    probs = no_vig_probabilities({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert sum(probs.values()) == pytest.approx(1.0)


def test_value_gate_rejects_negative_ev():
    result = evaluate_value(0.4, 2.0, uncertainty=0.01)
    assert result["category"] == "NO VALUE"
    assert result["eligible"] is False


def test_risk_controls():
    assert flat_stake(1000, 0.01) == 10
    assert 0 <= fractional_kelly(0.6, 2.0) <= 0.02
