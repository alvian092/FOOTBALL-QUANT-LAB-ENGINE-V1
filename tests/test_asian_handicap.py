import numpy as np
import pytest
from quantlab.asian_handicap import asian_handicap_outcomes, asian_handicap_ev


def test_asian_handicap_push_and_win():
    # Deterministic 2-1 home win.
    matrix = np.zeros((4, 4)); matrix[2, 1] = 1.0
    assert asian_handicap_outcomes(matrix, -1.0)["push"] == pytest.approx(1.0)
    assert asian_handicap_outcomes(matrix, 0.0)["full_win"] == pytest.approx(1.0)


def test_quarter_handicap_splits_stake():
    matrix = np.zeros((3, 3)); matrix[1, 1] = 1.0
    result = asian_handicap_outcomes(matrix, -0.25)
    assert result["push"] == pytest.approx(0.5)
    assert result["full_loss"] == pytest.approx(0.5)


def test_asian_ev_accounts_for_push():
    outcomes = {"full_win": 0.4, "half_win": 0, "push": 0.2, "half_loss": 0, "full_loss": 0.4}
    assert asian_handicap_ev(outcomes, 2.0) == pytest.approx(0.0)
