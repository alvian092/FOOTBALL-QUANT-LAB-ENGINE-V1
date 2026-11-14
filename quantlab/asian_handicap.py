"""Asian handicap settlement probabilities from a scoreline matrix."""
from __future__ import annotations
import numpy as np


def asian_handicap_outcomes(score_matrix: np.ndarray, home_line: float) -> dict[str, float]:
    """Return full-win, half-win, push, half-loss, full-loss probabilities.

    home_line is the handicap applied to the home team's goal difference.
    Quarter lines are split into adjacent half-stakes (e.g. -0.25 = 0 and -0.5).
    """
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("score_matrix must be a non-empty square matrix.")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or matrix.sum() <= 0:
        raise ValueError("score_matrix must contain finite non-negative probability mass.")
    if round(home_line * 4) != home_line * 4:
        raise ValueError("Asian handicap line must be in quarter-goal increments.")
    matrix = matrix / matrix.sum()
    diff = np.subtract.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    # A quarter line is the average settlement of the two adjacent half-goal lines.
    quarter = abs(home_line * 4) % 2 == 1
    if quarter:
        lower = home_line - 0.25
        upper = home_line + 0.25
        a = _single_line(matrix, diff, lower)
        b = _single_line(matrix, diff, upper)
        return {k: (a[k] + b[k]) / 2 for k in a}
    return _single_line(matrix, diff, home_line)


def _single_line(matrix: np.ndarray, diff: np.ndarray, line: float) -> dict[str, float]:
    adjusted = diff + line
    return {
        "full_win": float(matrix[adjusted > 0].sum()),
        "push": float(matrix[np.isclose(adjusted, 0)].sum()),
        "full_loss": float(matrix[adjusted < 0].sum()),
        "half_win": 0.0,
        "half_loss": 0.0,
    }


def asian_handicap_ev(outcomes: dict[str, float], decimal_odds: float) -> float:
    """Expected net return per unit stake, including pushes and half outcomes."""
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    return ((outcomes["full_win"] * (decimal_odds - 1))
            + outcomes["half_win"] * ((decimal_odds - 1) / 2)
            - outcomes["half_loss"] * 0.5
            - outcomes["full_loss"])
