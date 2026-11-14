"""Goal-distribution and match-probability calculations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import poisson


@dataclass(frozen=True)
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    over_15: float
    over_25: float
    over_35: float
    under_15: float
    under_25: float
    under_35: float
    btts_yes: float
    btts_no: float
    score_matrix: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {
            "home_win": self.home_win, "draw": self.draw, "away_win": self.away_win,
            "over_1_5": self.over_15, "over_2_5": self.over_25, "over_3_5": self.over_35,
            "under_1_5": self.under_15, "under_2_5": self.under_25, "under_3_5": self.under_35,
            "btts_yes": self.btts_yes, "btts_no": self.btts_no,
        }


def dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float,
                    lambda_away: float, rho: float) -> float:
    """Low-score Dixon-Coles correction factor."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def goal_probability_matrix(lambda_home: float, lambda_away: float,
                            max_goals: int = 10, rho: float = -0.08) -> np.ndarray:
    """Return a normalized score probability matrix with DC low-score adjustment."""
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Expected goals must be greater than zero.")
    if not 4 <= max_goals <= 20:
        raise ValueError("max_goals must be between 4 and 20.")
    home = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(home, away)
    for h, a in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[h, a] *= dixon_coles_tau(h, a, lambda_home, lambda_away, rho)
    matrix = np.clip(matrix, 0.0, None)
    total = float(matrix.sum())
    if total <= 0:
        raise ArithmeticError("Goal matrix has zero probability mass.")
    return matrix / total


def calculate_probabilities(lambda_home: float, lambda_away: float,
                            max_goals: int = 10, rho: float = -0.08) -> MatchProbabilities:
    matrix = goal_probability_matrix(lambda_home, lambda_away, max_goals, rho)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    over = {line: float(matrix[totals > line].sum()) for line in (1, 2, 3)}
    btts = float(matrix[1:, 1:].sum())
    return MatchProbabilities(
        home_win, draw, away_win,
        over[1], over[2], over[3],
        1-over[1], 1-over[2], 1-over[3],
        btts, 1-btts, matrix
    )


def top_correct_scores(matrix: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    indices = np.dstack(np.unravel_index(np.argsort(matrix.ravel())[::-1][:limit], matrix.shape))[0]
    return [{"score": f"{int(h)}-{int(a)}", "probability": float(matrix[h, a])}
            for h, a in indices]


def estimate_expected_goals(home_scored: float, home_conceded: float,
                             away_scored: float, away_conceded: float,
                             league_home_avg: float, league_away_avg: float,
                             home_advantage: float = 1.08) -> tuple[float, float]:
    """Estimate lambdas from per-match goals; this is not an xG estimate."""
    vals = (home_scored, home_conceded, away_scored, away_conceded,
            league_home_avg, league_away_avg, home_advantage)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in vals):
        raise ValueError("All scoring rates and home advantage must be positive finite numbers.")
    home_attack = home_scored / league_home_avg
    home_defence = home_conceded / league_away_avg
    away_attack = away_scored / league_away_avg
    away_defence = away_conceded / league_home_avg
    lam_h = league_home_avg * home_attack * away_defence * home_advantage
    lam_a = league_away_avg * away_attack * home_defence / home_advantage
    return max(0.05, lam_h), max(0.05, lam_a)
