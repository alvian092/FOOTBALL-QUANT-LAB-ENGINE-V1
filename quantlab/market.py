"""Odds normalization and value-bet calculations."""
from __future__ import annotations


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / decimal_odds


def no_vig_probabilities(odds: dict[str, float]) -> dict[str, float]:
    if len(odds) < 2:
        raise ValueError("At least two mutually exclusive outcomes are required.")
    raw = {k: implied_probability(float(v)) for k, v in odds.items()}
    overround = sum(raw.values())
    return {k: v / overround for k, v in raw.items()}


def overround(odds: dict[str, float]) -> float:
    if len(odds) < 2:
        raise ValueError("At least two outcomes are required.")
    return sum(implied_probability(float(v)) for v in odds.values()) - 1.0


def evaluate_value(model_probability: float, decimal_odds: float,
                   uncertainty: float = 0.0) -> dict[str, float | str | bool]:
    """Evaluate a single binary/selection bet; uncertainty is an absolute probability margin."""
    if not 0 < model_probability < 1:
        raise ValueError("Model probability must be between 0 and 1.")
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    if not 0 <= uncertainty < 1:
        raise ValueError("Uncertainty must be in [0, 1).")
    ev = model_probability * decimal_odds - 1.0
    lower = max(0.0, model_probability - 1.96 * uncertainty)
    upper = min(1.0, model_probability + 1.96 * uncertainty)
    conservative_ev = lower * decimal_odds - 1.0
    if ev <= 0:
        category = "NO VALUE"
    elif conservative_ev <= 0:
        category = "LOW VALUE"
    elif ev >= 0.05:
        category = "STRONG VALUE"
    elif ev >= 0.02:
        category = "MODERATE VALUE"
    else:
        category = "LOW VALUE"
    return {
        "implied_probability": 1.0 / decimal_odds,
        "model_probability": model_probability,
        "ev": ev,
        "value_difference": model_probability - (1.0 / decimal_odds),
        "ci_low": lower,
        "ci_high": upper,
        "conservative_ev": conservative_ev,
        "category": category,
        "eligible": ev > 0 and conservative_ev > 0,
    }
