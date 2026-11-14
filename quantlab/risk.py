"""Conservative staking utilities."""
def flat_stake(bankroll: float, fraction: float = 0.01) -> float:
    if bankroll <= 0 or not 0 < fraction <= 0.05:
        raise ValueError("Bankroll must be positive and flat fraction must be in (0, 5%].")
    return bankroll * fraction


def fractional_kelly(probability: float, decimal_odds: float,
                     fraction: float = 0.25, cap: float = 0.02) -> float:
    if not 0 < probability < 1 or decimal_odds <= 1:
        raise ValueError("Invalid probability or decimal odds.")
    if not 0 < fraction <= 1 or not 0 < cap <= 0.05:
        raise ValueError("Kelly fraction and cap are outside safe bounds.")
    b = decimal_odds - 1.0
    full = (b * probability - (1.0 - probability)) / b
    return max(0.0, min(cap, full * fraction))


def stake_amount(bankroll: float, fraction: float) -> float:
    if bankroll <= 0 or not 0 <= fraction <= 0.05:
        raise ValueError("Invalid bankroll or stake fraction.")
    return bankroll * fraction
