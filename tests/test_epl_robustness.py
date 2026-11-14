import numpy as np
import pandas as pd
import pytest

from quantlab.epl_robustness import evaluate_predictions, paired_cluster_bootstrap


def _predictions():
    rows = []
    for day in range(4):
        for fixture in range(2):
            outcome = (day + fixture) % 3
            elo = [0.1, 0.1, 0.1]
            elo[outcome] = 0.8
            ensemble = [0.2, 0.2, 0.2]
            ensemble[outcome] = 0.6
            rows.append({
                "date": f"2020-08-{1 + day * 7:02d}", "outcome": outcome,
                **{f"elo_{name}_p": elo[i] for i, name in enumerate(("home", "draw", "away"))},
                **{f"ensemble_{name}_p": ensemble[i] for i, name in enumerate(("home", "draw", "away"))},
            })
    return pd.DataFrame(rows)


def test_robustness_reports_paired_cluster_and_season_metrics():
    seasons, summary = evaluate_predictions(_predictions(), samples=1000, seed=7)
    assert summary["fixtures"] == 8
    assert summary["paired_bootstrap"]["clusters"] == 4
    assert summary["paired_bootstrap"]["metrics"]["brier"]["elo_better"] is True
    assert set(seasons.model) == {"elo", "ensemble"}
    assert seasons.season.nunique() == 1


def test_robustness_rejects_invalid_probability_rows():
    frame = _predictions()
    frame.loc[0, "elo_home_p"] = 2.0
    with pytest.raises(ValueError, match="between zero and one"):
        evaluate_predictions(frame, samples=1000)


def test_paired_bootstrap_requires_minimum_samples():
    losses = (np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    with pytest.raises(ValueError, match="at least 1,000"):
        paired_cluster_bootstrap(
            losses, losses, pd.Series(["2020-01-01", "2020-01-02"]), samples=99
        )
