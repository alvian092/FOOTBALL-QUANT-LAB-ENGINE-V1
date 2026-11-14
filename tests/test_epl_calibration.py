import pandas as pd

from quantlab.epl_calibration import run_expanding_calibration


def _predictions():
    rows = []
    for day in range(4):
        for fixture in range(2):
            outcome = (day + fixture) % 3
            row = {"date": f"2020-08-{1 + day * 7:02d}", "outcome": outcome}
            for model, confidence in (("poisson", 0.55), ("dixon_coles", 0.58), ("elo", 0.72), ("ensemble", 0.60)):
                probs = [0.0, 0.0, 0.0]
                probs[outcome] = confidence
                for k in range(3):
                    if k != outcome:
                        probs[k] = (1.0 - confidence) / 2.0
                for i, name in enumerate(("home", "draw", "away")):
                    row[f"{model}_{name}_p"] = probs[i]
            rows.append(row)
    return pd.DataFrame(rows)


def test_calibration_is_expanding_window_and_does_not_leak_future_outcomes():
    frame = _predictions()
    baseline, params, _ = run_expanding_calibration(
        frame, min_history=2, refit_every_dates=1, bootstrap_samples=1000, seed=11
    )
    changed = frame.copy()
    future = changed.date == "2020-08-22"
    changed.loc[future, "outcome"] = 2 - changed.loc[future, "outcome"]
    altered, _, _ = run_expanding_calibration(
        changed, min_history=2, refit_every_dates=1, bootstrap_samples=1000, seed=11
    )
    probability_cols = [column for column in baseline if column.endswith("_p")]
    assert baseline.loc[:5, probability_cols].equals(altered.loc[:5, probability_cols])
    assert params.iloc[0].history_fixtures == 0
    assert params.iloc[0].temperature_elo == 1.0
    assert params.iloc[0].weight_elo == 0.2


def test_calibration_outputs_valid_probabilities():
    output, _, _ = run_expanding_calibration(
        _predictions(), min_history=2, refit_every_dates=1, bootstrap_samples=1000, seed=3
    )
    for model in ("poisson_cal", "dixon_coles_cal", "elo_cal",
                  "adaptive_ensemble_raw", "adaptive_ensemble_cal"):
        columns = [f"{model}_{outcome}_p" for outcome in ("home", "draw", "away")]
        values = output[columns].to_numpy()
        assert (values >= 0).all()
        assert (values <= 1).all()
        assert abs(values.sum(axis=1) - 1.0).max() < 1e-6
