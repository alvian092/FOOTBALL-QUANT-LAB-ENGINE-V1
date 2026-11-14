# EPL V4 Walk-Forward Pilot

Run from the repository root:

```bash
pip install -r requirements.txt
python -m quantlab.epl_backtest --input data/epl_pilot/epl_results_2010_2026.csv
python -m quantlab.epl_robustness --input artifacts/epl_v4_backtest/epl_v4_oos_predictions.csv
pytest -q
```

The backtest writes:
- `artifacts/epl_v4_backtest/epl_v4_oos_predictions.csv`
- `artifacts/epl_v4_backtest/epl_v4_backtest_summary.json`

The robustness runner writes:
- `artifacts/epl_v4_robustness/epl_v4_paired_bootstrap.json`
- `artifacts/epl_v4_robustness/epl_v4_season_robustness.csv`

## Evaluation design

- Predictions are generated using only matches with dates strictly earlier than the target date.
- All fixtures on a calendar date are predicted before any result from that date enters history.
- Minimum training history defaults to 100 prior fixtures.
- Team scoring/conceding rates use up to 20 prior home/away venue fixtures with exponential recency weighting (five-match half-life).
- Components: Poisson (rho=0), Dixon-Coles (rho=-0.08), Elo (K=20, home advantage 60, draw prior 0.26), and a fixed V4 blend (0.4 / 0.4 / 0.2).
- Elo updates are batched by historical date so fixture ordering within a date cannot change ratings.
- Metrics: multiclass Brier score, log loss, accuracy, and fixture-level bootstrap 95% interval for Brier.
- Elo-versus-Ensemble robustness uses paired percentile bootstrap resampling by calendar match date (20,000 samples), plus descriptive season-level metrics.

## Interpretation limits

- This is a goals/results-only model test. The source has no xG, so goal rates are not xG.
- The fixed ensemble weights are provisional V4 defaults, not weights fitted on this test set.
- The original fixture-level bootstrap interval does not account for temporal dependence; the paired robustness interval clusters by match date but may still retain serial dependence across dates.
- There are no bookmaker odds in this dataset. Do not interpret predictive metrics as ROI, yield, CLV, or evidence of betting profitability.
- No bet recommendations are produced by this backtest.
- See [ROBUSTNESS.md](ROBUSTNESS.md) for the paired Elo-versus-Ensemble results and season breakdown.
