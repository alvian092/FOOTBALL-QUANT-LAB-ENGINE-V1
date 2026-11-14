# EPL V4 Paired Bootstrap & Season Robustness

## Reproduce

After generating the walk-forward OOS prediction CSV:

```bash
python -m quantlab.epl_robustness \
  --input artifacts/epl_v4_backtest/epl_v4_oos_predictions.csv \
  --output-dir artifacts/epl_v4_robustness \
  --bootstrap-samples 20000 \
  --seed 20261114
pytest -q
```

Outputs:
- `epl_v4_paired_bootstrap.json`: paired Elo-minus-Ensemble loss differences and date-cluster bootstrap intervals.
- `epl_v4_season_robustness.csv`: Brier, log loss, and accuracy for Elo and Ensemble in every EPL season.

## Method

- Comparison is paired: both models are scored on the exact same out-of-sample fixtures.
- Resampling unit is the UTC calendar match date, not the individual fixture. All fixtures on a sampled date travel together, preserving within-date dependence.
- 20,000 percentile-bootstrap replicates; fixed seed 20261114.
- Reported differences are Elo minus Ensemble. Negative values favor Elo.
- Per-season summaries are descriptive, not separate significance tests. No multiple-testing correction is applied.
- The bootstrap probability is descriptive; it is not labeled as a formal null-hypothesis p-value.

## Results on the EPL pilot

Dataset: 6,030 OOS fixtures across 17 season labels (2010-11 through 2026-27), 1,745 distinct match dates.

| Paired metric | Elo − Ensemble | Cluster-bootstrap 95% interval | Result |
|---|---:|---:|---|
| Brier | -0.009120 | [-0.013921, -0.004351] | Elo lower |
| Log loss | -0.015484 | [-0.022893, -0.008162] | Elo lower |

Season-level head-to-head:
- Elo has lower Brier in **14 of 17** seasons.
- Elo has lower log loss in **15 of 17** seasons.
- Elo has higher accuracy in **13 of 17** seasons.

The pattern supports using Elo as the current predictive baseline for this historical EPL goals/results pilot. It does not establish that Elo is profitable for betting.

## Limitations

- Results/goals only; no xG or bookmaker odds.
- No ROI, yield, CLV, or value-bet profitability claim is possible from this dataset.
- The test compares Elo against the current fixed-weight Ensemble; it does not prove Elo is optimal versus a newly trained, leakage-safe ensemble.
- Season comparisons are descriptive and the final 2026-27 label contains only 50 OOS fixtures.
- Bootstrap uncertainty accounts for same-date clustering, not all possible serial dependence across adjacent match dates.
- Calibration and market-price evaluation remain separate next steps.
