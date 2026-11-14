# EPL V4 Expanding Calibration & OOF Ensemble Weights

## Reproduce

```bash
python -m quantlab.epl_calibration \
  --input artifacts/epl_v4_backtest/epl_v4_oos_predictions.csv \
  --output-dir artifacts/epl_v4_calibration \
  --min-history 300 \
  --refit-every-dates 10 \
  --bootstrap-samples 20000 \
  --seed 20261114
pytest -q
```

## Leakage controls

- All input predictions are already out-of-sample walk-forward predictions.
- Each match date is calibrated and weighted using only strictly earlier prediction dates and their settled outcomes.
- First 300 prior OOS fixtures use neutral temperature 1.0 and provisional weights 0.4 / 0.4 / 0.2.
- Parameters are refit every 10 distinct match dates after the minimum history threshold.
- Each base model gets its own multiclass temperature parameter, fitted by prior-period log loss.
- Non-negative ensemble weights constrained to sum to one are fitted by prior-period log loss on historical calibrated base predictions.
- The adaptive blend gets a separate temperature fitted only on prior adaptive-blend predictions.
- The same-date fixtures are never used to fit parameters for one another.
- Paired confidence intervals resample calendar-date clusters, comparing identical fixtures. Negative left-minus-right differences favor the left-hand model.

## EPL pilot results

6,030 OOS fixtures, 1,745 date clusters, 17 season labels. Brier and log loss are lower-is-better; accuracy is higher-is-better. ECE is the mean of one-vs-rest classwise expected calibration errors (10 equal-width probability bins).

| Model | Brier | Log loss | Accuracy | Mean classwise ECE |
|---|---:|---:|---:|---:|
| Raw Elo | 0.59002 | 0.99089 | **53.40%** | 0.03092 |
| Calibrated Elo | 0.58566 | 0.98645 | **53.40%** | 0.01581 |
| Fixed V4 Ensemble | 0.59914 | 1.00637 | 51.67% | 0.03198 |
| Adaptive OOF Ensemble (raw) | **0.58551** | **0.98538** | 52.97% | 0.01392 |
| Adaptive OOF Ensemble (calibrated) | 0.58636 | 0.98656 | 52.97% | **0.01213** |

Paired date-cluster bootstrap, 20,000 replicates:

| Comparison (left minus right) | Brier difference, 95% CI | Log-loss difference, 95% CI |
|---|---:|---:|
| Adaptive OOF Ensemble (raw) − Raw Elo | -0.00450 [-0.00653, -0.00242] | -0.00550 [-0.00854, -0.00236] |
| Calibrated Elo − Raw Elo | -0.00436 [-0.00639, -0.00236] | -0.00443 [-0.00754, -0.00134] |
| Adaptive OOF Ensemble (calibrated) − Raw Elo | -0.00365 [-0.00564, -0.00162] | -0.00433 [-0.00731, -0.00126] |
| Adaptive OOF Ensemble (raw) − Fixed V4 Ensemble | -0.01362 [-0.01720, -0.01007] | -0.02099 [-0.02657, -0.01543] |

Late-period parameter snapshot (not a constant global weight):
- Poisson temperature: approximately 1.579
- Dixon-Coles temperature: approximately 1.591
- Elo temperature: approximately 0.777
- Adaptive ensemble temperature: approximately 0.984
- Adaptive weights: Poisson 0.029, Dixon-Coles 0.206, Elo 0.765

## Interpretation

- Calibration materially improves probability quality for all three base models on this pilot; Elo remains the strongest standalone base model.
- Expanding OOF weights shift toward Elo over the later period and outperform the fixed 0.4 / 0.4 / 0.2 blend on both proper scoring rules.
- The raw adaptive blend has the best Brier and log loss among the tested variants; calibrating the final blend slightly improves ECE but slightly worsens Brier and log loss. Keep both outputs available rather than assuming one is universally best.
- Accuracy is unchanged by temperature scaling and remains a useful but coarse secondary metric.
- These are historical predictive results, not evidence of positive betting ROI.

## Limitations and next validation

- Goals/results-only dataset; no xG or bookmaker odds.
- No ROI, yield, CLV, market edge or bet selection has been tested.
- OOF weights optimize prior log loss, but are not yet compared against alternative regularization, weight-stability constraints or a nested hyperparameter search.
- ECE depends on binning and is reported as a descriptive calibration metric.
- Adjacent-date serial dependence may remain despite matchday-cluster resampling.
- Before live use, validate on another competition and a newer untouched time window, then integrate contemporaneous bookmaker prices and CLV tracking.
