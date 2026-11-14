# International Football V1 — Extended Walk-Forward Evaluation

**Evaluation run:** [GitHub Actions run 43584942210](https://github.com/alvian092/FOOTBALL-QUANT-LAB-ENGINE-V1/actions/runs/43584942210)  
**International validation:** success.  
**Scope:** senior men's full internationals; women's rows are filtered out.

## Data coverage

- Primary source: Mon Sport Info international-results CSV, derived from martj42 and marked CC0-1.0.
- Fallback source: openfootball/internationals (CC0-1.0) if the primary feed is unavailable.
- Primary snapshot page update: 2026-07-20; latest settled match in this filtered run: 2026-07-15.
- Date range in this run: 2000-01-04 to 2026-07-15.
- Settled senior men's matches: 25,456.
- Out-of-sample predictions: 25,406 after a 50-match minimum history.
- Neutral-venue matches: 7,327. The model sets home advantage to zero for these fixtures.
- Friendly Elo update multiplier: 0.55. This is a heuristic, not a fitted optimum.

The source is a community-maintained historical dataset, not an official FIFA feed. The source's men's full-international classification is used, but FIFA A-match status is not independently verified for every friendly.

## Baseline Elo walk-forward

| Metric | Result |
|---|---:|
| Brier score (multiclass) | 0.56900 |
| Log loss | 0.95831 |
| Accuracy | 56.71% |

Every fixture on a date is forecast before any result from that date updates ratings. Neutral fixtures receive no home-advantage adjustment. These are historical predictive metrics, not evidence of betting profitability.

## Competition-segment performance

Metrics below use the same 25,104 post-warm-up OOS predictions for raw and calibrated comparisons. Expanding temperature scaling is fitted only on earlier OOS forecasts and outcomes. The first 300 OOS predictions are excluded from calibrated evaluation.

| Segment | N | Raw Brier | Calibrated Brier | Raw Log Loss | Calibrated Log Loss | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Friendly | 8,316 | 0.60130 | 0.60739 | 1.00474 | 1.01656 | 0.03681 | 0.07449 |
| Qualification | 9,845 | 0.52939 | 0.50217 | 0.90116 | 0.86345 | 0.12045 | 0.01665 |
| Tournament | 6,943 | 0.58370 | 0.56998 | 0.97977 | 0.96175 | 0.08304 | 0.01033 |

## Overall prequential calibration

| Metric | Raw | Calibrated |
|---|---:|---:|
| Brier score | 0.56823 | 0.55578 |
| Log loss | 0.95721 | 0.94136 |
| Accuracy | 56.80% | 56.80% |
| ECE | 0.07910 | 0.02868 |

Interpretation:
- Calibration improved aggregate Brier, log loss, and ECE on the post-warm-up sample.
- Qualification and tournament segments improved on all reported calibration metrics.
- Friendlies became worse on Brier, log loss, and ECE after global temperature scaling. A separate friendly calibrator may be worth testing, but only with a properly separated validation sample and adequate history.
- Temperature scaling does not change predicted class ranking, so accuracy remains unchanged.
- Segment sample sizes and uncertainty intervals must be reviewed before tuning or production use.

## Odds, ROI, CLV and drawdown

The market ledger contained 8,075 eligible price rows from 2026-06-11 to 2026-11-06. The international results snapshot overlaps part of this window (through 2026-07-15), but the ledger's 86 distinct `league` labels are club competitions; it does not provide national-team fixture odds in this snapshot.

| Matching diagnostic | Result |
|---|---:|
| Eligible market odds rows | 8,075 |
| Exact team-pair candidates | 0 |
| Matches within ±1 day | 0 |
| Qualifying bets | 0 |

**ROI, yield, CLV and drawdown are not measurable.** This is an international-market coverage gap, not zero ROI or a loss-free result. The market backtest correctly refuses to invent a return when there are no matched national-team prices.

## Next validation gates

1. Add a licensed international odds source or capture real pre-kickoff international prices with closing prices for the same fixtures.
2. Extend international results coverage beyond 2026-07-15 and independently cross-check classifications against FIFA's international match archive.
3. Test segment-specific calibration only with expanding or nested chronological validation; avoid tuning on the same final evaluation sample.
4. Add paired bootstrap confidence intervals by date and competition segment.
5. Keep **NO VALUE = NO BET**. Predictive accuracy and calibration do not establish a betting edge without matched odds, conservative EV, CLV validation, and adequate forward sample size.
