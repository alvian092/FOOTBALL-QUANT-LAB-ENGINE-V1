# International Football V1 — Initial Walk-Forward Evaluation

**Evaluation run:** GitHub Actions run [43584616204](https://github.com/alvian092/FOOTBALL-QUANT-LAB-ENGINE-V1/actions/runs/43584616204)  
**Result:** international-validation job passed.  
**Scope:** selected senior men's internationals only.

## Data coverage

- Source: `openfootball/internationals`, CC0-1.0 community-maintained mirror; not an official FIFA feed.
- Date range in this snapshot: 2000-01-04 to 2026-03-31.
- Settled matches parsed: 19,984.
- Out-of-sample predictions: 19,934 after a 50-match minimum history.
- Match classes: 8,318 friendlies; 11,666 competitive matches.
- Friendly Elo update multiplier: 0.55. This is a heuristic, not a fitted optimum.

## Baseline Elo walk-forward

| Metric | Result |
|---|---:|
| Brier score (multiclass) | 0.57010 |
| Log loss | 0.95964 |
| Accuracy | 56.23% |

Every fixture on a calendar date is forecast before any result from that date updates ratings. These are historical predictive metrics, not evidence of betting profitability.

## Competition-segment performance

Metrics below use the same 19,634 post-warm-up OOS predictions for raw and calibrated comparisons. Temperature scaling is fitted expanding-window using only earlier OOS predictions and outcomes. The first 300 OOS matches are excluded from calibrated evaluation.

| Segment | N | Raw Brier | Calibrated Brier | Raw Log Loss | Calibrated Log Loss | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Friendly | 8,162 | 0.60286 | 0.60563 | 1.00697 | 1.01328 | 0.04136 | 0.06350 |
| Qualification | 9,156 | 0.53341 | 0.50744 | 0.90674 | 0.87011 | 0.11911 | 0.02091 |
| Tournament | 2,316 | 0.59247 | 0.59494 | 0.99151 | 0.99754 | 0.03637 | 0.06198 |

## Overall prequential calibration

| Metric | Raw | Calibrated |
|---|---:|---:|
| Brier score | 0.56925 | 0.55858 |
| Log loss | 0.95841 | 0.94466 |
| Accuracy | 56.30% | 56.30% |
| ECE | 0.07191 | 0.03129 |

Interpretation:
- Calibration materially improved aggregate Brier, log loss, and ECE on this post-warm-up sample.
- Improvements are concentrated in qualification matches.
- Friendlies and tournament fixtures became worse on Brier, log loss, and ECE after the single global temperature adjustment. This warns against assuming one global calibrator fits every competition type.
- Temperature scaling does not change the predicted class ranking, so accuracy remains unchanged.
- Segment sample sizes, particularly individual tournaments, require uncertainty intervals and future-period confirmation before tuning or deployment.

## Odds, ROI, CLV and drawdown

The market ledger contained 8,075 eligible odds rows dated 2026-06-11 through 2026-11-06. The international results snapshot ended 2026-03-31.

| Matching diagnostic | Result |
|---|---:|
| Exact team-pair candidates | 0 |
| Matches within ±1 day | 0 |
| Qualifying bets | 0 |

**ROI, yield, CLV and drawdown are not measurable in this run.** The market ledger's date window does not overlap the international results snapshot. This is a coverage failure, not a zero-ROI or loss-free result. No profitability claim is made.

## Next validation gates

1. Extend and independently verify international results coverage through the odds-ledger period, preserving pre-match dates and senior A-international scope.
2. Add competition-specific prequential calibration only if there is sufficient historical calibration sample; otherwise retain the global model as a baseline and flag weak segments.
3. Add paired bootstrap confidence intervals by date/competition segment.
4. Re-run the market matcher only when actual international fixtures overlap the odds ledger; inspect aliases, kickoff timestamps, and price capture timing.
5. Keep **NO VALUE = NO BET**. Historical calibration or predictive accuracy is not a substitute for matched odds, positive conservative EV, CLV validation, and adequate forward sample size.
