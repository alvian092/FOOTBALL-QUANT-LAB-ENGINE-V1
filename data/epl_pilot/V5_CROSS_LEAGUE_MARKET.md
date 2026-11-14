# V5 Cross-League Validation & Market Intelligence

## Data sources and permitted use

- Cross-league match results: [openfootball/football.json](https://github.com/openfootball/football.json), public-domain CC0-1.0.
- Historical odds and closing prices: [TipsAudit public track record](https://github.com/tipsaudit/track-record-public), CC-BY-4.0. Attribution: TipsAudit.
- The TipsAudit ledger is a provider-selected prediction sample, not a complete league fixture census. It includes Pinnacle pre-kickoff price observations and closing odds; use only rows with valid results, no void reason, and `teams_reversed_after_logging=0`.
- We intentionally do not ingest football-data.co.uk odds into the automated AI/model pipeline because its current data page restricts use to private individuals and excludes commercial or AI training products. Obtain written permission before using it for this purpose.

## Reproduce

```bash
python -m quantlab.fetch_cross_league --output-dir data/cross_league --start-year 2010 --end-year 2026
python -m quantlab.cross_league \
  --input bundesliga=data/cross_league/bundesliga.csv \
  --input laliga=data/cross_league/laliga.csv \
  --input seriea=data/cross_league/seriea.csv \
  --input ligue1=data/cross_league/ligue1.csv \
  --input eredivisie=data/cross_league/eredivisie.csv
python -m quantlab.market_backtest \
  --predictions artifacts/cross_league/cross_league_oos_predictions.csv \
  --odds data/market/tipsaudit-predictions-all.csv \
  --model-prefix model_adaptive --min-ev 0.03
pytest -q
```

## Cross-league protocol

- Each competition is modelled independently; no team ratings or goal rates are shared across leagues.
- Predictions use strictly earlier match dates only. All fixtures on the same date are predicted before that date's results enter training history.
- Poisson, Dixon-Coles, Elo and fixed V4 ensemble are scored; calibration and adaptive OOF weights are fitted separately within each league using prior OOS dates only.
- Report per-league metrics first. Pooled metrics are descriptive and must not hide differences in league strength or scoring environment.
- Minimum 100 prior results for base predictions and 300 prior OOS predictions before adaptive calibration/weights.

## Odds / CLV protocol

- The TipsAudit monthly snapshot used here covers 11 June–6 November 2026. Therefore the results fetch must include settled 2026–27 fixtures (`end-year 2026`); stopping at 2025–26 creates a disjoint time window and guarantees zero temporal overlap. The current season is incomplete and only settled matches are used.

- Join on normalized home/away team names and kickoff date within ±1 UTC day. Only exact team-pair matches are retained; inspect join coverage and unmatched rows.
- Exclude void results and fixtures whose teams were reversed after the logged prediction.
- Entry price is the provider's recorded Pinnacle pre-kickoff observation (`pin_open_*` in this provider's schema); it is not necessarily the bookmaker's literal market-opening price.
- Closing price is the same provider's Pinnacle closing price (`pin_close_*`).
- Convert 1X2 prices to proportional no-vig market probabilities before comparing Brier/Log Loss.
- Expected value per selection is model probability × entry decimal odds − 1.
- Select at most one outcome per fixture: highest positive EV, default minimum 3%; flat one-unit stakes.
- ROI/yield = net profit / total stake. CLV = (entry odds / closing odds − 1) × 100 for the same outcome. Max drawdown is peak-to-trough cumulative flat-stake profit.
- Report sample coverage, bet count, ROI, hit rate, mean EV, mean CLV, positive-CLV share and maximum drawdown.

## Interpretation safeguards

- The provider's prediction ledger contains only a subset of fixtures; results are conditional on matched coverage, not representative of every market or league.
- The pre-kickoff price observation may have been added after the prediction was logged but is documented as quoted before kickoff. It is not guaranteed to be executable at the precise prediction timestamp.
- CLV is a price comparison, not proof of profitability.
- The initial 3% EV threshold is a research setting, not an optimized betting threshold.
- Do not promote the model to live staking based on one historical sample. Require untouched forward tracking, sufficient bet count, positive CLV, stable ROI and acceptable drawdown.
