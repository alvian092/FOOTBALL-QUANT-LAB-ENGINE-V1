# Football Quant Lab

A data-first football probability and bookmaker-market analysis workspace built around **NO VALUE = NO BET**.

## Current engine
- Poisson score matrix with Dixon-Coles low-score adjustment.
- 1X2, totals, BTTS, correct-score and Asian handicap probability utilities.
- CSV ingestion with optional xG/xGA columns and strict validation.
- Venue-specific home/away form and exponential recency weighting (configurable recent 5–20 matches).
- Out-of-sample home-win Brier score, log loss, calibration bins, bootstrap Brier confidence interval and Wilson hit-rate interval.
- Prediction snapshots, manual pre-kickoff closing-odds capture, settlement, ROI/yield, CLV and maximum drawdown.
- SQLite transactional backup and validated restore.
- Streamlit dashboard with mobile-friendly layout and Android browser access.

## Data requirements
Required CSV columns: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`. Optional: `match_id`.

For xG-aware modeling, provide **both** `home_xg` and `away_xg` as non-negative numeric values. xG values must come from a consistent, identified data provider. If xG is absent, the engine falls back to goals-derived venue rates and labels the model accordingly; it does not fabricate xG.

Recency weighting uses an exponential half-life measured in matches. Venue splits use the selected home team's home fixtures and selected away team's away fixtures. A minimum of five matching venue fixtures is required by the dashboard.

## Metrics definitions and caveats
- **ROI / yield:** total realized net P/L divided by total stake on settled selections.
- **CLV:** average `entry_odds / closing_odds - 1`; positive means the selected odds shortened versus the recorded close. Closing odds must be entered manually before kickoff.
- **Maximum drawdown:** largest peak-to-trough decline in cumulative realized P/L; percentage drawdown is relative to the prior positive peak.
- **Calibration:** currently evaluated as a home-win binary event using saved pre-match probabilities and settled results. Brier confidence intervals use bootstrap resampling; hit-rate intervals use Wilson's method. Small samples produce wide, unstable intervals.

These are retrospective metrics, not guarantees of future performance. Predictions recorded by older model versions are not automatically re-generated or calibrated. Asian handicap probabilities exist, but AH settlement is not yet integrated into the prediction tracker.

## V3 — Advanced Backtesting & Market Intelligence

The V3 workstream adds research tools that compare a chronological model forecast with a no-vig bookmaker benchmark. These tools do not create an automatic betting signal.

### Walk-forward protocol
- Fixtures are sorted chronologically and each prediction sees only rows with a strictly earlier date.
- Same-day fixtures are excluded from one another's training history to prevent result leakage.
- When a league column is supplied, training history is scoped to that league.
- min_train controls the minimum prior match count; optional max_train supports rolling rather than expanding windows.
- The predictor is a callback so alternative models can be tested under the same temporal protocol.

### Bookmaker benchmark
For rows containing home_odds, draw_odds, and away_odds, the backtest removes the overround by normalizing inverse decimal odds. It reports model and market multiclass Brier score, log loss and accuracy on the same fixtures, plus paired bootstrap intervals for the model-minus-market loss improvement. Positive improvement means the model has lower loss. An interval crossing zero does not establish superiority.

### Historical odds snapshots
The Market Intelligence dashboard accepts CSV fields:
match_id, market, selection, odds, captured_at, plus optional bookmaker.
Each timestamped bookmaker/market snapshot must contain at least two unique selections. Capture timestamps should include an explicit timezone. Snapshots are stored as immutable records; duplicate timestamp/selection rows are ignored by SQLite. Movement reports compare first and latest complete snapshots, including no-vig probability shifts and overround changes.

### Segmented performance
Settled prediction records can be grouped by market (and by league when that metadata is available). Segments below the configured minimum sample are explicitly labelled SMALL SAMPLE. Segment ROI and CLV are descriptive, not proof of a repeatable edge.

### Historical match CSV additions
The match CSV may include league. It is persisted with match history and enables league-specific walk-forward training. Do not combine competitions with materially different scoring environments unless you intentionally want a pooled benchmark.

### Important limitations
- No live bookmaker odds feed is bundled. Market movement depends on the timestamps and accuracy of uploaded snapshots.
- Walk-forward uses the current transparent Poisson/xG feature estimator; it is not a fitted machine-learning training pipeline.
- Bookmaker comparison is available only for matches with a complete 1X2 odds triplet.
- Bootstrap intervals quantify uncertainty from the tested sample; they do not guarantee future performance.
- Avoid tuning thresholds on the same period used to report final performance. Keep a final untouched time period for confirmation.

## V4 — Predictive Edge & Automated Market Scanner

V4 introduces a transparent three-component probability ensemble and conservative 1X2 market screening.

### Model components
- **Poisson:** independent goal model with Dixon-Coles correction disabled.
- **Dixon-Coles:** low-score dependence correction applied to the same expected-goals inputs.
- **Elo:** sequential historical ratings replayed in date order, with explicit home advantage and draw prior.
- **Ensemble:** non-negative weighted linear probability pool. Default dashboard weights are provisional. Fit weights with out-of-fold component forecasts and outcomes using the OOF weight-fitting tool.

The Elo implementation replays only the match history supplied to it. It does not use future fixtures. Model components are not independent sources when Poisson and Dixon-Coles share expected-goals inputs; ensemble diversity should not be overstated.

### Calibration
The dashboard accepts an OOF prediction CSV with date, outcome (0=home, 1=draw, 2=away), and model_home_p/model_draw_p/model_away_p. The oldest 70% fits a scalar temperature; the newest 30% is held out. Brier, log loss and expected calibration error (ECE) are reported on the holdout before and after scaling. Keep this holdout untouched after evaluation. Temperature scaling is a global multiclass adjustment, not league-specific calibration.

### Conservative market scanner
The scanner requires one row per fixture with:
- match_id, optional league
- home_odds, draw_odds, away_odds
- model_home_p, model_draw_p, model_away_p
- lower_home_p, lower_draw_p, lower_away_p

The lower probability fields must be genuine out-of-sample lower confidence bounds generated by a documented bootstrap or other uncertainty procedure. The scanner does not fabricate intervals. A selection is only a VALUE CANDIDATE when model edge versus no-vig market probability meets the threshold, conservative EV is positive/above the configured floor, and bookmaker overround is within the configured limit. Otherwise it is NO BET.

### Data and operational limits
- No live bookmaker odds API is bundled. Upload actual odds snapshots/fixture prices; do not treat manually typed or stale prices as live market data.
- The default ensemble weights are illustrative until fitted on OOF predictions and confirmed on a separate untouched time period.
- Calibration and ensemble fitting require chronological OOF forecasts. Never fit weights, temperature, thresholds, or feature choices on the final holdout.
- A value candidate is a research flag, not a guaranteed edge or betting recommendation. Use conservative stakes and track realized results and closing-line value.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for Android access
1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and create an app from this repository.
3. Select branch `main`, main file `app.py`, then deploy.
4. Open the generated HTTPS app URL in Chrome on Android.
5. In Chrome, use **Add to Home screen** to create an app-like launcher shortcut.

The dashboard is responsive and runs in a mobile browser; there is no separate native Android APK. Before production use, configure persistent storage or an external managed database. Free/ephemeral hosting may reset local SQLite files when the app restarts or redeploys. Use the Backup & Restore page regularly. Do not commit the live database or secrets to GitHub.

## Backup and restore
Use **Backup & Restore** in the dashboard to download a consistent SQLite backup. Restore validates SQLite integrity and required tables before replacing the active database. Always download a fresh backup before restoring another file.

## Tests
```bash
pytest -q
python -m compileall -q app.py quantlab tests
```

## Safety / limitations
The engine is a research and record-keeping tool, not a guaranteed-pick system or financial advice. xG depends on user-supplied source quality. The current calibration report covers home-win probability only. CLV depends on accurate manually captured closing odds. ROI is meaningful only after a sufficiently large, representative sample of settled bets. No lineup/injury feed, live odds history, or automated data provider is bundled.

Never chase losses or use martingale staking. When data quality, calibration, or expected value is inadequate, the correct output is **NO BET**.

## V6 — International Football (senior men's)

The separate international module is intentionally isolated from domestic club ratings. It uses the CC0-1.0 Mon Sport Info international-results CSV derived from martj42, filters to men's full internationals, and falls back to the CC0 `openfootball/internationals` mirror if the primary feed is unavailable. The current feed snapshot is updated 2026-07-20; the latest settled match parsed in the latest run was 2026-07-15. It is not an official FIFA data feed, and FIFA A-match status is not independently verified for every friendly.

### Included competitions
- FIFA World Cup and World Cup qualification.
- UEFA European Championship, qualification, and Nations League.
- AFC Asian Cup and qualification; Africa Cup of Nations and qualification.
- Copa América; CONCACAF Gold Cup and qualification; OFC Nations Cup.
- Senior men's international friendlies.

### Modeling policy
- Independent national-team Elo ratings; club Elo and domestic league goal rates are never imported.
- Strict-date walk-forward: every fixture on a date is predicted before any result from that date updates ratings.
- Friendlies use a lower Elo K multiplier (0.55) to reduce the influence of matches with more variable squads and incentives. This is an initial heuristic to be tested, not a proven optimum.
- Competitive tournament multipliers are explicit and configurable in `quantlab/international.py`.
- The current baseline uses a fixed draw prior and modest home advantage. Neutral venue, lineups, player availability, travel, rest, and squad rotation are not yet modeled.

### Run
```bash
python -m quantlab.international fetch --output data/international/senior_men.csv --start-year 2000 --end-year 2026
python -m quantlab.international backtest --input data/international/senior_men.csv --output-dir artifacts/international_v1 --min-train 50
```

The first release is a results-only baseline. It reports multiclass Brier score, log loss, and accuracy; segmented evaluation and expanding prequential temperature calibration are also included. The initial market-ledger test found no national-team fixture odds in the supplied club-market ledger, so ROI, yield, CLV, and drawdown are not measurable until a suitable international odds source is connected. Women's, youth, Olympic, and other non-senior competitions are out of scope for this phase and require separate source validation.


### International evaluation: competition segments, calibration, odds and CLV

The international workflow produces three separate evaluation layers:
1. **Competition segments:** friendly, qualification, and tournament summaries, plus raw competition-folder breakdowns. Each reports sample size, multiclass Brier score, log loss, accuracy, and top-label ECE.
2. **Prequential calibration:** expanding temperature scaling. A fixture date is calibrated only with forecasts and outcomes from earlier dates; same-day fixtures share the same pre-date calibrator. The first 300 OOS predictions are warm-up by default and are excluded from calibrated evaluation.
3. **Market backtest:** the calibrated international probabilities are matched to the CC-BY-4.0 TipsAudit ledger and eligible Pinnacle pre-kickoff/closing prices. The report includes matched coverage, selected bets, flat-stake ROI/yield, hit rate, CLV, max drawdown, and model-versus-no-vig-market Brier/log loss.

Run the evaluation locally:
```bash
python -m quantlab.international_evaluation \
  --input artifacts/international_v1/international_oos_predictions.csv \
  --output-dir artifacts/international_v1 \
  --min-calibration-history 300 --refit-every-dates 10

python -m quantlab.market_backtest \
  --predictions artifacts/international_v1/international_calibrated_predictions.csv \
  --odds data/market/tipsaudit-predictions-all.csv \
  --model-prefix model_intl_calibrated --min-ev 0.03 \
  --output-dir artifacts/international_market_intelligence
```

Market matching is restricted to the ledger's recorded fixtures and prices. The ledger is not a complete fixture-market census, and its recorded pre-kickoff price is not necessarily the bookmaker's true opening price. If no fixtures match, ROI/CLV are **not measurable**—they must not be reported as zero or interpreted as a loss-free strategy. Positive historical ROI or CLV is not proof of future profitability.


Initial results and coverage limitations are documented in [`docs/international_v1_initial_evaluation.md`](docs/international_v1_initial_evaluation.md). The first CI run covered 2000-01-04 through 2026-03-31; the available odds ledger started later, so ROI and CLV were not measurable in that run.
