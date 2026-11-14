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
