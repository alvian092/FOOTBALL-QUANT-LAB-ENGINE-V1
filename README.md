# Football Quant Lab Engine V1

A data-first football probability and bookmaker-market analysis workspace built around **NO VALUE = NO BET**.

## V1 foundation
- Poisson score matrix with Dixon-Coles low-score adjustment.
- 1X2, totals, BTTS, and correct-score probability calculations.
- Implied probability, overround, no-vig normalization, EV, and value difference.
- Conservative flat staking and fractional Kelly utilities.
- CSV validation, duplicate checks, SQLite match-history persistence.
- Prediction record storage, Brier score, log loss, and bankroll metric utilities.
- Streamlit dashboard designed for mobile browsers.

## Important limitations
The first estimator derives expected-goal rates from historical goals. Goals are not xG. xG/xGA feature engineering, recency weighting, league-specific calibration, lineup/injury feeds, live odds history, Asian handicap settlement, closing-line capture, and fully validated out-of-sample uncertainty are not yet production-ready. Value labels are provisional and are not betting advice. Do not stake based on an uncalibrated model.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
Required columns: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`. Optional: `match_id`.

## Tests
```bash
pytest -q
```

## Persistence
Imported matches and prediction records are stored in `data/football_quant_lab.sqlite3`. This is persistent on a host with persistent storage; ephemeral hosting may erase local files. Back up the database and use a persistent disk or managed database in deployment.

## Roadmap
1. Complete immutable pre-match prediction and settlement workflow.
2. Add xG/xGA, recency-weighted strengths, league priors, and lineup adjustment.
3. Implement Asian handicap and totals settlement.
4. Add calibration, rolling out-of-sample backtesting, CLV, ROI, yield, and drawdown.
5. Add migrations, dataset snapshots, backup/restore validation, and deployment CI.

Probabilities are estimates, not guarantees. Never chase losses or use martingale staking. If data quality or calibration is inadequate, the correct output is **NO BET**.
