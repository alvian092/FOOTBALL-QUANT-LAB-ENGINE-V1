# EPL pilot dataset — provenance and limitations

## Dataset
- Competition: English Premier League (EPL)
- Rows: 6130 settled matches
- Seasons: 2010-11 through 2026-27 (current season partial)
- Current-season coverage: 50 settled matches as present in the source snapshot retrieved 2026-11-14.
- Format: CSV, one row per settled fixture
- xG: not supplied; this dataset is goals/results only.
- Odds: not supplied; do not use this file alone for market ROI, CLV, or value-scanner claims.

## Source and licence
- Source: https://github.com/openfootball/football.json
- Data and schema are dedicated to the public domain under CC0-1.0: https://github.com/openfootball/football.json/blob/master/LICENSE.md
- Retrieved from the source repository on 2026-11-14.
- Team labels have been normalized by removing trailing " FC" and " AFC" suffixes; no other team-name transformations were applied.
- Score fields were normalized from either `score.ft` or the legacy two-integer `score` array.
- Unplayed fixtures and rows without a valid full-time score were excluded.
- Match IDs are deterministic from season, date, home team, and away team.

## Source file blob SHAs
- 2010-11/en.1.json — 9fe7c43945ed5bb7249ae2488397262d0d72ca88 (380 listed fixtures)
- 2011-12/en.1.json — 72c0c8ca26ce53bdafb71a3d12ea4e51fe8790a3 (380 listed fixtures)
- 2012-13/en.1.json — b9fba456fccee350f06ced85600229dfe8d895d4 (380 listed fixtures)
- 2013-14/en.1.json — f8f06a0494c657638a6ad055effdf65c33c164c3 (380 listed fixtures)
- 2014-15/en.1.json — 4ac8e36323109731fb60fc3db678a0b358074e12 (380 listed fixtures)
- 2015-16/en.1.json — d7384b9c162eea51ea0da500b8f6c73a170ee0bd (380 listed fixtures)
- 2016-17/en.1.json — b09de9d55b17571e6c9c64fefe83522dfc02ce6f (380 listed fixtures)
- 2017-18/en.1.json — c0ccb9c85503d1e24482be7c5117f6277c5a0e2f (380 listed fixtures)
- 2018-19/en.1.json — 334909dfb533959f15a6d2bf08a096adcfa7d21f (380 listed fixtures)
- 2019-20/en.1.json — f1af09cf49fb2ac69d17acfe76c030c67ae9c1dc (380 listed fixtures)
- 2020-21/en.1.json — 081509058af75c0c012d64e980dc4fa53720d933 (380 listed fixtures)
- 2021-22/en.1.json — 5eadd9d91c71d45634f3969192ad908653681b23 (380 listed fixtures)
- 2022-23/en.1.json — 0b6230e60d8cd7402810a5f515f6762d8b385b74 (380 listed fixtures)
- 2023-24/en.1.json — f62b11dfaf06cffd1236c2777caebb880a0cfea1 (380 listed fixtures)
- 2024-25/en.1.json — 504f2e732fbc3c9caeb98e4382b13b094453b840 (380 listed fixtures)
- 2025-26/en.1.json — 41e223ee5954e6b634beb4bc70d7188b3183c665 (380 listed fixtures)
- 2026-27/en.1.json — 60e281f0fcc51cfca1a901353666a688cf62a8ba (380 listed fixtures)

## Validation at import
- Settled rows: 6130
- Duplicate match IDs: 0
- Negative or non-integer scores: excluded
- Each completed season 2010-11 through 2025-26: 380 listed fixtures and 380 settled scores
- 2026-27: 380 listed fixtures; 50 settled scores, remaining fixtures excluded.

## Important temporal caveat
The source provides match dates but not authoritative kickoff timestamps or historical data-availability timestamps. The V4 walk-forward test must therefore exclude all same-date fixtures from one another's training history. This dataset is suitable for result-based chronological backtesting, not intraday or kickoff-time reconstruction.
