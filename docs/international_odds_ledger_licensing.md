# International odds ledger: source and licence gate

Status: **candidate identified; not yet integrated**.

## Candidate: Olivier Sports Data — International Football Timestamped Odds

- Dataset card: https://huggingface.co/datasets/oliviersportsdata/International-Football-Timestamped-Odds-Pinnacle-Match-Stats
- Product listing: https://sportsdataolivier.gumroad.com/l/hethzg
- Published coverage: 1,474 national-team matches within an 11,118-match product, 18 competitions, 2016–2026 (the product page states coverage through 2026-09-30).
- National-team competitions listed include UEFA Nations League, European Championship, FIFA World Cup, Copa America, AFC Asian Cup and Africa Cup of Nations. The published list does not establish comprehensive international friendlies or qualification coverage.
- Markets/data: named-bookmaker FT 1X2, Asian handicap and totals, opening/closing prices with timestamps, timestamped pre-match/in-play quotes, selected match/team data.
- Named bookmakers include Bet365, Pinnacle, Unibet, Betfair and Betsson; additional bookmaker snapshots are separate observations, not necessarily closing prices.
- Published licence: single commercial user. Research, modelling and internal use are permitted; redistribution and resale of raw files are prohibited. The public 20-match sample is for exploration/evaluation only.
- Published product price at review: USD 499. Confirm current terms and obtain the licence directly from the vendor before acquisition.

## Public sample findings and matching implications

- The public sample has 20 selected matches (latest final of each listed competition plus two kickoff-time error cases); it is not a representative random sample for profitability testing.
- The sample documents two fixtures with incorrect source kickoff timestamps, including one recorded on the wrong UTC date. Date-only or broad ±1-day matching is therefore unsafe.
- The odds rows and match-level tables use provider `Match_ID`; the dataset card explicitly says all tables join on this key. Preserve it as a string.
- Timestamped odds include `Book`, `Market`, `Phase`, `Timestamp_UTC`, 1X2 prices and `Suspended`. Filter to FT 1X2, pre-match, unsuspended, complete odds strictly before verified kickoff.
- A bookmaker's closing quote is its last recorded pre-match quote, and the median gap to kickoff varies substantially by bookmaker. Do not treat every close as a synchronized kickoff price.

## Why it is not integrated yet

1. No paid full-dataset licence or files have been supplied to this repository.
2. The 1,474 national-team matches are a subset of the wider 11,118-match product; this is not a complete national-team schedule.
3. Coverage is selected international competitions, not a demonstrated complete archive of FIFA A friendlies and all qualification fixtures.
4. The source is a single-provider archive. Keep bookmaker identity, market, phase and exact timestamp; do not treat snapshots as synchronized closing lines.
5. A vendor licence permitting internal modelling does not automatically permit committing source files to a public Git repository.

## Integration acceptance checklist

- Obtain and retain written proof of licence/receipt and verify the permitted user/entity, internal-use, storage, retention and derived-output terms.
- Keep raw paid data outside the public repository unless the licence explicitly permits redistribution.
- Validate the supplied manifest/checksums, schema, competition coverage, match IDs, result settlement conventions, timezone and bookmaker timestamps.
- Join vendor tables by provider `Match_ID`; reconcile model predictions to the vendor match master using ordered teams, competition context and verified kickoff UTC. Reject ambiguous matches and quarantine kickoff outliers; do not join prediction rows directly to quote rows.
- Follow the full matching and backtest contract in [international_odds_matching_spec.md](international_odds_matching_spec.md).
- Restrict the primary pre-match backtest to FT 1X2 prices recorded before verified kickoff; retain opening prices separately for CLV.
- Report match coverage by category, competition, year, bookmaker and market. Missing odds must remain missing, never become zero.
- Require at least 100 settled qualifying bets for an initial research assessment; report confidence intervals and drawdown, and avoid production claims on small samples.
- Keep friendlies as their own segment. Require a verified FIFA A-match/competition classification where possible; do not assume every friendly in a community results feed is a FIFA A international.

## Current project decision

The existing CC-BY-4.0 market ledger produces zero team-national match joins. It is not a valid international odds sample. Until a licensed, coverage-audited international ledger is supplied, international ROI, yield, CLV and drawdown are **not measurable**. This is a data-coverage limitation, not a zero-return result.

**Decision rule: NO VALUE = NO BET.**
