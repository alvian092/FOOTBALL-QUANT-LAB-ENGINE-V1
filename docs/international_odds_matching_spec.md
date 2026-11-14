# International odds fixture matching contract

Status: **design gate; no paid source files are present in this repository**.

## Source facts verified from the public sample card

The Olivier Sports Data international sample exposes a match-level `Match_ID` and timestamped odds rows keyed by the same `Match_ID`. The odds table includes `Book`, `Market`, `Phase`, `Timestamp_UTC`, `Price_Home`, `Price_Draw`, `Price_Away` and `Suspended`. The sample card explicitly states that all tables join on `Match_ID` and that the 20-match sample is selected to show edge cases, including two incorrect kickoff timestamps.

This is a better join key than fuzzy team-name/date matching. However, the current model's generated `match_id` is a local date/team string and is not the vendor's `Match_ID`; it must never be treated as interchangeable.

## Required matching stages

1. **Provider-native join first.** Join the vendor match master to opening/closing and timestamped odds only by provider `Match_ID`. Preserve the provider ID as a string (avoid spreadsheet numeric coercion).
2. **Cross-source fixture reconciliation.** Reconcile model predictions to the vendor match master using normalized home/away team identity, ordered home/away roles, competition context and kickoff UTC. Do not join predictions directly to odds quote rows.
3. **Kickoff-time integrity.** The two sample exceptions with incorrect source kickoff timestamps prove that date-only or ±1-day matching is unsafe. A candidate with a large time discrepancy must be quarantined and reconciled against the vendor's recorded pre-match cutoff / actual-played timestamp. Never silently shift kickoff dates.
4. **Ambiguity policy.** Require a unique candidate. If multiple model or provider fixtures satisfy the key, mark ambiguous and exclude pending manual review. Never choose the nearest candidate automatically when ties or collisions exist.
5. **Home/away orientation.** Require ordered team agreement. If source orientation is reversed, only allow explicit, audited reversal that swaps home/away odds and outcome labels together. Otherwise quarantine.
6. **Competition and cohort scope.** Retain provider competition, edition, stage and leg. Do not infer that every national-team fixture is a FIFA A friendly or qualifier. The published 1,474 national-team fixtures are drawn from selected international competitions, not a complete archive of all friendlies and qualifiers.
7. **Odds eligibility.** For a pre-match decision, use only FT 1X2 quotes with `Phase=Pre-match`, complete valid decimal prices, `Suspended=0`, and timestamp strictly before the verified kickoff/cutoff. Exclude in-play rows.
8. **Opening, entry and close.** Keep opening, chosen entry snapshot and closing prices with their own timestamps. A closing price is the last pre-match quote for that bookmaker/market; it may be minutes before kickoff and must not be presented as synchronized to kickoff. Never use post-kickoff data.
9. **Settlement.** Compare model outcome against the 90-minute result for FT 1X2; extra time and penalty shoot-outs are separate fields and must not alter a 90-minute market settlement. Void/abandoned/awarded matches require explicit settlement rules and quarantine when unresolved.
10. **Audit trail.** Emit a row-level match audit with prediction ID, provider Match_ID, exact/normalized team names, competition, both kickoff timestamps, time delta, orientation, bookmaker, market, quote timestamps, exclusion reason and match confidence. Keep excluded rows in diagnostics.

## Minimum output and safety checks

- Match counts at every stage: model predictions, provider fixtures, provider odds rows, unique candidates, unique accepted matches, ambiguous, reversed, kickoff-outlier and unmatched.
- Coverage by competition, edition, year, friendly/qualification/tournament cohort, bookmaker and market.
- Assert one accepted odds fixture per prediction and one prediction per provider Match_ID.
- Assert all selected pre-match odds timestamps precede verified kickoff; all odds are finite and >1.
- Assert all 3 1X2 outcomes exist for the same bookmaker, market and observation time.
- Report source-native and reconciled coverage separately. Missing odds remain missing; no zero imputation.
- Before any ROI/CLV result, publish the fixture audit and manually review all exceptions plus a random accepted-match sample.

## Backtest protocol

- Primary comparison: predeclared bookmaker and pre-match entry rule, not whichever bookmaker performs best after seeing results.
- Use model probabilities available strictly before kickoff; no post-match features.
- Calculate no-vig market probabilities, EV, flat-stake ROI/yield, closing-line value, hit rate, Brier/log loss versus market, and maximum drawdown.
- CLV convention must be explicitly documented (for decimal odds, compare entry price with the same bookmaker's final pre-match price on the selected outcome).
- Bootstrap by match date (cluster) for ROI/CLV uncertainty; report raw sample counts and calendar/competition cohorts.
- At least 100 settled qualifying bets is a minimum exploratory gate, not proof of a durable edge. Smaller samples are descriptive only.
- Do not publish a profitability conclusion when there are zero matched bets or inadequate sample size.

## Current blocker

The public free sample is documented as 20 selected matches and is explicitly for exploration/evaluation. The full archive is advertised as 1,474 national-team matches but does not claim complete friendlies/qualifiers coverage. The paid archive has not been acquired or licensed for this project. Therefore no production international ROI, CLV or drawdown result can be claimed yet.

**NO VALUE = NO BET.**
