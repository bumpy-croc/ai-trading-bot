# Polymarket underdogs, forward test: preregistration

**Date**: 2026-11-11
**Status**: PREREGISTERED. Rules, costs, gates and the scoring date are locked before the first snapshot is recorded. Corrections go in new dated sections, never edits.
**Authority**: Board, 2026-11-11 12:10 UTC, project thread: chose "Record Polymarket prices forward".
**Why**: the favourites backtest (`2026-11-10_prediction-market-favourites-results.md`) failed, and showed the mirror image: the underdog side won about 2 points more often than its price. That backtest chose markets by final volume, which over-represents upsets, so the pattern is most likely selection. History cannot separate the two because the API reports only final volume. A forward recording can: it sees volume and quotes as they were on the day.
**Recorder**: `tools/polymarket_recorder.py`, run by `.github/workflows/polymarket-recorder.yml` every 6 hours. Snapshots are committed to the `data/polymarket-recorder` branch.
**Scoring script**: `experiments/polymarket_forward_score.py`.
No money, no account, no orders. The recorder only reads public market listings.

## Hypothesis

**H1.** On Polymarket, the underdog side of a two-outcome market is priced too low, by enough to beat cash after paying the real ask and the real taker fee.
**H0.** The pattern in the backtest came from choosing markets by final volume. Bought at the ask on the day, underdogs earn nothing or lose.
H1 is supported if any candidate below passes. H0 is supported if none does with enough bets to tell.

## What is recorded

Every 6 hours: every open market with at least $100,000 traded so far, with its best bid, best ask, last trade, volume and liquidity so far, start and end dates, fee flag and fee rate, and its event. Nothing is filtered at recording time beyond the $100,000 floor, so the scoring rules below cannot be bent by what was or was not saved.

## Rule (one bet per market per candidate)

- **Eligible at a snapshot**: two outcomes; accepting orders; volume so far at least $250,000; end date at least 24 hours after the snapshot; a best bid and best ask with 0 < bid < ask < 1 and ask minus bid at most 0.05.
- **Underdog**: the side that is cheaper to buy. Buying Yes costs the best ask. Buying No costs 1 minus the best bid. Call the cheaper price u.
- **Entry**: the first snapshot at which the market is eligible and u is inside the candidate's band. Buy the underdog at u.
- **Fee**: where the market charges one, the published taker fee per share, rate x u x (1 - u), with the rate recorded in that snapshot.
- **Hold to resolution.** Payout is what the market finally paid that side (1, 0, or a split).
- **Candidates (gated)**: U1, u from 0.03 to 0.10. U2, u above 0.10 to 0.30. U3, u above 0.30 to below 0.50.
- **Reference, not gated**: the favourite side of the same markets at the same snapshots, bought at its own ask.

## Scoring

- **Bets counted**: entries from the first snapshot to 2027-01-10, in markets resolved by 2027-02-07. Entries in markets still open on 2027-02-07 are excluded and their number reported.
- **Scoring date**: 2027-02-08. One extension is allowed, to 2027-05-09, only under the INCONCLUSIVE rule below.
- **Primary metric**: mean excess return per bet, r = payout / (u + fee) - 1 - T-bill accrual from entry to close.
- **t-statistics**: across event-level means, and across means by entry date. The gate uses the smaller.
- **Secondary**: win rate against price paid, return on capital (profits / stake-days, annualised, minus T-bills), results by fee and no-fee markets, by entry month, and the favourite-side reference.

## Pass gate (per candidate)

1. **G1 worth having**: mean excess return per bet of at least +2% of stake.
2. **G2 not luck**: the smaller t-statistic is at least 2.82. That is the one-sided 5% level Bonferroni-adjusted for 21 candidates: the 18 already tested in this line of work plus these 3.
3. **G3 enough bets**: at least 300 resolved bets from at least 100 events.
4. **G4 consistent**: mean excess return per bet is positive in both halves of the entry period.

**PASS** = all four. **PROVISIONAL** = G1, G3, G4 with the smaller t-statistic between 1.65 and 2.82. **INCONCLUSIVE** = G3 fails, or G1 and G4 hold but the t-statistic is below 1.65 with fewer than 2,500 bets. **FAIL** = anything else.

Power, stated up front: an underdog bought at 14 cents that wins 16% of the time is a 14% edge on stake with a per-bet standard deviation near 260%. Detecting that at t = 2.82 takes roughly 2,700 independent bets. Twelve weeks may not deliver that many in one band. That is why INCONCLUSIVE exists and why it is not a pass.

## Decision each outcome triggers

- **PASS or PROVISIONAL**: report to the Board with the numbers. Any live trial needs the Board's decision on a venue account and on whether trading there is legal where they live. Nothing is traded on the strength of this file.
- **INCONCLUSIVE on 2027-02-08**: keep recording, score once more on 2027-05-10 with entries to 2027-04-11. No further extension.
- **FAIL, or INCONCLUSIVE at the extension**: disable the recorder workflow, write the result up, close the line of work.

## Risks

- **Size.** The listing gives the best ask, not how much is offered there. The test assumes stakes small enough to fill at the best ask (tens of dollars). `liquidity` is recorded for context.
- **Six-hour snapshots.** A market can open, qualify and resolve between snapshots; those are never seen. Entry is at a real quote at a real time, so what is seen is tradable.
- **Censoring.** Markets still open at the scoring date are dropped, which removes long-dated bets. The count is reported.
- **Missed runs.** Scheduled jobs can be skipped or delayed by the host. A missed snapshot delays entries; it does not bias them.
- **Rule changes at the venue** (fees, tick sizes) during the window are recorded in the snapshots as they happen.
- **Candidate count.** 21 with these three. The t gate reflects that.
