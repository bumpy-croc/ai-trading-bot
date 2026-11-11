# Prediction-market favourites: preregistration

**Date**: 2026-11-10
**Status**: PREREGISTERED. Rules, costs and gates below are locked before any price history is downloaded or any outcome is tabulated. Corrections go in new dated sections, never edits.
**Why**: the other-markets bake-off (`2026-11-10_other-markets-bakeoff-results.md`) found no edge in forex or ETFs, in line with two earlier crypto studies. Large liquid markets are where professional money competes. A small account's one structural advantage is that it fits into markets too small for that money. Prediction markets are the most accessible example with free, complete history.
**Script**: `experiments/polymarket_favourites.py` (did not exist when this file was written).
**What has been looked at so far**: market metadata only (counts by end year, lifetime, volume and fee flag), to size the download. No price history, and no tabulation of outcomes.

## Hypothesis

**H1.** On Polymarket, contracts on likely outcomes are priced too low (the favourite-longshot bias documented in betting markets and on Kalshi), by enough that buying favourites and holding to resolution beats cash after the spread.
Mechanism: buyers overpay for long shots, and capital tied up until resolution demands a return.
**H0.** Prices are calibrated to within the spread. Buying favourites earns nothing after costs, with occasional large losses.
H1 is supported if any candidate below passes. It is falsified for this rule set if none does.

## Data

- Polymarket Gamma API (`/markets/keyset`): every closed market with final volume >= $100,000 (115,090 listed on 2026-11-10).
- **Universe**: two-outcome markets, no trading fee (`feesEnabled` false), final volume >= $250,000, scheduled lifetime (start to end date) >= 7 days, end date from 2022-01-01 to 2026-09-30, closed with a final payout. About 14,700 markets by the metadata count. Fee-charging markets (about half of the listing, mostly sports and short crypto markets) are excluded because their fee schedules are not in the data; a fee can only make a taker's result worse.
- Polymarket CLOB API (`/prices-history`, daily points, full life) for each market's first outcome token. The other side's price is taken as 1 minus it.
- T-bill yield: `^IRX` as in the bake-off.

## Rule (one bet per market per candidate)

- Each day, the favourite is the side priced at or above 0.50, and f is its price.
- **Signal**: the first daily point at which f is inside the candidate's band.
- **Fill**: the same side at the next daily point, at that point's price plus the cost. No bet if there is no next point before the market closes, or if the fill price plus cost is 0.995 or more.
- **Hold to resolution.** Payout is 1, 0, or whatever the market finally paid (some resolve 50/50).
- **Candidates (gated)**: PM1 band 0.90 to 0.97; PM2 band above 0.97 to 0.99; PM3 band 0.70 to below 0.90.

## Costs

- Entry: 1 cent per share on top of the history price (half spread plus slippage). Sensitivity at 0.5 and 2 cents is reported; 2 cents is part of the gate.
- No exit cost (redemption at resolution).
- Opportunity cost: T-bill accrual on the stake from fill to close, plus one day for redemption.
- Not modelled: gas and bridging, the risk of holding USDC and of the platform itself, tax, disputed resolutions beyond what the final payout records.

## Metrics

- **Primary**: mean excess return per bet, r = payout / (fill + cost) - 1 - T-bill accrual, with two t-statistics: across event-level means (markets in one event move together) and across weekly means by fill week (events resolving together move together). The gate uses the smaller.
- **Return on capital**: sum of profits / sum of (stake x days held) x 365, minus the T-bill rate. This is what a fully recycled bankroll would have annualised.
- **Secondary**: win rate against average fill price, results by year, worst calendar month, bet count, median days held, and a calibration table (5-cent bands from 0.50 to 1.00).

## Pass gate (per candidate, all bets filled 2022-01-01 to 2026-09-30)

1. **G1 worth having**: mean excess return per bet > 0 and return on capital at least 5%/yr above T-bills.
2. **G2 not luck**: the smaller t-statistic is at least 2.77. That is the one-sided 5% level Bonferroni-adjusted for 18 candidates: the 15 in the other-markets bake-off plus these 3.
3. **G3 survivable**: no calendar month with 20 or more bets resolving loses more than 10% of the stakes resolving that month.
4. **G4 consistent**: mean excess return per bet is positive in each of 2024, 2025 and 2026 (each year with at least 200 bets).
5. **G5 cost-robust**: mean excess return per bet stays positive at 2 cents.

**PASS** = all five. **PROVISIONAL** = G1, G3, G4, G5 with the smaller t-statistic between 1.65 and 2.77. **FAIL** = anything else.

## Decision each outcome triggers

- **PASS or PROVISIONAL**: this is still a backtest on history prices that may be mids or last trades, not the price a buyer pays. The next step is four weeks of recording live order books (no money, no account) to measure the real cost of buying favourites, and a check with the Board on whether trading there is legal where they live. Nothing is traded on the strength of this file.
- **FAIL**: report it. No retuned bands against this data without a new preregistration.

## Risks of a false positive

- **Price source.** The history may be mid prices; real asks are higher. The 1 and 2 cent costs stand in for that.
- **Universe look-ahead.** The $250,000 filter uses final volume, which is not known at the time of the bet. Popular markets are selected with hindsight.
- **Stale prices near resolution.** A history price of 0.95 on the last day may not have been buyable. The next-point fill and the no-next-point rule limit this.
- **Clustering.** Thousands of bets, far fewer independent outcomes. Both t-statistics address it.
- **Tail.** A favourites strategy earns a little often and loses a lot rarely. Four years may not contain the worst month.
- **Determinism**: the analysis runs twice on the cached download; identical output required.
