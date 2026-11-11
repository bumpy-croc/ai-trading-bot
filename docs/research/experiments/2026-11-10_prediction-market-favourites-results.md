# Prediction-market favourites: results

**Date**: 2026-11-10
**Status**: COMPLETE. All 3 candidates FAIL. H1 (favourites are priced too low) is falsified for this rule set; in this sample favourites were priced too high.
**Prereg**: [2026-11-10_prediction-market-favourites-prereg.md](2026-11-10_prediction-market-favourites-prereg.md), committed as `09b285b` before any price history was downloaded. Rules, costs and gates unchanged.
**Script**: `experiments/polymarket_favourites.py`. **Raw output**: `experiments/markets_results/polymarket_favourites.json`.
**Data**: 115,090 closed Polymarket markets listed; 14,630 in the preregistered universe (two outcomes, no fee, final volume >= $250,000, lifetime >= 7 days, end date 2022-01 to 2026-09); 14,085 of those returned a usable daily price history. By end year: 2022 41, 2023 38, 2024 1,479, 2025 8,192, 2026 4,880. Cached under `experiments/markets_data/polymarket/` (git-ignored).
**Determinism**: analysis run twice on the cached download; output byte-identical.
**Candidates tested in this thread so far**: 18, all FAIL.

## Verdict

| Candidate | Bets | Events | Avg fill price | Win rate | Excess return per bet | t by event | t by week | Return on capital above T-bills | Worst month | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| PM1, favourite at 0.90 to 0.97 | 5,727 | 2,377 | 0.922 | 90.9% | -3.4% | -7.37 | -3.98 | -17.5%/yr | -15.3% (2024-08, 67 bets) | FAIL |
| PM2, favourite above 0.97 to 0.99 | 3,623 | 1,647 | 0.968 | 96.7% | -1.9% | -4.94 | -3.09 | -11.6%/yr | -5.2% (2026-08, 30 bets) | FAIL |
| PM3, favourite at 0.70 to below 0.90 | 7,031 | 3,974 | 0.783 | 75.4% | -5.4% | -5.90 | -2.12 | -34.4%/yr | -24.4% (2024-08, 66 bets) | FAIL |

Costs: 1 cent per share on entry, T-bill accrual over the holding period (median 24 to 30 days). Gates: PM1 and PM3 fail all five. PM2 passes only G3 (no month worse than -10%).

The favourite won **less** often than its price implied in every band, before any cost. A favourite bought at 92 cents won 90.9% of the time. Costs then make it worse.

Cost sensitivity (mean excess return per bet): PM1 -2.8% at 0.5 cents, -4.6% at 2 cents. PM2 -1.3% and -2.7%. PM3 -4.8% and -6.7%. No candidate is positive even at half a cent.

By fill year (mean excess per bet at 1 cent): PM1 2024 -2.3%, 2025 -3.4%, 2026 -4.5%. PM2 -1.4%, -2.3%, -1.6%. PM3 -3.7%, -6.0%, -5.3%. Negative in every year with data.

## Calibration (descriptive, no costs)

First-touch bets by 5-cent band of the favourite's price, filled at the next daily point.

| Favourite's price band | Bets | Avg fill price | Win rate | Win rate minus price |
|---|---|---|---|---|
| 0.50-0.55 | 5,620 | 0.511 | 50.0% | -1.1 pts |
| 0.55-0.60 | 4,835 | 0.570 | 55.2% | -1.8 pts |
| 0.60-0.65 | 4,397 | 0.617 | 59.7% | -2.0 pts |
| 0.65-0.70 | 4,153 | 0.666 | 63.9% | -2.8 pts |
| 0.70-0.75 | 4,095 | 0.715 | 69.5% | -2.0 pts |
| 0.75-0.80 | 4,051 | 0.767 | 74.7% | -2.0 pts |
| 0.80-0.85 | 4,159 | 0.815 | 79.1% | -2.4 pts |
| 0.85-0.90 | 4,604 | 0.865 | 84.6% | -1.9 pts |
| 0.90-0.95 | 5,117 | 0.915 | 90.3% | -1.2 pts |
| 0.95-1.00 | 6,041 | 0.963 | 96.3% | 0.0 pts |

## Post-hoc reading (not preregistered, cannot promote anything)

The mirror image of this table is that the **underdog** side won about 2 points more often than its price implied. On a 14-cent contract that is a large percentage. This must not be read as "long shots are underpriced, buy them", for a reason the prereg already named:

- **The universe is chosen with hindsight.** Markets were selected on final volume of at least $250,000. An upset draws a rush of trading, so markets where the favourite lost are more likely to clear the volume bar than markets that resolved as expected. That alone produces this pattern.
- **The data agree with that explanation.** Across all favourite bets (13,379, band 0.50 to 0.99) the shortfall is -2.2 points (t by event -4.9). Split by final volume it is -1.5 points for $250k to $1M, -2.9 for $1M to $5M and -5.3 for $5M and above. The more a market's final volume was inflated, the worse favourites look.
- The shortfall is also absent in bets held longer than 90 days (-0.4 points, not significant), where late volume surges matter less.

The API reports only final volume, so this cannot be untangled from history. The only clean test is forward: record live order books for markets that qualify on volume at the time, and score them at resolution.

## Decision implied by the prereg

FAIL: buying favourites on Polymarket and holding to resolution does not beat cash. No retuned bands against this data without a new preregistration. Nothing is traded and no account is opened.

Whether to build the forward recorder (a small scheduled job, no money, no account, about 8 to 12 weeks to a clean answer on the underdog side) is a Board question, not something these gates answer.

## Caveats

- History prices may be mids or last trades rather than the price a buyer pays. That can only make a buyer's real result worse than shown.
- Fee-charging markets (about half of all listed, mostly sports and short crypto markets) are excluded.
- 99% of the universe ends in 2024 to 2026. Polymarket barely existed at this volume before 2024.
- 30 of 13,379 favourite bets resolved 50/50.
- Platform, stablecoin, tax and legal-access risks are not in the numbers.
