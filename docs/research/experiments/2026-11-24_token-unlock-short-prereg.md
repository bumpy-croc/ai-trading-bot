# Shorting coins around large scheduled token unlocks: preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. Written and pushed before the simulator has been run on any real price. The event table (`results/unlock_events.csv`) was built first; it contains dates and sizes only. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-23, project thread: "carry on finding a profitable strategy"; and 2026-11-24 10:44 UTC, the decision "Yes, stricter bar, paper first" (each new idea may be tested once on 2025-26, written down first; a pass must paper trade before any money).
**Script**: `experiments/strategy_search/unlocks.py` (`events`, `design`, `exam`).

## 0. The idea, and why it is different from what has failed

Every trading rule tested so far in this programme used past prices, volume or funding to guess the next move, and none cleared its costs out of sample. This one uses information that is not in the price history: the dates on which large blocks of locked tokens become sellable by a project's team and early investors. The dates are published years ahead in each project's vesting schedule. The holders receiving the tokens usually paid little or nothing for them, so some sell, and anyone who knows the date can sell first. Industry studies of unlocks (for example Keyrock's 2024 review of about 16,000 events) report prices drifting down from about a month before team and investor unlocks and settling about two weeks after.

If the effect is real and larger than the cost of shorting, it is a trade with a cause behind it. If everyone already trades it, the short will be crowded, funding will run against it, and nothing will be left. That is what this measures.

## 1. Hypotheses

**H1**: a coin with a large team or investor unlock coming underperforms similar coins over a window around the unlock by more than the cost of holding the short, in 2022-24 and again in 2025-26.

**H0**: the underperformance is smaller than fees, slippage and the funding paid by the short, or it has been competed away, or it does not exist once the coin is compared with similar coins instead of with BTC.

## 2. Data

- **Unlock schedules**: DefiLlama's emission files for 375 tokens, downloaded 2026-11-24 (`https://defillama-datasets.llama.fi/emissions/<slug>`). 202 map to a Binance USDT perpetual in the Phase 1b cache.
- **An event** is a cliff unlock to the categories DefiLlama labels `insiders` (team, advisers) or `privateSale` (investors), on a single day, worth between 1% and 100% of the supply already unlocked the day before, for a coin whose perpetual had traded for at least 45 days. Larger ratios are launch-day artefacts and are dropped.
- **Count**: 877 events on 76 coins from 2021 to October 2026: 14 (2021), 37 (2022), 88 (2023), 175 (2024), 251 (2025), 312 (2026). The median event is 2.5% to 3% of unlocked supply. Many coins unlock monthly, so events cluster by coin (ENS, CELO, APE, OP, SEI and others have 15 to 35 each).
- **Prices, funding, liquidity ranks**: the Phase 1b cache (918 Binance USDT perpetuals, delisted included, to 2026-10-31).

## 3. The trade (fixed)

- **Position**: short the coin's perpetual, long the same dollar amount of a hedge basket. The basket is the equal-weight average of the perpetuals ranked 31 to 100 by liquidity the day before (point-in-time), leaving out any coin that itself has a position scheduled. This compares the coin with coins of its own kind. A hedge in BTC would instead be a bet that small coins fall against BTC, which they did by a wide margin in 2025-26 for reasons that have nothing to do with unlocks.
- **Windows** (three, chosen from the published studies and not from this data): U1 enter 14 days before the unlock and exit on the unlock day; U2 enter 7 days before and exit 7 days after; U3 enter 30 days before and exit 14 days after. Entry and exit at the daily close (UTC). When a coin's next event would start inside an open position, the position is extended to the later exit; one position per coin.
- **Size**: 10% of capital per position on each leg. When more than ten positions are open, each is scaled to an equal share so that the short side never exceeds 100% of capital.
- **Stop**: a position is closed inside the day once it has lost 30% since entry, judged on the coin's daily high. The exit is at the stop level, or at the open if the coin gapped past it, less a further 2%. No re-entry until that position's scheduled exit has passed.
- **Costs**: 5 bp fee and 20 bp slippage per side on each leg (1.0% per round trip of a pair), and the actual funding on both legs every day. The short receives or pays the coin's funding; the basket pays or receives the basket's average.
- **Not modelled**: the cost of re-sizing positions when the count changes, and the cost of holding a 70-coin basket in practice. If the idea passes, the paper stage uses a hedge that a small account can hold.

## 4. Stage A: design on data up to 2024-12-31

Prices and events are cut at 2024-12-31 before the simulator sees them. Fit: 2022-01-01 to 2023-12-31 (125 events). Validation: 2024 (175 events).

A window is shortlisted if all of these hold: portfolio Sharpe (in excess of 4% a year, after costs) above 0.7 in both periods; drawdown under 25% in both; mean net return per position positive in both. If more than one qualifies, the one with the higher validation Sharpe is taken. **If none qualifies, the idea stops there with H0 and the 2025-26 window is not touched.**

Reported for each window: positions, mean and median net return per position, share of winners, worst position, stops, and the same portfolio with a BTC hedge.

## 5. Stage B: one look at 2025-01-01 to 2026-10-31

Only the shortlisted window is run, once. This is candidate fourteen on that window, so the bar is the one-sided 5% level shared across fourteen: a t-statistic of the mean daily excess return of at least 2.69.

**PASS**, all of:

1. t-statistic at least 2.69.
2. Return at least 10% a year.
3. Maximum drawdown at most 20%.
4. 2025 and January-October 2026 both positive.
5. Mean net return per position positive.

Otherwise FAIL. Reported beside it and never decisive: the BTC-hedged and unhedged versions, the t-statistic across entry months, and the trade list.

## 6. Decision per outcome

- **No shortlist at Stage A**: H0, no exam, no build.
- **Stage B PASS**: propose to Alex a forward paper record (upcoming unlocks are published in advance, so each position can be written down before it opens), with a hedge a small account can hold. No money, and nothing deployed.
- **Stage B FAIL**: H0. No further windows, thresholds or hedges on this data.

The bot stays paused in every case.

## 7. Risks of a false result

- **Schedules as they look today.** The files are DefiLlama's current record. A date that was moved after the fact appears where it ended up, not where the market expected it. Most vesting schedules are fixed in contracts, but not all. A forward record is the only clean test, which is why a pass leads there.
- **Which tokens are covered.** DefiLlama documents tokens it chose to, mostly larger ones. Tokens that died are under-represented; for a short this is more likely to hide gains than to invent them.
- **Clustering.** A handful of monthly unlockers supply a large share of the events, especially in 2022-23. The portfolio statistics are on daily returns, which accounts for this; the per-position averages do not.
- **Small-coin drift.** Coins with heavy unlock schedules are newer and smaller than average. The basket hedge (ranks 31 to 100) removes most of this, not all.
- **Squeezes.** Phase 1b's lesson: any strategy that shorts single coins must cap the loss per position. The stop is in the simulation, with gaps charged at the open.
- **Stage A is thin**: 300 events on 32 coins. A shortlist from it is weak evidence, which is why the exam bar is high.
- **Determinism**: each stage is run twice; outputs must match.
