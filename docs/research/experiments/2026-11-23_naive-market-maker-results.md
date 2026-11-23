# The "51 lines of Python" market maker on BTC: results

**Date**: 2026-11-23
**Preregistration**: `2026-11-23_naive-market-maker-prereg.md` (commit `e439d75`, pushed before the simulator had seen any market data).
**Scripts**: `experiments/strategy_search/mm_data.py`, `mm_sim.py`. Output: `results/mm_naive.json`, `results/mm_naive_primary_daily.csv`.
**The run**: Binance BTCUSDT perpetual, every aggregated trade from 2025-11-01 to 2026-10-31 (580 million trades), plus 2026-11-01 to 2026-11-21 (37 million) as a separate fresh slice. 10,000 USDT account, 1,000 USDT orders, position limit ±5,000 USDT, maker fee 0.02% per side, orders arrive 50 ms after the price they are based on. Two full runs gave byte-identical output (sha256 `b3229e96ea112b00…`).

## Verdict

**H0 is supported. The primary cell fails and 1 of 25 cells passes (5 were needed).** On a real market this rule loses money, and it loses it faster the closer the orders sit to the price.

The primary cell (orders 5 basis points each side, refreshed every second) lost **198% of the account** in the year: 84,327 fills, 84 million dollars traded, a loss of 2,940 before fees and 16,865 in fees. The account reached zero on 20 February 2026.

## Why it loses

The post's idea is that buying a little below the price and selling a little above earns the gap. That holds only if the fills are random. On BTC they are not: an order resting near the price is filled when the market is moving through it.

- **Before any fee, every cell with orders within 2 bp of the price lost between 0.76 and 1.00 bp per dollar traded.** The spread the bot thought it was collecting never arrived.
- Measured directly on one day (2026-11-20, 5 cells, about 100,000 fills): one second after a fill the price was on average 0.7 to 0.8 bp on the wrong side of it in the four cells with enough fills to measure, and it had not come back at 10, 60 or 300 seconds. The trade that fills the order moves the price for good.
- The fee is then 2 bp per side on top. A bot quoting 1 bp away and refreshing every second traded 2.9 billion dollars in the year on a 10,000 account, lost 245,000 before fees and 572,000 in fees, and was at zero in four days.
- In the contest the post describes, the exchange was a simulation with no fees and counterparties that traded at random around a fair value. That is the difference.

## The grid (conservative fills, 0.02% fee)

Distance is on each side of the last trade. "bp per dollar" is profit before fees per dollar traded. The account is 10,000 and is not topped up, so losses beyond 100% mean it was wiped out on the date shown.

| Cell | Fills | Traded ($m) | Before fees, bp per dollar | Before fees ($) | Fees ($) | Net ($) | Net return | Max drawdown | Nov-Apr / May-Oct ($) | Account at zero | Fresh slice, Nov 2026 ($) | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 bp, 100 ms | 8,866,880 | 8,866.9 | -0.96 | -850,530 | 1,773,376 | -2,623,906 | -26,239% | wiped out | -1,611,154 / -1,012,753 | 2025-11-03 | -205,470 | fail |
| 0.5 bp, 100 ms | 6,693,139 | 6,693.1 | -0.88 | -589,420 | 1,338,628 | -1,928,048 | -19,281% | wiped out | -1,182,696 / -745,352 | 2025-11-03 | -159,770 | fail |
| 1 bp, 100 ms | 2,668,677 | 2,668.7 | -0.84 | -223,954 | 533,735 | -757,689 | -7,577% | wiped out | -475,187 / -282,502 | 2025-11-05 | -68,009 | fail |
| 2 bp, 100 ms | 488,902 | 488.9 | -0.76 | -37,058 | 97,780 | -134,838 | -1,348% | wiped out | -88,536 / -46,302 | 2025-11-28 | -11,835 | fail |
| 5 bp, 100 ms | 41,275 | 41.3 | +0.84 | +3,486 | 8,255 | -4,769 | -47.7% | 49.0% | -3,052 / -1,717 | | -1,374 | fail |
| 10 bp, 100 ms | 6,158 | 6.2 | +5.99 | +3,686 | 1,232 | +2,455 | +24.5% | 7.7% | +1,181 / +1,274 | | -902 | pass |
| 25 bp, 100 ms | 283 | 0.3 | +10.30 | +292 | 57 | +235 | +2.3% | 15.0% | +455 / -220 | | -777 | fail |
| 0.25 bp, 1 s | 6,182,442 | 6,182.4 | -0.98 | -604,963 | 1,236,488 | -1,841,451 | -18,415% | wiped out | -1,091,069 / -750,382 | 2025-11-03 | -138,122 | fail |
| 0.5 bp, 1 s | 5,114,839 | 5,114.8 | -0.90 | -457,966 | 1,022,968 | -1,480,934 | -14,809% | wiped out | -883,883 / -597,051 | 2025-11-04 | -115,789 | fail |
| 1 bp, 1 s | 2,860,907 | 2,860.9 | -0.86 | -245,420 | 572,181 | -817,601 | -8,176% | wiped out | -505,438 / -312,164 | 2025-11-04 | -67,975 | fail |
| 2 bp, 1 s | 893,045 | 893.0 | -0.86 | -76,802 | 178,609 | -255,411 | -2,554% | wiped out | -168,729 / -86,682 | 2025-11-14 | -21,992 | fail |
| **5 bp, 1 s (primary)** | 84,327 | 84.3 | -0.35 | -2,940 | 16,865 | **-19,805** | **-198%** | wiped out | -13,300 / -6,505 | 2026-02-20 | -2,369 | **fail** |
| 10 bp, 1 s | 9,778 | 9.8 | +1.49 | +1,454 | 1,956 | -502 | -5.0% | 14.4% | -479 / -23 | | -932 | fail |
| 25 bp, 1 s | 458 | 0.5 | +12.22 | +560 | 92 | +468 | +4.7% | 15.7% | -288 / +756 | | -337 | fail |
| 0.25 bp, 15 s | 1,924,703 | 1,924.7 | -1.00 | -193,354 | 384,941 | -578,294 | -5,783% | wiped out | -314,643 / -263,652 | 2025-11-06 | -37,880 | fail |
| 0.5 bp, 15 s | 1,766,120 | 1,766.1 | -0.94 | -165,415 | 353,224 | -518,639 | -5,186% | wiped out | -285,454 / -233,185 | 2025-11-07 | -34,652 | fail |
| 1 bp, 15 s | 1,390,039 | 1,390.0 | -0.90 | -124,628 | 278,008 | -402,636 | -4,026% | wiped out | -227,382 / -175,254 | 2025-11-08 | -28,396 | fail |
| 2 bp, 15 s | 845,081 | 845.1 | -0.85 | -71,797 | 169,016 | -240,813 | -2,408% | wiped out | -143,743 / -97,070 | 2025-11-13 | -18,399 | fail |
| 5 bp, 15 s | 229,606 | 229.6 | -0.75 | -17,151 | 45,921 | -63,073 | -631% | wiped out | -42,456 / -20,617 | 2025-12-10 | -5,058 | fail |
| 10 bp, 15 s | 46,030 | 46.0 | -0.53 | -2,426 | 9,206 | -11,632 | -116% | wiped out | -7,046 / -4,586 | 2026-08-01 | -1,240 | fail |
| 25 bp, 15 s | 2,526 | 2.5 | +1.49 | +377 | 505 | -128 | -1.3% | 12.5% | +419 / -546 | | -497 | fail |
| 0.25 x volatility, 1 s | 1,397,365 | 1,397.4 | -0.81 | -113,387 | 279,473 | -392,860 | -3,929% | wiped out | -202,626 / -190,233 | 2025-11-10 | -24,773 | fail |
| 0.5 x volatility, 1 s | 285,244 | 285.2 | -0.70 | -19,923 | 57,049 | -76,972 | -770% | wiped out | -36,138 / -40,834 | 2025-12-30 | -4,066 | fail |
| 1 x volatility, 1 s | 43,281 | 43.3 | +0.27 | +1,159 | 8,656 | -7,497 | -75.0% | 76.0% | -2,675 / -4,822 | | -1,352 | fail |
| 2 x volatility, 1 s | 6,295 | 6.3 | -0.12 | -75 | 1,259 | -1,334 | -13.3% | 20.2% | -227 / -1,107 | | -74 | fail |

- **Optimistic fills** (first in the queue instead of last) change little, because the orders sit many ticks from the market: the same single cell passes and no cell changes sign (`mm_naive.json` has every cell under both rules).
- **Latency, primary cell**: at 5 ms the loss is 184% of the account, at 250 ms it is 268%. Being faster helps a little and does not rescue it.
- **Widening the orders when the market is volatile**, the post's own suggested fix, loses in all four versions.
- **Fresh slice (1 to 21 November 2026)**: all 25 cells lost money. BTC fell from about 78,000 to 60,000 in those three weeks (low 55,000) and every version was carrying a long position into it.

## The one cell that passed

Orders 10 bp away, refreshed ten times a second: +24.5% for the year, 6,158 fills, both half-years positive, drawdown 7.7%, Sharpe 1.54. It then lost 9.0% in the three fresh weeks of November.

This is one cell in 25, which the preregistration says is what chance produces, and its neighbours do not support it (5 bp at the same speed lost 48%, 10 bp refreshed every second lost 5%). What it does is different from the post's idea: its orders are filled only when the price jumps 10 bp in about a tenth of a second, so it is buying into sudden spikes and selling the rebound. That depends on replacing orders ten times a second with 50 ms of delay in exactly the moments when exchanges slow down, and on about 17 fills a day. It is recorded as an observation. It is not evidence the rule works.

## What this says about market making for this project

- Earning the spread on a liquid market is a speed and fee-tier business. Firms that do it pay no maker fee or are paid a rebate, and react in microseconds. At the standard fee the break-even is 2 bp per dollar traded, and the naive rule starts about 1 bp below zero.
- The earlier finding stands: the practical way for a small account to be on the market-making side is a pooled vault run by people with that speed (Hyperliquid's HLP vault earned +17% in 2025 and +8% in 2026 to October), with its own risks, not a quoting bot of our own.
- Smaller, less liquid coins have wider spreads, and also much larger jumps. That was not tested here. It would be a separate preregistered test.

## Checks

- **Independent reimplementation.** A slow second implementation, written slot by slot with array searches instead of the event loop, was run on one day (2026-11-20, 3.3 million trades) for five cells. Fills, cash and position matched the simulator exactly in all five.
- **Hand-checked tapes.** `mm_sim.py selftest`: fill timing, one fill per refresh, both fill rules, the position limit and the post-only rejection.
- **Determinism.** Two runs, identical output files.
- **Environment.** Python 3.11, numpy 1.26.2, pandas 2.1.4, numba 0.60.0, pyarrow 17.0.0.

## Limits

- Trades only, no order book. The two fill rules bracket the queue position.
- One market and one year. BTC fell from about 110,000 to 78,000 over the window; the results per dollar traded do not depend on that, the inventory part of each cell's profit does.
- Funding on the position is ignored (average position about 2,500 USDT either way).
- Losses beyond 100% are arithmetic: the simulated bot keeps trading a fixed size after the account is gone, to show the rate of loss.

## Decision

Per the preregistration: **H0 supported, no build.** The bot stays paused. Nothing in `src/` changed.
