# Resting orders that only fill in sudden spikes, on ETH, SOL and XRP: preregistration

**Date**: 2026-11-23
**Status**: PREREGISTERED. Written and pushed before the simulator has been run on any of the test data below. Corrections go in new dated sections.
**Authority**: Alex, 2026-11-23, project thread: "merge the pr and carry on finding a profitable strategy".
**Scripts**: `experiments/strategy_search/mm_data.py` (tick data), `experiments/strategy_search/spike_sim.py` (simulator and report).

## 0. Where the idea comes from (exploration, disclosed)

The market-maker test of the same day (`2026-11-23_naive-market-maker-results.md`) lost money everywhere except where the orders sat far from the price and were refreshed fast. Those orders are only filled when the price jumps a long way in a fraction of a second. After that result I looked further at the same BTC data (2025-11 to 2026-11), which is exploration after the fact and proves nothing:

- 16 more cells (distance 10, 15, 25, 50 bp; order age 50 ms to 3 s) and the price 1 second to 30 minutes after each fill.
- Fills on far, fresh orders were followed by a move back: for 25 bp orders at most 350 ms old, the price was 16 bp better five minutes later on average over all fills.
- Most of that came from the later fills inside a run of spikes. Taking only the first fill after five quiet minutes, the move back was 3 to 6 bp (220 to 1,400 fills a year depending on the cell, t about 2 to 4 by day). A round trip costs about 8 bp (maker in, taker out, slippage). **So on BTC the effect is real before costs and does not pay after them.**

The hypothesis worth one clean test: on more volatile coins the same kind of spike is proportionally larger, and so is the move back, while the fees are the same. If so, the effect clears costs on ETH, SOL or XRP where it does not on BTC.

## 1. Hypotheses

**H1**: buying (selling) a sudden spike down (up) through a far resting order and closing five minutes later earns more than its costs on the three most liquid alt perps.

**H0**: the move back is too small to pay the fees there as well, or the spikes that fill the orders keep going.

## 2. Test data (none of it touched before this file was pushed)

- **Candidates**: the three most traded USDT perps after BTC by median daily volume from May to October 2025 (computed from the Phase 1b cache before the window): **ETHUSDT, SOLUSDT, XRPUSDT**. Every aggregated trade from 2025-11-01 to 2026-10-31.
- **Control**: BTCUSDT from 2024-11-01 to 2025-10-31, the year before the exploration window. It checks whether the before-cost effect seen in exploration exists in a different year. It is not expected to pass after costs and is not part of the gates.
- **Fresh slice, secondary**: 1 to 21 November 2026 for the three candidates.

## 3. Rule (one configuration)

- **Orders.** Every 100 ms the bot reads the last trade price. A bid at price × (1 − d) and an ask at price × (1 + d) based on it go live 250 ms later and rest until replaced (so a live order is 250 to 350 ms old). Rounded away from the price to the exchange tick.
- **Distance d.** 25 bp for BTC, scaled for each coin by the ratio of its daily volatility to BTC's over 2024-11-01 to 2025-10-31 (before the test window): ETH 1.724, SOL 1.991, XRP 2.200. So **ETH 43 bp, SOL 50 bp, XRP 55 bp**.
- **Entry.** An order is filled only when a trade prints strictly through its price (last in the queue). Fill at the order's price. An order that would already cross when it arrives is rejected.
- **One position at a time per coin.** After a fill, both orders are pulled. No new orders until the position is closed.
- **Exit.** 300 seconds after the fill, at the last trade price, with 1 bp of slippage against the position.
- **Size.** Each trade is the whole account at 1x (10,000 USDT per coin account, no compounding).
- **Costs.** Maker fee 0.02% in, taker fee 0.05% out, 1 bp slippage out: 8 bp per round trip. Funding ignored (five-minute holds).

## 4. Metric and gates (candidates only, all required)

**Primary metric**: mean net return per trade across the three candidates pooled, with a t-statistic computed on daily sums (so a cluster of trades in one day counts once).

1. Pooled mean net return per trade above zero with t at least 2.5. The bar is above 2 because about 40 exploratory cells came before this test.
2. Net positive in each of the three coins.
3. Average net return per coin account at least 10% a year.
4. In every coin account: worst single trade no worse than −5%, maximum drawdown at most 15%.
5. At least 100 trades in each coin.

## 5. Reported beside the gates

- The control (BTC, previous year): mean move back before costs, and net.
- Before-cost figures, by half-year, long and short sides separately.
- Sensitivities: order age 50 ms and 1 s; exit after 60 s and after 30 minutes; a maker exit (4 bp round trip, optimistic).
- The fresh November 2026 slice.

## 6. Decision per outcome

- **H1 (all five gates)**: propose to Alex a forward shadow test on live data (orders computed and logged in real time, no money), because a pass here still assumes fills that only a live test can confirm. No live trading.
- **H0**: no build. Recorded with the other results that depend on fee tier.
- Mixed (gates 1 and 2 pass, another fails): reported as mixed, no build without a new preregistration.

## 7. Known limits (stated before the run)

- Trades only, no order book. A resting order far from the price is filled when the market trades through it, which is what the entry rule requires, so the entry is realistic. The exit at the last trade price plus 1 bp may be optimistic in the minutes after a spike, when the gap between buyers and sellers is wide.
- The simulated account is never refused an order. Replacing two orders ten times a second is at the edge of the exchange's order-rate limit.
- One year per coin. The largest liquidation event of recent years (10 October 2025) is inside the control's year and outside the candidates' year.
