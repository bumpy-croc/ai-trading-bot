# The "51 lines of Python" market maker on BTC: preregistration

**Date**: 2026-11-23
**Status**: PREREGISTERED. Written and pushed before the simulator has been run on any data. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-23, project thread: "Can we backtest this strategy and tell me what it returns?", with a link to an X post (`x.com/foodloss_otoko/status/2124557763594637648`, 22 November 2026).
**Scripts**: `experiments/strategy_search/mm_data.py` (download and convert), `experiments/strategy_search/mm_sim.py` (simulator and report).

## 0. What the post describes

The post retells a story about a student team in a Jane Street trading contest who made "26 million dollars" with a 51-line market-making bot, and suggests rebuilding it on Hummingbot for BTC-USDT. The post itself says the figure comes from the contest's simulated exchange and is not a claim about real markets. There is no real-money record to check, so the only thing that can be tested is the rule.

The rule, as the post states it:

1. Fair value is the price of the most recent trade.
2. Keep a buy order a little below it and a sell order a little above it.
3. Whenever new trades arrive, cancel the old orders and place new ones around the new price.

The post's suggested additions for a real market: stop buying (or selling) when the position reaches a limit, and move the orders further away when the market is volatile.

The post gives no numbers for "a little", for the order size or for the limit. They are fixed here, as a grid, before anything is run. Every cell is reported.

## 1. Hypotheses

**H1**: quoting both sides around the last trade earns more than idle dollars after exchange fees, because the bot collects the distance between its buy and sell prices more often than the market moves against the inventory it picks up.

**H0**: on a real, competitive market the orders that get filled are mostly the ones that should not have been (the price is moving through them), so the spread collected is smaller than the loss on inventory plus the fees. Net of fees the bot loses at every distance, or earns less than idle dollars.

## 2. Market, data, window

- **Market**: Binance USDT-margined BTCUSDT perpetual. It is the most traded BTC market, the position can be long or short, and its maker fee (0.02% per side at the standard tier) is the lowest fee a small account can get on Binance. Spot at 0.10% per side would be strictly worse for this strategy.
- **Data**: every aggregated trade (time in milliseconds, price, size, which side was the aggressor) from the public archive, `data/futures/um/monthly/aggTrades/BTCUSDT/`. About a billion trades.
- **Window**: 2025-11-01 00:00 UTC to 2026-10-31 24:00 UTC (12 months). Reported in two halves (Nov-Apr, May-Oct).
- **Fresh slice, secondary**: 2026-11-01 to 2026-11-21 from the daily files, reported separately.
- **Window accounting.** The daily-bar exam window (2025-01 to 2026-10) was declared closed for new candidates on 2026-11-11. This test overlaps it. It is run anyway because Alex asked for this specific rule to be backtested, it uses tick data that no earlier test touched, nothing is being selected (the whole grid is reported), and a pass leads only to a forward paper test, never to a trade.

## 3. Simulator (fixed)

- **Quotes.** The bot looks at the last trade price on a clock every `T`. Its new orders reach the exchange `L` = 50 ms later and rest until the next pair replaces them. Bid = last price × (1 − d), rounded down to the 0.1 tick; ask = last price × (1 + d), rounded up.
- **Refresh clock `T`**: 100 ms (the closest a small account can get to "on every trade" within Binance's order-rate limits), 1 s, and 15 s (roughly Hummingbot's default refresh).
- **Distance `d`**: 0.25, 0.5, 1, 2, 5, 10 and 25 basis points on each side (fixed), and four volatility-scaled versions at `T` = 1 s: d = k × the standard deviation of one-minute returns over the previous 60 complete minutes, k = 0.25, 0.5, 1, 2, with a floor of 0.25 bp. That is 21 + 4 = 25 cells.
- **Fills, conservative (primary)**: a resting buy is filled only when a trade prints strictly below its price; a resting sell only when a trade prints strictly above. This assumes the bot is last in the queue at its price. Fill is at the order's own price. Each order fills at most once per refresh.
- **Fills, optimistic (reported beside it)**: also filled when an aggressor trade prints exactly at the order's price (first in the queue).
- **Orders that would cross.** If the market has already moved past a new order's price when it arrives (bid at or above the last trade, ask at or below), the order is treated as rejected (post-only) for that refresh. This favours the strategy: a plain limit order would be filled as a taker at a higher fee.
- **Size and limit.** Account 10,000 USDT. Each order 1,000 USDT. No new buy when the position would exceed +5,000 USDT, no new sell when it would go below −5,000 USDT. No compounding. No inventory skew (the post has none).
- **Costs.** Maker fee 0.02% per side (Binance USDT-M, standard tier, no discount). Results are also shown before fees. Funding on the perpetual is ignored; the position averages near zero.
- **Marking.** Equity = cash + position at the last trade price, recorded every hour. Drawdown and Sharpe are computed on hourly and daily marks.
- **Sensitivity (primary cell only)**: `L` = 5 ms and 250 ms.

## 4. Metrics and gates

**Primary metric**: net profit after the 0.02% maker fee, as a percentage of the 10,000 USDT account over the 12 months, conservative fills.

Also reported for every cell: profit before fees, profit per dollar traded in basis points (before and after fees), number of fills, dollars traded, maximum drawdown, daily Sharpe ratio in excess of 4% a year, each half-year, and the fee at which the cell would break even.

A cell **passes** if, with conservative fills and the 0.02% fee: net return above 4% (what idle dollars earn), maximum drawdown at most 20%, and both half-years positive.

**Primary cell**: d = 5 bp, T = 1 s. It is named in advance because 5 bp is the smallest distance on the grid at which a completed buy-and-sell pair clears the two fees with room to spare (10 bp collected against 4 bp paid).

## 5. Decision per outcome

- **H1 supported** if the primary cell passes, or if at least 5 of the 25 cells pass (a region, not a lucky cell). Then: propose a forward paper test of the best region (Hummingbot paper mode or our own recorder) to Alex. No money, no live trading.
- **H0 supported** if the primary cell fails and fewer than 5 cells pass. Then: report the grid and what the losses are made of. No build.
- With 25 cells, one or two passing by chance is expected and is not a result.
- Nothing in either outcome changes the bot, which stays paused.

## 6. Known limits (stated before the run)

- A trade-only simulation cannot see the order book. The two fill rules bracket the queue position; they are almost the same thing once the orders sit more than a few ticks from the market.
- The simulated bot does not move the market and is never front-run. Real fills are no better than this.
- One market, one year. BTC only.
- The determinism check is two full runs with identical output.
