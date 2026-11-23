# NostalgiaForInfinity, frozen at the end of 2024, on 2025-26: results

**Date**: 2026-11-11
**Preregistration**: `2026-11-11_nfi-frozen-oos-prereg.md` (commit `51088db`, before the strategy had been run on any data; Amendment 1, written before the exam run, fixed the run at the top 40 pairs).
**Scripts**: `experiments/strategy_search/nfi_data.py`, `nfi_run.py`. Output: `results/nfi_exam.json`, `results/nfi_exam_trades.csv`, `results/nfi_exam_equity.csv`.
**The run**: `NostalgiaForInfinityX5` v15.1.312 (commit `e5b345a`, 30 December 2024), unmodified, on freqtrade 2024.11, Binance spot, 5-minute candles, top 40 pairs by December 2024 volume, 10,000 USDT, up to 6 open trades, fee 0.10% per side, no slippage. 1 January 2025 to 31 October 2026. Run once.

## Verdict

**H0 is supported. The strategy fails gate 2 (drawdown) and gate 4 (risk-adjusted return). No paper trial.**

It ended the 22 months up 33.7%, which is better than every benchmark. It got there by sitting fully invested in three falling coins for eleven months, 54% below its peak and 39% below its starting capital at the low, with no stop, and was rescued by a sharp rally in September 2026. That is the pattern H0 described: many small wins, and rare large losses that are held instead of taken.

## Gates

| Gate | Requirement | Result | |
|---|---|---|---|
| 1. Return | at least 10% a year net | **17.2% a year** (+33.7% in total) | pass |
| 2. Drawdown | at most 30%, marked to market daily | **53.6%** (23 Sep 2025 to 19 Aug 2026) | **FAIL** |
| 3. Both years | positive in 2025 and in Jan-Oct 2026 | +1.7% and +31.4% | pass |
| 4. Risk-adjusted | Sharpe at least 0.8, in excess of 4% a year | **0.50** | **FAIL** |
| 5. Sample | at least 100 closed trades | 168 | pass |
| 6. Cost stress | only if 1 to 5 pass | not run | |
| 7. Look-ahead check | only if 1 to 6 pass | not run | |

Scoring check: the daily equity rebuilt from the exported orders ends at 13,367.4012 USDT; freqtrade's own final balance is 13,367.4012.

## What happened

| Date | Equity (USDT) | What the account held |
|---|---|---|
| 1 Jan 2025 | 10,000 | cash |
| 1 Oct 2025 | 13,163 | cash. Nine months of small dip-buys, almost all closed at a profit within a day |
| 12 Oct 2025 | 11,433 | bought FET and ARB minutes before the 10 October flash crash; its own de-risk rule sold most of both at the bottom, losing about 2,400 in fifteen minutes inside positions that stayed open |
| 1 Jan 2026 | 10,170 | 114 in cash; the rest in ARB, FET and FIL (and a little AAVE), bought on the way down |
| 19 Aug 2026 | **6,108** | 115 in cash; ARB (down 82% from October 2025) and FIL (down 71%) |
| 1 Oct 2026 | 13,468 | ARB rose 2.7 times and FIL 1.65 times in six weeks; the strategy sold into it |
| 31 Oct 2026 | 13,367 | ARB and FIL positions still open, closed at market by the backtest at a loss of 2,186 |

- From December 2025 to August 2026 the strategy closed 21 trades for a total profit of about 45 USDT. It had no cash left to trade with. It kept adding to and trimming the stuck positions (252 orders in that period).
- The account was below its starting 10,000 on 282 of 670 days, and more than 30% below its peak on 207 days.
- 165 of 168 closed trades were winners. The three losers are the positions force-closed on the last day: ARB held 386 days, FIL held 357 days, and one opened two hours before the end. A third stuck position, FET, was held 359 days and closed for a profit of 38. The median winner was held about an hour; the average loser 248 days.
- Worst single days: −16.2% (2 Feb 2025), −15.8% (6 Feb 2026), −14.0% (11 Oct 2025).

**The result depends on where the window ends.** Measured to 19 August 2026 the same run shows −39%. Measured to 31 October it shows +34%. The preregistered window is the one that counts, and it happens to end six weeks after the rally.

## What freqtrade's own report says, against what happened

| | freqtrade's summary | Marked to market daily |
|---|---|---|
| Maximum drawdown | 14.2% | **53.6%** |
| Sharpe ratio | 0.74 | 0.50 |
| Win rate | 98.2% | |
| Profit factor | 2.52 | |

freqtrade measures drawdown on closed trades only. A loser that is held, or a loss taken inside a position that stays open (as on 10 October 2025), does not show until the position is finally closed. This is the same blind spot the repository's published backtests have, and it is why those backtests look so much safer than the strategy is. On its own tuning history (Amendment 1 of the preregistration) the same setup showed +30.7% in eight weeks, 67 wins out of 67, no drawdown.

## Benchmarks, same window

| | Total return | Per year | Maximum drawdown | Sharpe |
|---|---|---|---|---|
| **NostalgiaForInfinity (this run)** | **+33.7%** | 17.2% | 53.6% | 0.50 |
| Idle dollars at 4% | +7.5% | 4.0% | 0 | |
| BTC bought and held | −17.1% | −9.7% | 53.0% | −0.12 |
| The same 40 coins, equal weight, bought and held | −68.2% | −46.5% | 78.7% | −0.67 |

The strategy beat holding its own coins by a very wide margin, and beat BTC with the same depth of drawdown. That is a real point in its favour: buying dips and selling bounces did add value in a market where the coins themselves lost two thirds. It does not change the verdict, because the risk taken to get there (all capital in three coins with no exit) is the risk that ends accounts.

## Secondaries (reported, never decisive)

- **Capital deployed**: 60% of equity on average, 3.0 pairs held on average. About 98% from January to August 2026.
- **Profit by pair**: best NEIRO +530, USUAL +482, ACT +403, SUI +376, DOGE +371. Worst FIL −1,298, ARB −764. 33 of the 35 pairs it traded were positive.
- **Profit by month, closed trades (USDT)**: 2025: Jan 677, Feb 406, Mar 509, Apr 82, May 615, Jun 290, Jul 116, Aug 174, Sep 294, Oct 758, Nov 783, Dec 1. 2026: Jan 22, Feb 2, Mar 0, Apr 2, May 2, Jun 16, Jul 0, Aug 0, Sep 155, Oct −1,537 (the forced closes).
- **Trading volume**: 236,000 USDT over the window, 2,136 orders, average trade 11.8 days.
- **Worst trades**: FIL −1,375 (opened 9 Nov 2025), ARB −811 (opened 10 Oct 2025), RUNE −27 (opened two hours before the window ended). No other closed trade lost money.

## Limits of this test

- **40 pairs, not 80.** Memory limited the run to the lower end of the strategy's recommended range (Amendment 1). A wider list would have spread the stuck positions over more coins; it would not have removed them, because the strategy holds at most six positions whatever the list.
- **A fixed list.** A live pairlist refreshes toward whatever is traded most. That cuts both ways and cannot be tested without hindsight.
- **No slippage, limit fills at candle prices.** This favours the strategy in normal markets. Gate 6 would have tested it and was not reached. In the 10 October 2025 flash crash the backtest filled both the entries and the de-risk sells at 5-minute candle prices; real fills in those minutes could have been better or worse.
- **One window.** 22 months, one bull phase, one long altcoin decline, one rally.
- **Spot, long only.** The futures mode (3x leverage, shorts) was not run. With a 54% unlevered drawdown, 3x leverage would not have survived.

## Run record

- The freqtrade process ran from 18:45:33 to 20:50:03 UTC, peaked at about 34 GB (15 GB of memory plus a 26 GB swap file), and wrote its result files and summary normally.
- At 19:22 UTC the wrapper script that launched it (`nfi_run.py backtest`) was stopped, because the tool that started it has a two-hour limit and the run was on course to take slightly longer. Stopping the wrapper left the freqtrade process running untouched. The only effect is that `run_meta.json` (fee, pair count, run time) was written by hand, and says so.
- The local exchange-metadata stub lived in the wrapper. freqtrade reads market metadata once at start-up and makes no exchange calls during or after a backtest, and the log shows none.

## Decision

Per the preregistration (section 6, "Gate 1 or 2 fails"): **H0 supported, no trial.** This is recorded as the answer to "do the popular public strategies work": the most used one made money on unseen data in this window, by taking a risk its own statistics do not show and that would have had most users out of the market, 39% down, well before the recovery.

The second half of Alex's choice went ahead as preregistered: see `2026-11-11_slow-adaptive-trend-results.md`.

**Exam-window accounting.** The 2025-01 to 2026-10 window has now faced nine candidates. It is closed.
