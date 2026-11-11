# NostalgiaForInfinity, frozen at the end of 2024, on 2025-26: preregistration

**Date**: 2026-11-11
**Status**: PREREGISTERED. One strategy, one configuration, one run. Locked before the strategy has been run on any data. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-11, project thread: after the research report on public bot claims, chose "Both, freqtrade test first" (run the most popular public freqtrade strategy on data it never saw, then build a slow adaptive system as a paper trader).
**Scripts**: `experiments/strategy_search/nfi_data.py`, `experiments/strategy_search/nfi_run.py`.

## 0. Why this test

NostalgiaForInfinity (NFI) is the most used public freqtrade strategy (about 3,500 GitHub stars, 28,000 commits). Its repository publishes backtests in commit comments and no live record. Backtests by the authors are run on the same history the strategy was tuned on, so they say nothing about whether it has an edge.

A version of the code frozen before 2025 cannot have been tuned on 2025-26. Running that frozen version on 2025-26 is therefore a fair exam, and the only kind of evidence about a public strategy that is worth having.

## 1. Hypotheses

**H1**: the frozen strategy earns clearly more than idle dollars on unseen data after exchange fees, with a tolerable drawdown, because its dip-buying entries and staged exits capture a real, persistent rebound effect in liquid altcoins.

**H0**: its backtest record is the product of tuning on past data and of holding losing positions without a stop (many small wins, rare large losses). On unseen data it earns no more than idle dollars, or loses.

H1 is **supported** if every gate in section 5 passes. H0 is **supported** if gate 1 or gate 2 fails. Anything else is reported as mixed, with the failing gate named.

## 2. What is frozen

- **Strategy**: `NostalgiaForInfinityX5.py`, version string `v15.1.312`, at commit `e5b345a8d81e4559572cc0806d86d9178d0719e1` of `iterativv/NostalgiaForInfinity` (committed 2024-12-30 18:01 +02:00, the last commit of 2024). Not modified in any way.
- **Engine**: freqtrade `2024.11` (the stable release at that date) with its pinned requirements (numpy 1.26.4, pandas 2.2.3, TA-Lib 0.4.32, technical 1.4.4). One forced substitution: `pandas-ta==0.3.14b` no longer exists on PyPI, so freqtrade's own fork `ft-pandas-ta==0.3.15` (the same 0.3.14 code line) is used.
- **Configuration**: the repository's own files at that commit, as its own backtest script uses them: `configs/trading_mode-spot.json` and `configs/exampleconfig.json`. That means Binance spot, 5-minute candles, 6 open trades at most, unlimited stake, a 10,000 USDT wallet, `tradable_balance_ratio` 0.99, grinding and de-risking enabled, limit orders. The strategy's header at this commit recommends 4 to 6 open trades and a pairlist of 40 to 80 pairs.
- **Mode**: spot, long only. The futures mode (3x leverage, shorts) is not run. It would be a second candidate.

## 3. Pairs

The recommended live pairlist is the top 80 pairs by volume, refreshed every 30 minutes. Freqtrade cannot backtest a changing list, and the authors backtest on static lists. A static list has to be chosen with information from before the exam, so:

- Universe: every Binance spot pair quoted in USDT that traded in December 2024 (from the public archive, which keeps delisted pairs).
- Remove pairs matching the strategy's own `configs/blacklist-binance.json` at the frozen commit.
- Remove pairs with no November 2024 monthly file (listed less than 30 days, the recommended pairlist's age filter).
- Keep the **80 with the highest December 2024 quote volume**.

The list is in `experiments/strategy_search/results/nfi_pairs.json`, committed with this file. It is fixed for the whole exam. Pairs that are delisted or renamed during 2025-26 stay in; their data simply ends, and any open position is closed by freqtrade at the last available price.

Not reproduced: the recommended pairlist's price-tick and spread filters (they need live order books), and its refresh. A refreshed list would have rotated toward whatever was most traded each week. That cannot be backtested without choosing pairs with hindsight, so it is left out and named as a limitation.

**Memory fallback, fixed now.** The strategy computes hundreds of indicator columns per pair. If 80 pairs over the exam window do not fit in the container's 15 GB of memory, the run uses the top 60 of the same list, and failing that the top 40 (the lower end of the recommended range). The choice is made from a memory measurement on November-December 2024 data, before the exam run, and recorded in the results file. No other change is allowed.

## 4. Data and exam window

- Binance spot candles from the public archive: 5m, 15m, 1h, 4h and 1d for the 80 pairs and BTC/USDT, with enough earlier history for the strategy's 800 start-up candles on each timeframe.
- **Exam window: 2025-01-01 00:00 UTC to 2026-10-31 24:00 UTC** (22 months). Every candle in it is after the frozen commit.
- **Fee: 0.10% per side** (`--fee 0.001`), Binance's standard spot rate with no discount. No slippage is added in the primary run, which favours the strategy. Fills happen at 5-minute candle prices as freqtrade models them.

**Window accounting.** The 2025-01 to 2026-10 window has faced 5 daily crypto candidates (#1315, #1326) and 3 hourly event trades (#1327). This is the ninth candidate and the last one this window can carry (the repository's budget is about ten). It is defensible here for a reason the others lacked: nothing about the candidate was chosen by us. The code, the settings and the pair rule all pre-date the window.

## 5. Metric and gates (all required)

**Primary metric**: net return of the 10,000 USDT wallet over the exam window, with open positions valued at market. Freqtrade closes open trades at the last candle, so its final balance is already marked to market. Daily equity is rebuilt from the exported orders and daily closes, because freqtrade's own drawdown figure only sees closed trades and understates the drawdown of a strategy that holds losers.

1. **Return**: net annualised return at least 10% (idle dollars earned about 4%).
2. **Drawdown**: maximum drawdown of daily marked-to-market equity at most 30%.
3. **Both years**: positive net return in 2025 and in January-October 2026, each measured on marked-to-market equity.
4. **Risk-adjusted**: daily marked-to-market Sharpe ratio, in excess of 4% a year, at least 0.8.
5. **Sample**: at least 100 closed trades.
6. **Cost stress**: run only if gates 1 to 5 pass. A second run at 0.20% per side (fee plus 0.10% for spread and slippage) must still return more than 4% a year.
7. **No peeking**: run only if gates 1 to 6 pass. Freqtrade's `lookahead-analysis` on the frozen strategy must report no look-ahead bias in entries or exits. A strategy that reads future candles would pass an out-of-sample backtest and still fail live.

Secondaries, reported and never decisive: freqtrade's own summary (profit factor, win rate, average trade duration, closed-trade drawdown), profit by month, profit by pair, the ten worst trades, time in market, average capital deployed, and three benchmarks over the same window: idle USDT at 4%, BTC bought and held, and an equal-weight basket of the same 80 pairs bought on 2025-01-01 and held.

## 6. Decision each outcome triggers

- **All gates pass**: report to Alex with a proposal for an 8-week paper trial of the same frozen configuration. Nothing is deployed, and the bot stays paused, until Alex decides.
- **Gates 1 to 5 pass, 6 or 7 fails**: reported as "profitable in a frictionless backtest only". No trial.
- **Gate 1 or 2 fails**: H0 supported. No trial. Recorded as the answer to "do the popular public strategies work".
- **Either way** the second half of Alex's choice goes ahead: the slow adaptive many-coin system as a paper trader, with its own preregistration.
- **The run cannot be completed faithfully** (the frozen code does not run on the frozen engine, or does not fit in memory even at 40 pairs): reported as inconclusive, with what was tried. The strategy is not edited to make it run.

## 7. Risks of a false result

- **False pass from look-ahead.** Covered by gate 7.
- **False pass from frictionless fills.** Covered by gate 6. Limit orders at candle prices in thin pairs are optimistic.
- **False pass from one regime.** The window holds a bull phase and a deep altcoin drawdown, but it is still one 22-month draw. A pass leads to a paper trial, not to live trading.
- **False fail from the static list.** A live user's list would have refreshed. The fixed list keeps coins that later died, which a live list would have dropped only after they stopped being heavily traded. This cuts both ways and cannot be resolved without hindsight.
- **False fail from version skew.** The strategy is run on the engine it was written for. The one substituted package is named above.
- **Survivorship**: none in the pair choice. The list is built from December 2024 data and includes pairs that were later delisted.
- **Determinism**: the same run on December 2024 data is executed twice and must produce identical trades before the exam run is trusted. The exam itself is run once.

## Amendment 1 (2026-11-11, written before the exam run): the run uses the top 40 pairs

Nothing in the exam window has been run. This section records the memory measurement that section 3 called for and what it decides.

- **Measurement** (November-December 2024, before the exam window): 40 pairs over 59 days peaked at 3.33 GB, about 0.037 GB per pair per month. The strategy keeps roughly 500 indicator columns per 5-minute candle for every pair for the whole run.
- **What that implies for 22 months**: 80 pairs need about 65 GB, 60 pairs about 49 GB, 40 pairs about 33 GB.
- **What the container has**: 15 GB of memory, plus a 26 GB swap file added for this run, 41 GB in all.
- **Decision by the ladder in section 3**: the exam runs on the **top 40 pairs** of `nfi_pairs.json` (the first 40 entries). 40 is the lower end of the strategy's recommended 40 to 80. The engine is unmodified freqtrade 2024.11.
- **If the 40-pair run is killed for memory**: inconclusive, per section 6. No smaller list and no splitting of the window.

Setup checks done on the same pre-window data:

- **Determinism**: 10 pairs over December 2024 run twice gave the same 20 trades, order for order.
- **Scoring code**: the equity rebuilt from exported orders ends at 13,065.0042 USDT against freqtrade's own final balance of 13,065.0042.
- **What the setup produces on the strategy's own tuning history** (5 November to 31 December 2024, 40 pairs; this is in-sample and is not evidence of anything): +30.7% in eight weeks, 67 trades, every one a winner, no drawdown. This is the kind of backtest the repository publishes, which confirms the frozen code runs as its authors intended.
