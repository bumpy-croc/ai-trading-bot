# The trend book on 2025-26, once: preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. Written and pushed before either book has been run on any day after 2024-12-31. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-24 10:44 UTC, project thread: chose "Yes, stricter bar, paper first" on the decision card "Allow one-off tests on the 2025-26 period I had declared used up?". The option read: "I test the trend book and each new idea once on 2025-26, written down first. A pass must still paper trade before any money."
**Script**: `experiments/strategy_search/adaptive.py exam` (output `results/trend_exam.json`, `results/trend_exam_daily_returns.csv`).

## 0. Why, and what changes about the window

The 2025-01-01 to 2026-10-31 window was declared closed on 2026-11-11 after nine daily-bar candidates had faced it (log `D-2026-11-11-04`). The slow trend book B0 was the nearest miss of the whole search and was never run on it; it went straight to a forward paper test (day 0 = Monday 2026-11-23), where a verdict takes 26 to 52 weeks.

Alex has now allowed one look per candidate at that window, on two conditions that this file implements: the bar is stricter than before because the window has been used, and a pass leads only to paper trading. This file covers two candidates, which makes eleven for the window.

## 1. Candidates (one configuration each, nothing varied)

**B0, the plain trend book.** Exactly as specified in Phase 1b (`T_slow_30_60_120`) and recorded forward since 2026-11-23: the 30 most liquid Binance USDT perpetuals (point-in-time), signal per coin = average of the signs of its 30-, 60- and 120-day returns, weight proportional to signal divided by 30-day volatility, weekly at the Monday close, filled one day later, scaled to 15% annual volatility, gross exposure capped at 2. Design figures (2020-24 only): Sharpe 0.86 in the fit period, 0.52 in validation, no losing calendar year.

**Q1, the trend book in quiet markets only.** B0's weights, switched off in weeks when the market has been moving in a straight line. The switch is the mirror image of V3 from the 2026-11-11 design: trendiness = efficiency ratio of the equal-weight top-30 index over 60 days; Q1 is on when today's value is at or below the median of all its past values (at least 180 days of history, today excluded) and off otherwise, including while there is not enough history. The switch is applied after volatility scaling, as for V3, so time spent flat does not raise leverage.

Where Q1 comes from, stated plainly: it was not designed in advance. The 2026-11-11 design run tested the opposite rule (V3, trade only when the market is trending), which failed, and the results file recorded as "an observation, not a result" that B0 earned nearly all its return when that switch was off. Q1 is that observation written as a rule. On 2020-24 it shows Sharpe 0.78 in the fit period (below B0's 0.86) and 1.26 in validation (above B0's 0.52), with drawdowns of 8.8% and 7.1%, on in 48% and 56% of weeks; years 2020 +23.0%, 2021 −4.0%, 2022 +16.5%, 2023 +6.7%, 2024 +30.8%. Those figures were found by looking, so they are worth little. The run below is its first out-of-sample test.

## 2. Hypotheses

**H1 (B0)**: slow trend across the liquid coins kept earning in 2025-26: a positive return above idle dollars, with a drawdown inside what 2020-24 showed.

**H0 (B0)**: the 2020-24 result belonged to that period (a rising market with high funding paid to shorts in the falls), and in 2025-26 the book earns nothing after costs or loses.

**H1 (Q1)**: trend entries pay after quiet stretches and are late after straight-line moves, so Q1 beats B0 on risk-adjusted return out of sample as it did in 2023-24.

**H0 (Q1)**: the 2020-24 split was noise found by looking, and Q1 is no better than B0, or worse, in 2025-26.

## 3. Data, window, costs

The Phase 1b cache as built on 2026-11-10 (918 Binance USDT perpetuals including 166 delisted, daily candles and funding, to 2026-10-31), not refreshed. Exam window 2025-01-01 to 2026-10-31 (669 days). Costs are Phase 1b's, unchanged: fee 5 bp per side, slippage 5 / 10 / 20 bp by liquidity rank, actual funding, fills one day after the signal.

## 4. Metrics and gates

Primary metric: Sharpe ratio in excess of 4% a year, net of all costs, over the exam window, and the t-statistic of the mean daily excess return (the same quantity scaled by the square root of the number of days; a Sharpe of 1.0 over this window is a t of 1.35).

**PASS (the stricter bar)**, all of:

1. t-statistic at least 2.61. This is the one-sided 5% level shared across the eleven candidates the window has now faced. It corresponds to a Sharpe of about 1.9.
2. Return at least 10% a year.
3. Maximum drawdown at most 20%.
4. 2025 and January-October 2026 both positive.
5. B0 only: exposure on at least 80% of days. (Q1 is flat about half the time by construction; its share of weeks on is reported.)

**CONSISTENT** (not a pass; the book behaved in line with its design, which 22 months cannot prove or disprove): Sharpe at least 0.5, total return over the window above idle dollars, drawdown at most 20%.

**FAIL**: anything else.

Reported for both, never decisive: return and drawdown, each calendar year, worst day, turnover, cost and funding per year, correlation with BTC, Sharpe if filled at the signal close, and the same window for BTC and for the equal-weight top 30.

## 5. Decision per outcome

**B0**

- PASS: propose to Alex that the futures executor a live trial would need is built now, so that a small trial could start at week 26 if the forward checkpoint is also met. The forward test and its checkpoints do not change. No money moves on this result.
- CONSISTENT: the forward paper test continues unchanged. No build.
- FAIL: reported as a failed out-of-sample period. The forward recording continues to its fixed checkpoints because it costs nothing and its rules were set in advance, and the report to Alex says the odds of it leading to a trial are now low. No executor work.

**Q1**

- Q1 is added to the forward recorder as a second paper book (its own day 0, under a dated amendment to the 2026-11-11 file) only if it is at least CONSISTENT **and** its exam Sharpe beats B0's by at least 0.15, the margin the design stage used. No money in any case.
- Otherwise Q1 is closed. No further versions of the switch (other windows, other thresholds) are tried on historical data.

Neither outcome changes the bot, which stays paused.

## 6. Risks of a false result

- **A used window.** Nine candidates before these two. The t-statistic gate carries the correction. A CONSISTENT verdict is deliberately weak and changes nothing.
- **Q1 is post-hoc.** It is the fifth add-on tried on the 2020-24 design data and the only one chosen after seeing a split. That is why it needs a margin over B0 out of sample, and why even then it only earns a paper slot.
- **Same data family.** Both books use the cache and simulator that every Phase 1b result used; an error there would be shared. B0's 2020-24 figures reproduce Phase 1b's exactly, which is checked again in the output (`fit`, `validation`).
- **Short window.** 22 months. A book with a true Sharpe of 0.7 has roughly a one in six chance of showing a negative Sharpe over a window this long, and a small chance of clearing 1.9. Both verdicts other than PASS should be read with that in mind.
- **Determinism**: the exam is run twice; the daily-return hash must match.
