# Slow, adaptive, many-coin system: design on 2020-24, then forward paper test: preregistration

**Date**: 2026-11-11
**Status**: PREREGISTERED. Stage A (design on data to 2024-12-31) has not been run. Stage B (forward recording) has not started. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-11, project thread: chose "Both, freqtrade test first". This file is the second half: "build the slow adaptive many-coin system as a paper trader".
**Scripts**: `experiments/strategy_search/adaptive.py` (Stage A), `tools/paper_trend_recorder.py` and `.github/workflows/paper-trend-recorder.yml` (Stage B), `experiments/strategy_search/paper_score.py` (scoring).

## 0. Why this, and what it can and cannot show

The research report of 2026-11-11 found one prediction-based style with a verified record: slow systematic trend trading across many coins (quant directional funds, about +30% from January 2025 to August 2026). It also found four levers that matter more than a cleverer model: a slow horizon, selectivity, breadth, and adapting exposure to the market type. Our own nearest miss in Phase 1b was exactly this style: long/short trend on the 30 most liquid perps (`T_slow_30_60_120`: Sharpe 0.86 on 2020-07 to 2022-12, 0.52 on 2023-24, positive in every calendar year). It failed the shortlist rule and was never run on 2025-26.

The 2025-01 to 2026-10 window is spent (nine candidates). So this experiment has two stages, and neither touches that window:

- **Stage A** decides, on data up to 2024-12-31 only, whether any of four pre-specified improvements earns a place beside the baseline.
- **Stage B** records the chosen books forward, every day, before the prices that will judge them exist.

**What Stage B cannot show quickly.** A strategy with a true Sharpe ratio of 0.7 needs about four years to be distinguished from zero at the usual level. Eight weeks of paper trading proves the plumbing and catches a blow-up; it does not prove an edge. The checkpoints in section 6 are written with that in mind: they are stop rules and a minimum bar for a small live trial, not a claim of statistical proof.

## 1. Hypotheses

**H1 (design)**: at least one of the four changes improves the baseline's risk-adjusted return in both the fit and the validation period, because it removes trades the baseline takes without conviction (V1), spreads the same edge over more coins (V2), stands aside when the market is not trending (V3), or ranks coins better than raw trend does (V4).

**H0 (design)**: none does. The baseline already captures what there is, and added rules fit noise.

**H1 (forward)**: the recorded books earn more than idle dollars after costs with a drawdown no worse than seen in 2020-24.

**H0 (forward)**: the 2020-24 result was a property of that period (rising market, high funding), and forward returns are zero or negative after costs.

## 2. Data (Stage A)

The Phase 1b cache: every Binance USDT perpetual that ever listed (delisted included), daily candles and funding, crypto-asset perps only, point-in-time liquidity rank (trailing 30-day median quote volume, 60 daily closes). **Truncated at 2024-12-31 before any strategy code sees it**, as in Phase 1b's discovery step. Fit: 2020-07-01 to 2022-12-31. Validation: 2023-01-01 to 2024-12-31.

Costs and mechanics are Phase 1b's, unchanged: fee 5 bp per side, slippage by liquidity rank (5 / 10 / 20 bp), actual funding, signals at the Monday close (UTC), fills one day later, scaled to 15% annual volatility on trailing 60-day realised volatility, gross exposure capped at 2.

## 3. Candidates (Stage A): one configuration each

**B0 baseline** (already measured, re-run as a check): `trend(lookbacks = 30, 60, 120)` on the top 30 by liquidity. Signal per coin = average of the signs of its 30-, 60- and 120-day returns. Weight proportional to signal divided by 30-day volatility, normalised to gross 1, weekly.

| ID | Change from B0 | Lever |
|---|---|---|
| V1 selective | A coin gets a position only when all three lookbacks agree (signal is +1 or −1). Otherwise it is flat. | Selectivity |
| V2 broad | Universe is the top 50 by liquidity instead of 30. | Breadth |
| V3 regime | The whole book is switched off (weights zero) when the market is not trending. Trendiness = efficiency ratio of the equal-weight top-30 index over 60 days: the absolute sum of its daily log returns divided by the sum of their absolute values. The book is on when today's value is above the median of all its past values (at least 180 days of history, computed point-in-time). The switch is applied after volatility scaling, so time spent flat does not inflate leverage. | Adapting to market type |
| V4 ML ranker | Replaces the trend signal with a learned cross-sectional ranking. Gradient-boosted trees (`sklearn` `HistGradientBoostingRegressor`, depth 3, 200 trees, learning rate 0.05, minimum leaf 50, fixed seed) predict each coin's return over the 7 days after the fill, relative to the universe average. Features, each converted to its percentile within the day's universe: returns over 7, 14, 30, 60 and 120 days; 30-day volatility; mean funding over 7 and 30 days; 7-day over 60-day median volume; distance below the 120-day high; liquidity rank. Trained on an expanding window, refit every 13 weeks, using only weeks whose outcome was fully known before the prediction date; first prediction after 52 weeks of training data. Portfolio: long the top fifth, short the bottom fifth by predicted score, equal weight, top 30 universe, weekly. | Better prediction |

No combinations are tested. No parameter of any candidate is varied.

## 4. Stage A metric and selection rule

Primary metric: Sharpe ratio in excess of 4% a year, net of costs, in the fit window and in the validation window separately.

- **V1, V2, V3**: a variant qualifies if it beats B0's Sharpe by at least 0.15 in **both** windows and its validation drawdown is at most 20%. If more than one qualifies, the one with the highest validation Sharpe is taken. If none qualifies, B0 goes forward alone.
- **V4** is judged on Phase 1b's shortlist bar, because it is a different strategy and not a tweak: Sharpe above 0.7 in both windows (the fit figure covers only the part of the fit window where predictions exist), validation drawdown under 25%, and positive total return over both windows together. It must also beat B0's validation Sharpe. If it passes, it is recorded forward beside the trend book.
- **B0 is recorded forward in every case.** It was specified in Phase 1b before any of this.

At most three books go forward. Secondaries, reported and never decisive: return per year, drawdown, turnover, cost and funding per year, correlation with BTC, share of weeks the regime switch is on (V3), rank correlation between predicted and realised weekly returns and directional accuracy of the long-minus-short call (V4).

**Counting**: Stage A adds four candidates to the 2020-24 design data, which has already seen Phase 1b's 21 variants. A pass here is weak evidence by construction. That is why the pass leads only to forward recording.

## 5. Stage B: forward recording

- A scheduled job runs once a day shortly after the UTC daily close. It downloads daily candles for all current Binance USDT perps, rebuilds each book's target weights from data up to the close just finished, and commits them to a data branch with the time of recording. Target weights for a Monday close are filled, in the scoring, at the Tuesday close, so every recorded weight pre-dates the price it is filled at by about a day.
- The job reads public market data only. No account, no orders, no money. The live bot is not touched and stays paused.
- **Scoring** uses the recorded weights and independently downloaded archive prices and funding, with Phase 1b's costs. It also recomputes the weights from archive data and reports any difference from what was recorded, which is the check that the live pipeline does what the backtest does.
- If V4 goes forward, it follows the rule it was designed under: an expanding training window, refit every 13 weeks on everything known at the time. The 2025-26 data enters its training set only. It is never used to evaluate V4.

## 6. Checkpoints and decisions (fixed dates; day 0 is the first recorded Monday)

| When | What is checked | Decision |
|---|---|---|
| Any time | Drawdown of a book from its forward peak exceeds 20% (the worst seen in 2020-24 was 17.5%) | That book is stopped and recorded as failed |
| Week 8 | Plumbing only: no more than 3 missed days, recorded weights match recomputed weights (mean absolute difference under 1% of gross) | Fix or continue. No claim about returns either way |
| Week 26 | Net return since day 0, drawdown, and the same for idle dollars, BTC and the equal-weight top 30 | Net return at least +5% with drawdown at most 15%: proposal to Alex for a small live trial of that book. Net return below zero: reported as leaning against, recording continues to week 52. In between: continue |
| Week 52 | Same | Net return at least +10% with drawdown at most 20%: proposal for a live allocation. Below idle dollars: stop, record as failed. In between: Alex decides whether to keep recording |

Any live trial needs Alex's decision, a futures executor that does not exist yet, and a risk review. Nothing in this experiment deploys anything.

## 7. Risks of a false result

- **Stage A overfitting**: four more looks at data that has had 21. Mitigated by one configuration each, a both-windows rule, and the fact that passing only buys a forward test.
- **V4 leakage**: labels overlap the prediction date if the training cut is wrong. The training set for a prediction on day t uses only weeks whose 7-day outcome ended before t. `selfcheck.py` gets a test that a model trained on pure noise earns nothing and that shifting the labels forward by a week does earn (the test has power).
- **Regime switch leakage (V3)**: the median is expanding and excludes the current day.
- **Forward test power**: stated in section 0. A good first 26 weeks is weak evidence; a bad one is also weak evidence. The stop rule protects capital; it does not settle the question.
- **Recorder failure**: a blind recorder must be visible. The job fails loudly if fewer than 100 perps return data, and posts to the alert webhook.
- **Venue**: Binance perps for both design and recording, so there is no venue mismatch. Execution on a live account would differ (slippage in size, liquidation rules) and is out of scope.
- **Determinism**: Stage A is run twice; outputs must match.
