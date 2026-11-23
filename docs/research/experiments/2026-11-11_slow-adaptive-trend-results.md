# Slow, adaptive, many-coin system: Stage A results (design on 2020-24)

**Date**: 2026-11-11
**Preregistration**: `2026-11-11_slow-adaptive-trend-prereg.md` (commit `4bceb53`, before any Stage A code was run; Amendment 1 added after Stage A and before any forward record).
**Status**: Stage A complete. Stage B (forward recording) is built and tested but has recorded nothing yet: the scheduled job starts once its workflow file is on `develop`.
**Scripts**: `experiments/strategy_search/adaptive.py design`, `selfcheck.py`. Output: `results/adaptive_design.json`, `results/adaptive_design_daily_returns.csv`.
**Data**: the Phase 1b cache, cut at 2024-12-31 before any strategy code ran. Fit 2020-07-01 to 2022-12-31, validation 2023-01-01 to 2024-12-31. Nothing in 2025-26 was touched.

## Verdict

**H0 (design) is supported.** None of the four pre-specified changes earns a place beside the plain trend book. Three of them made it worse in both periods, and the machine-learning ranker passed in one period and failed in the other. By the selection rule, **the baseline B0 goes forward alone.**

## Results

Sharpe ratio is return in excess of 4% a year divided by volatility, after fees, slippage and funding. All books are scaled to 15% annual volatility.

| Book | Fit Sharpe | Fit return / yr | Fit drawdown | Validation Sharpe | Validation return / yr | Validation drawdown | Outcome |
|---|---|---|---|---|---|---|---|
| **B0** plain trend, top 30 | **0.86** | 19.7% | 15.1% | **0.52** | 12.2% | 17.5% | recorded forward in every case |
| V1 selective (all three lookbacks agree) | 0.74 | 16.7% | 20.6% | 0.34 | 8.5% | 19.0% | fails: worse than B0 in both |
| V2 broad (top 50) | 0.67 | 15.3% | 14.4% | 0.26 | 7.3% | 14.4% | fails: worse than B0 in both |
| V3 regime switch (trade only when the market is trending) | −0.14 | 1.6% | 20.1% | −0.65 | −5.9% | 24.6% | fails: loses money, drawdown over 20% |
| V4 machine-learning ranker | 0.15 | 5.1% | 23.8% | 0.73 | 16.0% | 12.3% | fails: fit Sharpe 0.15 against the 0.7 required |

V1 to V3 needed to beat B0's Sharpe by 0.15 in both periods. V4 needed a Sharpe above 0.7 in both. V4's fit figure covers 2021-03-15 to 2022-12-31, the part of the fit period where it had 52 weeks of training data, as preregistered. For scale, holding BTC had a Sharpe of 0.63 in the fit period and 1.93 in validation (a 5.6-fold rise in two years).

B0 reproduces the Phase 1b figure for `T_slow_30_60_120` exactly, which is the check that this run uses the same simulator and data.

### Secondaries (reported, never decisive)

| Book | Turnover / yr (validation) | Trading cost / yr | Funding paid / yr | Average gross exposure | Correlation with BTC | Sharpe if filled at the signal close (fit / validation) |
|---|---|---|---|---|---|---|
| B0 | 12.4x | 1.8% | 0.6% | 0.37 | 0.14 | 0.75 / 0.45 |
| V1 | 13.9x | 1.9% | 0.4% | 0.30 | 0.17 | 0.67 / 0.32 |
| V2 | 11.5x | 2.1% | 1.1% | 0.36 | 0.12 | 0.64 / 0.30 |
| V3 | 8.7x | 1.2% | 0.8% | 0.17 | 0.16 | −0.37 / −1.00 |
| V4 | 43.7x | 6.1% | −2.7% (received) | 0.60 | −0.11 | 0.14 / 0.78 |

Calendar years, B0: 2020 +31.3%, 2021 +21.4%, 2022 +2.4%, 2023 +19.9%, 2024 +5.1%. No losing year, but the return roughly halves from the first period to the second.

## What each result says

- **V1, selectivity.** Dropping the coins whose 30-, 60- and 120-day trends disagree removed profitable positions, not noise. Partial agreement carried information.
- **V2, breadth.** Coins ranked 31 to 50 by liquidity added cost (they sit in the 20 bp slippage tier and paid more funding) and no extra trend return. The edge is in the most traded coins.
- **V3, adapting to the market type.** The switch was on in 39% of fit weeks and 44% of validation weeks, and those were the wrong weeks. Splitting B0's own daily returns by the switch's state (Sharpe before the 4% hurdle): fit period 0.36 when on, 1.55 when off; validation −0.49 when on, 2.26 when off. A market that has already moved in a straight line for 60 days was late in its move. The idea as specified, "trade trend when the market is trending", is refuted on this data. The switch also could not turn on before 2020-10-26, because it needs 180 days of history; that costs it the strong second half of 2020, but the validation period has no such handicap and fails on its own.
- **V4, better prediction.** The ranker's predicted order of next week's returns correlated 0.03 with the realised order in the fit period and 0.07 in validation, and its long leg beat its short leg in 50% and 56% of weeks. That is a small amount of skill in one period and none in the other, bought with 44x turnover and 6% a year in trading costs. It cleared the validation bar (0.73) and missed the fit bar by a wide margin (0.15). The rule required both, because one good period out of two is what chance produces.

**An observation, not a result.** The V3 split suggests the reverse rule (trade trend when the market has *not* been trending) would have looked good in both periods. That was found by looking, on the same data, after the preregistered rule failed, so it carries no weight. It would need its own preregistration and a forward test.

## What this means for the question Alex asked

The question was whether the edge is better prediction plus a strategy that adapts to the market type. On 2020-24, with one honest configuration each, neither helped: the adaptive switch made the plain book much worse, and the learned ranker did not beat it reliably. The plain slow trend book is what there is. Its 2023-24 Sharpe of 0.52 is modest, and whether it holds up after 2024 is exactly what the forward recording is for.

## Checks

- **Determinism**: Stage A was run twice; `adaptive_design.json` was byte-identical (SHA-256 of the daily returns `d0b0b6fe…6d3f`).
- **Leakage (V4)**: `selfcheck.py` trains the ranker on pure noise and it earns nothing (t = 1.44); given features dated one week ahead it earns a great deal (t = 53.6), so the noise test would have caught a leak. The training set for each prediction is asserted in code to contain only weeks whose outcome ended before the prediction date.
- **Regime switch timing (V3)**: `selfcheck.py` checks that the switch on a given day does not move when that day's or any later value changes, and that the switched book carries the unswitched book's leverage.
- **Environment**: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, one thread.
- **Counting**: the 2020-24 design data has now seen 25 variants (21 in Phase 1b, 4 here). Nothing was selected, so nothing rests on that count.

## Stage B as built (nothing recorded yet)

- `tools/paper_trend_recorder.py`, scheduled by `.github/workflows/paper-trend-recorder.yml`, writes one file per UTC day to the `data/paper-trend-recorder` branch: B0's target weights on Mondays, a heartbeat on other days. It reads the public Binance archive only. No account, no orders, no money. The live bot is not touched and stays paused.
- `experiments/strategy_search/paper_score.py` scores the recorded weights against freshly downloaded prices and funding, recomputes the weights independently, and prints where the book stands against the fixed checkpoints (stop above 20% drawdown; plumbing at week 8; returns at weeks 26 and 52).
- **Plumbing checks so far.** The recorder's weights for 2026-10-26, computed from 420 days of archive candles, match the weights from the full research cache with actual funding to 0.14% of gross exposure (same 30 coins). 32 unit tests cover the record format, the exit codes, half-published days, new listings, the delisting and gap handling, and the checkpoint rules. The scoring script was run end to end on dry-run records for 2026-10-26, 2026-11-02 and 2026-11-09 (written after the fact, so not a forward result and not kept): recorded and recomputed weights differed by 0.15% of gross on average.
- The operating details (timing, what counts as a record, missed days) are fixed in Amendment 1 of the preregistration. An independent code review found no leak in the Stage A code and four gaps in the recorder and scoring code; the fixes are in Amendment 2.

## Decision

Per the preregistration: B0 is recorded forward alone. No live trial is possible before week 26, and any trial needs Alex's decision, a futures executor that does not exist yet, and a risk review.
