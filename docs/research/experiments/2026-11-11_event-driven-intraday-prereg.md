# Event-driven intraday trades (Phase 1d): preregistration

**Date**: 2026-11-11
**Status**: PREREGISTERED. Three ideas, one configuration each, locked before any hourly price data is downloaded or any trade P&L computed. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-11, project thread: chose "Keep searching new ideas" (intraday and event-driven) after Phases 1, 1b and 1c (#1315, #1324, #1326) all came back negative.
**Scripts**: `experiments/strategy_search/events.py`.

## 0. What is different this time

Phases 1-1c held positions for days or weeks and lived or died on premia that depend on the market regime. These three trades last two to four hours and are triggered by a specific event in one coin. Each has a mechanical reason to exist:

| ID | Trade | Why it might pay |
|---|---|---|
| E1 | **Funding-settlement capture**: just before a funding payment, take the side that gets paid, in coins where the payment is extreme (≥ 0.5% per settlement). Close an hour after. | The payment is known in advance and is large relative to two hours of costs. It pays if the price does not fully move against the receiver around the settlement. |
| E2 | **Buy capitulation**: after a coin falls very hard in one hour on a volume spike, buy it and hold four hours. | Forced liquidations push price past fair value; the rebound pays whoever provides liquidity. |
| E3 | **Follow a volume breakout**: after a coin rises very hard in one hour on a volume spike, buy it and hold four hours. | Short squeezes and news take hours to play out in thin-float coins (the MYX pattern from Phase 1b, taken from the other side). |

The 2026-08-13 scan (#1105) looked at next-bar predictability one coin at a time on 20 symbol/timeframe cells. It did not condition on extreme events across the whole perp universe, and it did not look at funding settlements.

## 1. Hypotheses

For each idea, **H1**: the trade has a positive net expected return per trade after realistic costs, stable across 2020-22, 2023-24 and 2025-26. **H0**: the price already adjusts (E1), the move continues instead of reversing (E2), or reverses instead of continuing (E3), or the edge is smaller than costs.

Supported / falsified by the gate in section 5, per idea.

## 2. Data

- Binance USDⓈ-M hourly klines from the public archive, for the symbol-months each idea needs, plus the Phase 1b daily cache (liquidity ranks) and the per-print funding history already downloaded.
- Crypto-asset perps only (same exclusion lists as Phase 1b). Delisted coins included.
- Liquidity rank = Phase 1b's rank (trailing 30-day median quote volume, 60 daily closes), **taken from the previous UTC day** so it is known when the event happens.
- Range: 2020-01-01 to 2026-10-31.

## 3. Rules

Costs for all three: fee 5 bps per side plus slippage per side by liquidity rank (1-10: 5 bps, 11-30: 10 bps, 31+: 20 bps). **For E2 and E3 slippage is doubled**, because the entry is made into a disorderly market. One open trade per coin per idea: an event that fires while that coin is already held is skipped.

**E1 funding-settlement capture**
- Universe: liquidity rank ≤ 100.
- Trigger: a funding settlement at time T whose *previous* settlement for the same coin was ≥ 0.5% in absolute value. (The previous print is known; the coming one is not used to decide.)
- Enter at T − 1h (the open of the hourly bar that ends at T). Short if the previous print was positive, long if negative.
- Exit at T + 1h (the close of the bar that starts at T).
- P&L = side × price return over the two hours + funding received at T (the actual print at T, which is negative for us if its sign flipped) − costs. Only the print at T is counted, even when the coin settles hourly.

**E2 buy capitulation**
- Universe: liquidity rank ≤ 50.
- Trigger at the close of hour h: the hour's return ≤ −max(3%, 4σ), where σ is the standard deviation of that coin's hourly returns over the previous 168 hours (at least 120 present), and the hour's quote volume ≥ 3× its mean over the same 168 hours.
- Enter long at the open of hour h+1. Exit at the close of hour h+4. Funding settlements inside the hold are paid or received at the real rate.

**E3 follow a volume breakout**
- Same as E2 with the sign flipped: hour return ≥ +max(3%, 4σ), same volume condition. Enter long at the open of h+1, exit at the close of h+4.

## 4. Portfolio rule used for the return gate

Each trade is sized at 10% of equity, at most 5 open trades per idea (an event that arrives when 5 are open is skipped). No leverage beyond that (gross ≤ 0.5). Idle capital earns nothing. Trade P&L is booked on the exit day.

## 5. Metric and gates (per idea; all required)

**Primary metric**: mean net return per trade, with a t-statistic computed on daily sums of trade returns (events cluster in time, so trades are not independent; days are the unit).

1. At least 300 trades.
2. Mean net return per trade > 0 with day-clustered t ≥ 3.0. (Three ideas tested; 3.0 is roughly a one-sided 5% level after Bonferroni, with margin.)
3. Mean net return per trade is positive in every sub-period (2020-22, 2023-24, 2025-26) that has at least 50 trades, and 2025-26 has at least 50.
4. Portfolio rule: net CAGR ≥ 10% from the first trade to 2026-10-31, max drawdown ≤ 20%.
5. Cost stress: the mean net return per trade is still positive with every slippage figure doubled again.

Secondaries, reported and never decisive: win rate, median, worst trade, P&L by liquidity tier and by year, E1 split into funding received versus price move, E2/E3 with a BTC hedge of equal notional, E2/E3 at 1, 2, 8 and 24-hour holds, the mirror trades (shorting instead of buying).

## 6. About reusing 2025-26

Phase 1c's file says the 2025-01 → 2026-10 window is closed for crypto strategies. Alex has since asked for the search to continue, and no later data exists yet, so this experiment uses the full history including that window, and says so here instead of doing it quietly. Three things limit the damage:

- **No selection step.** Each idea has one configuration fixed above. Nothing is tuned on any period and then "confirmed" on another.
- **The bar is statistical and multi-period**, not a single-window Sharpe: thousands of independent events, a t ≥ 3 requirement, and positivity in each sub-period.
- **What is known about 2025-26 from earlier phases** (prices fell, funding was low, squeezes were frequent) is a daily-level fact. E1 and E2 were not designed from it. **E3 was**: the idea of joining squeezes comes directly from watching MYX in the Phase 1b holdout. E3 is therefore flagged as contaminated, and a pass for E3 counts for less.

A pass in this experiment is **a candidate for paper trading, not proof**. The clean out-of-sample test for any of these is forward data.

## 7. Decision each outcome triggers

- **An idea passes all five gates**: proposal to the Board for an 8-week paper trial of that trade (it needs an hourly event scanner over the perp universe and a futures executor; nothing is deployed from this experiment).
- **An idea passes gates 1-3 but fails 4 or 5**: reported as "real but too small to trade"; no build.
- **All fail**: no build. Record it. The descriptive section below then stands as the only positive finding.

## 8. Descriptive section, not a strategy test: what passive yields actually paid

Alongside the three trades, record the realised returns of passive dollar-denominated yield sources over the same years, because (a) that is the real hurdle a strategy has to beat, and (b) Phases 1-1c found that the premia a small trader can harvest are yield-like:

- Hyperliquid's HLP vault (the venue's own market-making vault), from its on-chain account-value and P&L history: return per year, worst drawdown.
- Stablecoin lending rates (Aave v3 USDT/USDC on Ethereum, from DefiLlama's history): average per year.

These are realised track records, not backtests. No gate applies. They carry smart-contract, bridge and depeg risk that a backtest number does not show, and the report will say so.

## 9. Risks of a false positive

- **Hourly bars hide the path.** E1's entry and exit are at bar boundaries; real fills around a settlement in a squeezing coin are worse than the tier slippage. Gate 5 doubles slippage for that reason.
- **E2/E3 enter at the open after an extreme bar**, when spreads are widest. Slippage is doubled in the base case and quadrupled in the stress.
- **Event clustering**: handled by day-clustered t-statistics and the 5-trade cap.
- **Survivorship**: none; delisted coins are in.
- **Look-ahead**: σ and average volume use only hours before the event hour; rank is from the previous day; E1 decides on the previous print. `selfcheck.py` gets a synthetic test for each rule.
- **Determinism**: run twice, outputs must match.
