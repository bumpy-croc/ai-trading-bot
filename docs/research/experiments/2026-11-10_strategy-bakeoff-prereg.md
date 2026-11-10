# Strategy bake-off (Phase 1): preregistration

**Date**: 2026-11-10
**Status**: PREREGISTERED. Gates and parameters below are locked before any result is computed. Corrections go in new dated sections, never edits.
**Why**: the 2026-07-12 returns-levers synthesis and the 2026-08-13 model-free scan (#1105) found no directional edge that beats ~11 bps round-trip costs. This bake-off tests strategies that earn a structural premium instead of predicting the next bar. Roadmap: "Trading bot: assessment and roadmap to profitability" (Board-approved 2026-11-10).

## Data

- Binance spot daily klines (UTC close), 2019-01-01 to 2026-10-31.
- Universe, fixed now: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, LINK, LTC, AVAX, DOT, TRX (all vs USDT). A coin joins once it has 200 days of history. This list is today's large caps, so results carry survivorship bias; report it, don't correct it.
- Binance USDⓈ-M perpetual funding history (8h), from listing to 2026-10-31, for the same coins.

## Costs (all strategies)

- Spot: 10 bps per side (taker fee) + 5 bps slippage.
- Perp: 5 bps per side + 5 bps slippage.
- Funding is paid or received on the perp leg at every 8h print.

## Strategies (parameters fixed, no sweeps)

- **S1 Funding carry.** Hold long spot + short perp of equal notional in BTC and ETH, 50/50 capital split; perp collateral is half of each leg's capital (1x effective short). Hold a coin's pair only while its trailing 7-day average funding is above 0; exit when it goes below 0, re-enter when above +0.005% per 8h. Return counted on total capital.
- **S2 Trend with volatility targeting.** Weekly (Monday close). For each eligible coin, long if close > 100-day SMA, else cash. Weight = (20% annualised vol target / N eligible coins) / that coin's 30-day realised vol, capped so total gross <= 1.0. Spot only, no leverage, no shorts.
- **S3 Cross-sectional momentum.** Weekly. If BTC close > its 100-day SMA, hold the top 3 coins by 28-day return, equal weight; otherwise all cash. Spot only.
- **Control.** Buy and hold BTC.

## Metrics

Net CAGR, annualised Sharpe (daily returns, rf = 0), max drawdown, return per calendar year, turnover, and fees paid.

## Pass gate (per strategy)

All of:
1. Net Sharpe > 1.0 over 2020-01-01 to 2024-12-31 (in-sample period; parameters were not fitted to it, but it is the period people knew about when choosing these designs).
2. Max drawdown < 25% over the same period.
3. Positive net return in at least 4 of the 5 calendar years 2020-2024.
4. Holdout 2025-01-01 to 2026-10-31: net Sharpe > 0.5 and max drawdown < 25%.
5. Robustness: changing each lookback by ±50% (one at a time) must not turn the 2020-2024 net Sharpe negative.

A strategy that passes goes to Phase 2 (8 weeks of paper trading on staging). If none passes, active development stops (roadmap Phase 1 fail branch).

## Not in scope

No live or staging change. No parameter search. No ML.
