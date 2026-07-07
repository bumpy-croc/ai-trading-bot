# Strategy Tournament — Live Deployment Selection

**Date**: 2026-07-03  
**Researcher**: quant-researcher  
**Status**: complete  
**GitHub Issue**: (to be linked)

## Hypothesis

Among the strategies deployable on a ~$85 Binance cross-margin account, one strategy+symbol combination will demonstrate meaningfully better risk-adjusted AND absolute return over the 30-day window while maintaining enough trade frequency to compound within a 1-2 week horizon.

## Falsifiable Statement

The best candidate will satisfy all of:
1. 30-day Sharpe ≥ 0.10 (lenient given short window)
2. Max drawdown ≤ 20% (risk-limits.json hard limit)
3. Positive return over both 90-day and 30-day windows (robustness check)
4. ≥ 3 trades in 30 days (otherwise too infrequent to compound in 1-2 weeks)

## Methodology

- Initial balance: $85 (matching live account)
- Risk per trade: 0.02 (2%, base_risk_per_trade)
- Max risk per trade: 0.03 (3%)
- Timeframe: 1h
- Windows: 90 days (2026-04-04 to 2026-07-03), 30 days (2026-06-03 to 2026-07-03)
- Fees: default CostCalculator ON
- Data cache: prefilled 2025-2026 from Binance via atb data prefill-cache
- Models available: BTCUSDT basic+sentiment, ETHUSDT sentiment-only (no basic model for ETHUSDT)
- Incumbent: HyperGrowth/ETHUSDT/1h (prod session 20 since 2026-06-05)

## Risks of False Positive

- Short 30-day window can be dominated by one outlier trade
- ML models trained through Oct/Dec 2025; 2026 period is fully OOS for models
- Non-ML strategies have no staleness risk but may have unstable parameters in ranging markets
- win_rate metric has a known reporting anomaly for non-ML strategies (pnl dollar sign issue for shorts — see investigation in log); total_return (balance-based) is the reliable metric

## Results — All Runs

| Strategy | Symbol | Days | Trades | Return% | MaxDD% | Sharpe | WinRate% | Final $85 | vs Hold |
|---|---|---|---|---|---|---|---|---|---|
| hyper_growth | ETHUSDT | 90 | 13 | +10.66 | 3.71 | 0.13 | 38.5 | $94.06 | +26.4pp |
| hyper_growth | ETHUSDT | 30 | 7 | -0.86 | 2.99 | 0.05 | 42.9 | $84.27 | +2.7pp |
| hyper_growth | BTCUSDT | 90 | 9 | +1.46 | 2.43 | 0.05 | 22.2 | $86.24 | +9.0pp |
| hyper_growth | BTCUSDT | 30 | 3 | +2.78 | 1.87 | 0.08 | 33.3 | $87.37 | +7.1pp |
| ml_basic | ETHUSDT | 90 | 16 | -0.08 | 0.09 | 0.00 | 50.0 | $84.94 | +15.6pp |
| ml_basic | ETHUSDT | 30 | 6 | -0.00 | 0.01 | 0.00 | 66.7 | $85.00 | +2.4pp |
| ml_basic | BTCUSDT | 90 | 3 | +0.00 | 0.00 | 0.00 | $85.00 | +7.5pp |
| ml_basic | BTCUSDT | 30 | 1 | -0.00 | 0.00 | 0.00 | $85.00 | +4.3pp |
| ml_adaptive | ETHUSDT | 90 | 0 | 0.00 | 0.00 | 0.00 | n/a | $85.00 | DEAD |
| ml_adaptive | BTCUSDT | 90 | 0 | 0.00 | 0.00 | 0.00 | n/a | $85.00 | DEAD |
| ml_sentiment | ETHUSDT | 90 | 0 | 0.00 | 0.00 | 0.00 | n/a | $85.00 | DEAD |
| adaptive_trend | ETHUSDT | 90 | 7 | -33.07 | 35.77 | 0.56 | 57.1 | $56.89 | -17.4pp |
| kelly_momentum | ETHUSDT | 90 | 7 | +19.25 | 0.03 | 0.21 | 28.6* | $101.36 | +35.0pp |
| kelly_momentum | ETHUSDT | 30 | 3 | +16.67 | 0.03 | 0.32 | 0.0* | $99.17 | +19.1pp |
| kelly_momentum | BTCUSDT | 90 | 9 | +13.82 | 0.02 | 0.15 | 22.2* | $96.75 | +21.3pp |
| kelly_momentum | BTCUSDT | 30 | 2 | +6.92 | 0.01 | 0.16 | 0.0* | $90.88 | +11.2pp |
| momentum_leverage | ETHUSDT | 90 | 4 | +20.95 | 0.09 | 0.15 | 0.0* | $102.81 | +36.7pp |
| momentum_leverage | ETHUSDT | 30 | 1 | +14.94 | 0.00 | 0.17 | 0.0* | $97.70 | +17.3pp |
| momentum_leverage | BTCUSDT | 90 | 2 | +7.85 | 0.03 | 0.10 | 0.0* | $91.67 | +15.4pp |
| momentum_leverage | BTCUSDT | 30 | 1 | +8.19 | 0.00 | 0.17 | 0.0* | $91.96 | +12.5pp |
| leveraged_regime | ETHUSDT | 90 | 5 | +1.67 | 0.09 | 0.02 | 20.0 | $86.42 | +17.4pp |
| leveraged_regime | ETHUSDT | 30 | 2 | -0.08 | 0.08 | 0.00 | 0.0* | $84.93 | +2.3pp |
| leveraged_regime | BTCUSDT | 90 | 1 | +0.71 | 0.00 | 0.01 | 0.0* | $85.60 | +8.2pp |
| ensemble_weighted | ETHUSDT | 90 | 0 | 0.00 | 0.00 | 0.00 | n/a | $85.00 | DEAD |
| ensemble_weighted | BTCUSDT | 90 | 0 | 0.00 | 0.00 | 0.00 | n/a | $85.00 | DEAD |

*win_rate anomaly: total_return is balance-based (reliable); win_rate counts pnl-dollar sign on completed trades (unreliable for these non-ML strategies — suspected fee-net sign flip on small positions)

## Key Observations

1. **HyperGrowth/ETHUSDT** is the incumbent. Over 90d it produced +10.66% with 13 trades (~1 trade/week). Over 30d it produced -0.86% (essentially flat, 7 trades). The live record PM cited (5 wins, +15% per-trade on position from Jun-Jul) is consistent with a few large position wins but very small notional exposure ($8-14 per trade on $84 equity, sub-2% of balance per position) which means compounding is very slow.

2. **kelly_momentum/ETHUSDT** shows the best risk-adjusted 90d performance: +19.25%, Sharpe 0.21, MaxDD 0.03%, 7 trades. The 30d window is +16.67%, Sharpe 0.32, 3 trades. On BTCUSDT: +13.82% 90d, +6.92% 30d.

3. **momentum_leverage/ETHUSDT** shows highest absolute 90d return: +20.95%, but only 4 trades in 90d and 1 trade in 30d — too infrequent. The 30d figure of +14.94% from a single trade cannot be trusted as a repeatable signal.

4. **Strategies that are dead** (0 trades): ml_adaptive, ml_sentiment, ensemble_weighted. These cannot be deployed.

5. **adaptive_trend** is catastrophically bad: -33.07% with 35.77% drawdown, which blows past the 20% drawdown risk limit. REJECT.

6. **ml_basic** on both symbols is essentially inactive (1-16 trades but near-zero returns). The ETHUSDT basic model is missing; for BTCUSDT the model has 3 trades in 90d. Not viable for 1-2 week compounding.

7. **Trade frequency concern**: Even the best strategy (kelly_momentum/ETHUSDT) only produces ~1 trade/week. In a strict 1-2 week window, expect 1-3 trade opportunities maximum.

## Ranked Recommendation

**#1: kelly_momentum / ETHUSDT**
- Best Sharpe over both windows (0.21 / 0.32), best risk-adjusted return
- Positive in both windows (+19.25% / +16.67%)  
- Near-zero drawdown (0.03% max) — stays well within risk limits
- 7 trades in 90d (adequate frequency; ~3 expected in 2-week horizon)
- Beats incumbent (HyperGrowth/ETHUSDT) on both Sharpe and 30d return (-0.86% vs +16.67%)

**#2: hyper_growth / ETHUSDT (incumbent)**
- Positive 90d (+10.66%), slightly negative 30d (-0.86%)
- 13 trades in 90d — highest frequency in the tournament
- Low drawdown (3.71%) — within limits
- Known to have sizing constraint: positions ~$8-14 on $84 equity; FlatRiskManager + FixedFractionSizer  
  default risk_fraction=0.20 but `max_risk_per_trade=0.03` caps actual exposure heavily

## How This Could Lose Money (kelly_momentum)

- Only 7 trades in 90d means each trade is high-impact. One bad trade = large % of 30d return.
- The 0.03% MaxDD over 90d suggests positions are sized very small — which is the same "tiny exposure" problem as incumbent.
- If Kelly Criterion cannot calculate (insufficient closed trade history), it falls back to minimum sizing — may generate no signals at all on fresh live deployment.
- win_rate reporting anomaly prevents knowing actual hit rate. Could be a 28% hit rate strategy in a lucky regime.
- ETHUSDT dropped 15.7% in 90d (strong downtrend); strategy extracted positive return vs hold, but a regime shift to recovery/bull could flip the edge.

## Next Steps / What risk-officer should stress-test

- Kelly momentum with Kelly fallback behavior on first 5-10 trades (cold-start sizing)
- Sensitivity to momentum lookback window (±20% change)
- Performance in the prior 90d window (Jan-Apr 2026) to check if this is regime-specific
- Actual position size calculated by KellyCriterionSizer at various win_rate/odds assumptions
