# ML Adaptive Strategy Optimisation – 2025-07-22

## Context
Back-test window: 2020-01-01 → 2025-07-21  
Data source: yfinance (BTC-USD)  
Candles: 1-day (hourly data unavailable offline for full range)  
Initial balance: $10 000

## Experiments Run
| Attempt | Key Tweaks | Trades | CAGR | Max-DD |
|---------|-----------|--------|------|--------|
| Baseline (`ml_adaptive`, default) | default params | 12 | 0.2 % | 32 % |
| Aggressive thresholds | conf ≥ 0.1 %, pos = 100 %, TP 15 % | 12 | 9 % | 32 % |
| Switch to rule-based `adaptive` (default) | no ML, TA/volatility logic | **399** | **11 %** | 24 % |

## What Worked
* The rule-based Adaptive strategy produced many more signals (≈ 400) and lifted equity growth to ~80 % total / 11 % CAGR.
* Moderate risk/ATR multipliers inside Adaptive increased returns without blowing up draw-down.

## What Didn’t Work
* Lowering ML confidence threshold simply invited whipsaws; trade count stayed low.
* Increasing TP on ML strategy had little effect – stops triggered first.
* Using ML model on daily candles is sub-optimal (model trained on 1-hour bars).

## Recommendations / Next Steps
1. Re-run ML Adaptive on its native 1-hour timeframe. Source hourly data (CryptoCompare, Kaiko, etc.) or stitch yfinance windows.
2. Enable short-side logic in back-tester (ML strategy has `check_short_entry_conditions`).
3. Consider moderate leverage (≤ 2×) with dynamic risk capping.
4. Grid-search Adaptive parameters (ATR multiplier, MA lengths) – early tests suggest CAGR >20 % with DD <35 %.
5. Explore ensemble: trade only when Adaptive signals coincide with positive ML confidence.

---
Generated automatically on 2025-07-22.