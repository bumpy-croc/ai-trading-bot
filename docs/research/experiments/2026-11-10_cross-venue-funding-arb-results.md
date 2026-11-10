# Cross-venue funding arbitrage, Binance vs Hyperliquid (Phase 1c): results

**Date**: 2026-11-10
**Prereg**: [2026-11-10_cross-venue-funding-arb-prereg.md](2026-11-10_cross-venue-funding-arb-prereg.md) (one configuration, gates unchanged).
**Script**: `experiments/strategy_search/xvenue.py` (`download`, `sanity`, `holdout`). Raw numbers: `experiments/strategy_search/results/xvenue_*.json`.

## Verdict

**FAIL. H0 is supported.** The trade earned 10.7% a year after costs in 2023-24 and lost 2.8% a year in the 2025-26 holdout. The funding gap between the two venues roughly halved, and what is left does not cover the cost of getting in and out.

| Gate (holdout 2025-01 → 2026-10) | Required | Result | |
|---|---|---|---|
| G1 net CAGR on total capital | ≥ 8% | **-2.8%** | FAIL |
| G2 Sharpe in excess of idle USDT | ≥ 1.0 | -4.4 | FAIL |
| G3 max drawdown | ≤ 10% | 6.2% | pass |
| G4 2025 and 2026 both positive | yes | -2.9%, -2.3% | FAIL |
| G5 a slot filled on ≥ 50% of days | yes | 83% | pass |

This was candidate 5 on the 2025-01 → 2026-10 calendar window. The window is now closed for every crypto strategy.

## Order of events

1. Prereg committed (ec73475) before any Hyperliquid funding data was downloaded.
2. Candles for all 235 Hyperliquid perps, funding to 2024-12-31 only, for the 75 coins ever eligible in that period.
3. Sanity run on 2023-07 → 2024-12, twice, byte-identical. It passed (CAGR above idle USDT) and was committed (814e631).
4. Only then was 2025-26 funding downloaded (114 coins) and the holdout run, twice, byte-identical.

## Numbers

All returns are net, on total capital, with each leg at 1x (hedged notional is at most 50% of capital per side).

| Version | Window | CAGR | Max DD | Funding earned /yr | Basis P&L /yr | Costs /yr | Round trips | Forced closes | Avg slots (of 5) |
|---|---|---|---|---|---|---|---|---|---|
| **Primary** (7d, enter 20%, exit 5%) | 2023-07 → 2024-12 | **10.7%** | 0.7% | 14.5% | 0.1% | 4.4% | 90 | 0 | 4.6 |
| **Primary** | **2025-01 → 2026-10** | **-2.8%** | 6.2% | 2.2% | 0.3% | 5.4% | 124 | 5 | 2.4 |
| 2x legs, forced close at 40% | sanity / holdout | 20.9% / -5.7% | 1.3% / 12.4% | 29.2% / 5.1% | | 10.5% / 11.5% | 91 / 123 | 5 / 8 | |
| Lookback 3d | sanity / holdout | 8.0% / -6.7% | 1.8% / 13.1% | | | 7.6% / 9.6% | 171 / 249 | | |
| Lookback 14d | sanity / holdout | 10.6% / 0.0% | 0.6% / 1.8% | 13.1% / 3.0% | | 3.1% / 2.9% | 56 / 64 | | |
| Enter at 10% | sanity / holdout | 10.3% / -5.8% | 1.1% / 11.3% | | | 4.9% / 9.8% | 104 / 242 | | |
| Enter at 30% | sanity / holdout | 11.0% / -1.1% | 0.3% / 3.2% | 13.9% / 0.9% | | 3.5% / 2.2% | 73 / 56 | | |

No secondary version is positive in the holdout after the 4% idle-USDT hurdle. The best (14-day lookback) is flat.

### The gap itself (eligible coins, annualised)

| | 2023-07 → 2024-12 | 2025-01 → 2026-10 |
|---|---|---|
| Mean Hyperliquid funding | 20.2% | 5.6% |
| Mean Binance funding | 10.4% | -0.5% |
| Mean gap (Hyperliquid − Binance) | 9.8% | 6.0% |
| Median absolute gap | 13.0% | 5.5% |
| BTC mean gap | 10.6% | 4.1% |

## Reading

- **The P&L in 2023-24 was honest funding income, not a basis artefact**: 14.5 points of funding against 0.1 of basis. The mechanism is real.
- **It is a bull-market premium.** Funding on both venues collapsed in 2025-26 (Hyperliquid from 20% to 6%, Binance from 10% to about zero). The gap between them shrank with the level. In the holdout the 7-day gap still crossed the 20% entry line often enough to trigger 124 round trips, but those gaps closed within days, so the rule paid 5.4% a year in costs to collect 2.2%.
- **The slow version loses least.** A 14-day lookback trades half as often and breaks even. That says the persistent part of the gap is worth about 3% a year on capital at present, below idle USDT.
- Five forced closes in the holdout against none before: the same squeeze-prone tape that broke the Phase 1b carry trade.

## Deviations and notes

- Liquidity rank is taken across all 235 Hyperliquid perps (as the prereg says), then restricted to the 213 that pair with a Binance crypto perp.
- "Avg slots" in the first version of the script was computed from peak exposure; it was replaced by a direct slot count before any run.
- Funding coverage on eligible days: 100% on both venues in both windows.
- Not in the numbers, and all of it would make a live result worse: bridge and smart-contract risk on Hyperliquid, withdrawal delays, auto-deleveraging.

## Decision implied by the prereg

No build. With Phase 1 (#1315), Phase 1b (#1324) and this Phase 1c all negative, and the parallel non-crypto bake-off (#1325) negative as well, the recommendation to the Board is to treat systematic trading at this account size as not viable in the present regime, keep the bot halted, and keep (and widen) the funding watcher, because every premium found in this programme is a funding premium that pays when leverage demand is high and pays nothing when it is not.
