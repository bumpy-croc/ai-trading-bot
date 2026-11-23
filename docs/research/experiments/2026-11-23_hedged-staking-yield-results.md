# Staking with a hedge (ETH and SOL), 2023-2026: results

**Date**: 2026-11-23
**Preregistration**: `2026-11-23_hedged-staking-yield-prereg.md` (commit `18b9c96`, before the calculation was run).
**Script**: `python experiments/strategy_search/yields.py hedged`. Output: `results/hedged_staking.json`.

## Verdict

**H0 is supported for both coins. Staking with a hedge now pays about what lending dollars pays, or less.** No build.

It was a good trade in 2023-24 (10% to 14% a year on total capital for ETH) and has faded every year since, for the same reason every other yield in this project faded: the funding that shorts collect has gone to almost nothing, and staking yields themselves have fallen.

## Net yield on total capital (primary: 25% extra idle margin, after 0.30% a year in costs)

| | 2023 | 2024 | 2025 | 2026 (Jan-Oct, annualised) | Gate: at least 7% in 2025 and 2026 |
|---|---|---|---|---|---|
| **ETH** (Lido stETH, short ETHUSDT perp) | 10.2% | 13.5% | **6.1%** | **3.4%** | fail |
| **SOL** (JitoSOL, short SOLUSDT perp) | no data | no data | no full year | **3.7%** | fail |
| Idle dollars | 4% | 4% | 4% | 4% | |
| Aave USDT lending | 4.1% | 6.3% | 4.0% | 3.0% | |
| Ethena sUSDe (the same trade, run at scale) | | 17.5% | 6.5% | 4.2% | |
| Hyperliquid HLP vault | 13.2% | 62.9% | 17.3% | 7.9% | |

The comparison rows are from the 2026-11-11 survey (`results/yields.json`); HLP's 2023 covers June to December and its 2026 runs to late October, not annualised.

## Where the yield came from (per hedged dollar, before costs)

| | Year | Staking | Funding received | Change in perp-spot gap | Net, token as collateral | Days with negative funding |
|---|---|---|---|---|---|---|
| ETH | 2023 | 4.3% | 8.3% | -0.2% | 12.9% | 5% |
| ETH | 2024 | 3.2% | 12.9% | 0.0% | 17.2% | 3% |
| ETH | 2025 | 2.8% | 4.9% | -0.1% | 7.7% | 9% |
| ETH | 2026 | 2.4% | 2.1% | 0.0% | 4.2% | 28% |
| SOL | 2026 | 5.4% | -0.6% | 0.1% | 4.7% | 46% |

- "Token as collateral" assumes the staking token itself backs the short, with no extra margin. Even on that generous basis 2026 is 4.2% for ETH and 4.7% for SOL.
- **SOL history.** The JitoSOL yield series starts on 2025-03-26, so 2026 is the only full year the preregistered definition covers, and the 2025 gate cannot be met. For information only: from 26 March to December 2025 staking averaged 6.9% and funding 0.6%, about 5.8% on total capital after costs. SOL funding alone was 1.4% in 2023, 13.6% in 2024, 0.4% in 2025 and -0.6% in 2026.
- Largest drawdown in 2025-26, marked daily: 0.2% (ETH), 0.8% (SOL). The position is quiet. It simply does not pay.
- Latest staking yields (23 November 2026): stETH 2.3%, JitoSOL 4.9%.

## Reading

The hedged staking trade is the retail version of what Ethena does, and its record matches Ethena's: about 17% in 2024, about 4% now. In 2026 the short paid funding on 28% of days for ETH and 46% for SOL. The structure carries risks the numbers do not show (the staking token trading below the coin, the exchange, liquidation of the short in a fast rally) for a return at or below a plain lending rate.

This closes the "no prediction" yield list. Ranked by what they paid in 2026: the Hyperliquid vault about 8%, Ethena 4.2%, idle dollars 4%, hedged SOL 3.7%, hedged ETH 3.4%, Aave USDT 3.0%.

## Decision

Per the preregistration: **H0, no build.** If funding returns (the daily funding watcher from #1323 opens an issue when BTC or ETH funding exceeds about 10% a year), this structure and plain carry both become worth re-measuring.
