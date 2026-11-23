# Staking with a hedge (ETH and SOL), 2022-2026: preregistration

**Date**: 2026-11-23
**Status**: PREREGISTERED. Definitions fixed and pushed before the calculation is run. Corrections go in new dated sections.
**Authority**: Alex, 2026-11-23, project thread: "merge the pr and carry on finding a profitable strategy", after the plan item "measure what the no-prediction yields paid in 2023-26, mainly staking with a hedge".
**Script**: `experiments/strategy_search/yields.py hedged` (extends the descriptive survey of 2026-11-11).

## 0. What is measured

Hold a liquid staking token and short the same amount of the coin's perpetual. The coin's price cancels out. What is left is the staking yield plus the funding the short receives (or pays). There is no signal and no parameter to tune, so this is a measurement of a known structure, one definition per coin.

## 1. Hypotheses

**H1**: the hedged position earned at least 3 points a year more than idle dollars (so at least 7%) on total capital, net of costs, in both 2025 and January-October 2026.

**H0**: it earned about what lending dollars earns, or less.

## 2. Definition (fixed)

- **ETH**: Lido stETH staking yield (DefiLlama pool `747c1d2a-c668-4682-b9f9-296708a3dd90`, daily APY) with a short in Binance ETHUSDT perp.
- **SOL**: Jito JitoSOL staking yield (DefiLlama pool `0e7d0722-9054-4907-8593-567b353c0900`) with a short in Binance SOLUSDT perp.
- **Daily return on the hedged dollar** = staking APY / 365 + funding received by the short that day + (spot return − perp return), the last term being the day's change in the gap between the perp and spot. Prices and funding come from the Phase 1b cache (Binance archive, to 2026-10-31).
- **Costs**: one round trip a year on both legs at taker fees (spot 0.10%, perp 0.05%, each way): 0.30% a year, charged daily.
- **Capital**: reported two ways. (a) "token as collateral": the staking token itself backs the short, so capital equals the hedged dollar. (b) "separate margin": an extra 25 cents of idle dollars per hedged dollar sits as margin and earns nothing, so returns are divided by 1.25. Version (b) is the primary one; (a) needs an exchange that accepts the token as collateral and is exposed to its haircut.
- **Window**: each calendar year from the first full year both series exist, to 2026-10-31. This is a realised yield measurement on data up to the cache date; it selects nothing.

## 3. Metric and gate

**Primary**: net return on total capital, version (b), in 2025 and in January-October 2026 (annualised).

H1 is supported if both are at least 7% and the largest daily mark-to-market drawdown over 2025-26 is at most 5%. Otherwise H0.

Also reported: each component by year (staking, funding, basis change), version (a), worst drawdown, share of days with negative funding, and the same years for idle dollars (4%), Aave USDT and Ethena sUSDe from the earlier survey.

## 4. Decision per outcome

- **H1**: propose a small trial to Alex, with the risks it does not measure listed (the staking token trading below the coin, the exchange, liquidation of the short in a fast rally if margin is not topped up).
- **H0**: report it as "no better than lending" and do not build.

## 5. Limits stated before the run

- The staking token is assumed to track the coin. stETH traded up to 6% below ETH in 2022; that risk is not in the numbers.
- Funding is Binance's. Other venues differ slightly.
- Rebalancing between the legs after large moves is not modelled; with one round trip a year charged, the cost side is roughly right for a position that is left alone.
