# Dollar yields on-chain, 2025-26: results

**Date**: 2026-11-24
**Preregistration**: `2026-11-24_dollar-yield-prereg.md` (commit `5c384bd`, pushed before any yield or mix was computed).
**Script**: `experiments/strategy_search/yields.py dollar`. Output: `results/dollar_yields.json`. Two runs, identical output.
**Window**: 2025-01-01 to 2026-10-31. 200 stablecoin pools from DefiLlama, each counted only while it held at least 50 million dollars.

**Every figure here is an upper bound.** The pool list is today's survivors, and the published yield says nothing about lost principal.

## Verdict

**H0 is supported for all five rules. No build.** No rule reached 7% in both years. The nearest, rotating into the three highest-yielding pools of any kind, made 6.6% and 8.2%, and it got there by holding new synthetic-dollar tokens.

| Rule (monthly, after 0.10% switching cost) | 2025 | 2026 (Jan-Oct, annualised) | Including reward tokens | Gate: 7% in both years |
|---|---|---|---|---|
| D1: every large pool, equal weight | 3.1% | 2.5% | 4.5% / 3.3% | fail |
| D2: lending pools, top three by last month's yield | 5.2% | 5.4% | 6.6% / 5.2% | fail |
| D3: any pool, top three by last month's yield | 6.6% | 8.2% | 8.8% / 8.0% | fail (2025) |
| D4: stablecoin exchange pools, equal weight | 0.0% | −0.2% | 2.2% / 2.2% | fail |
| D5: yield-bearing dollar tokens, equal weight | 3.3% | 3.9% | 6.3% / 4.4% | fail |
| Idle dollars | 4.0% | 4.0% | | |

## Where a dollar could have sat (pools large for at least 300 days; base yield)

| Pool | Kind | 2025 | 2026 |
|---|---|---|---|
| f(x) Protocol fxUSD stability pool | synthetic dollar | 9.9% | 6.8% |
| Maple USDC | lending to trading firms | 6.5% | 4.9% |
| sDAI on Gnosis | savings token | 6.8% | 4.4% |
| Ethena sUSDe | basis-trading token | 6.6% | 4.2% |
| Fluid USDC | lending | 4.3% | 4.2% |
| Sky sUSDS | savings token | 4.8% | 3.7% |
| Morpho Steakhouse USDC | lending vault | 4.6% | 3.6% |
| Aave v3 USDC | lending | 4.2% | 3.3% |
| Aave v3 USDT | lending | 3.9% | 2.9% |
| Curve DAI-USDC-USDT | exchange pool | 0.2% | 0.0% |

Newer pools with a 2026 figure only: Cap stcUSD 5.7%, Falcon sUSDf 5.5%, Reservoir wsrUSD 5.2%. The full table of 42 pools is in the output file.

## Reading it

- **The large, established places pay 3 to 4%.** Aave, Sky, Morpho's main vaults and Fluid all sit at or just under what idle dollars earn on an exchange. The average large pool paid less than idle dollars in both years.
- **Yields fell from 2025 to 2026 almost everywhere**, by one to two and a half points. This is the same fading of demand for leverage that removed the funding premium from every carry trade tested earlier: lending rates are what leveraged traders pay to borrow dollars.
- **Getting to 7 or 8% means holding the newest tokens.** In 2026 the any-pool rule held, in turn, Falcon sUSDf, Cap stcUSD, the fxUSD stability pool, USD.AI, infiniFi, Mainstreet, Avant, Apyx, Saturn and Accountable. Most did not exist two years ago. The rule changed holdings at 18 of 21 month ends. A yield two to four points above Aave is the price of a risk, and the list used here cannot contain the tokens where that risk came due.
- **Stablecoin exchange pools pay almost nothing** before reward tokens (0 to 0.2%).
- **Reward tokens add one to two points** on paper for the average pool. They have to be claimed and sold, which is not worth the fees for a small account.

## What it means

There is no dollar yield here that both clears 7% and can be called safe. The honest range is about 4% for established lending, 5 to 6.5% for the better-known savings and credit tokens (Maple, sUSDe in a good year), and 7 to 8% for a rotating basket of new synthetic dollars with unmeasured risk to principal. On an account of about 85 dollars, the difference between the best and the simplest of these is two or three dollars a year, and one Ethereum transaction can cost more than that.

## Decision (as preregistered)

H0 for all: no build. D3 did not pass, so no forward record is started for it.

## Limits

As preregistered: survivors only (16 of today's pools were large in January 2025 against 91 in October 2026); principal risk not measured; fixed-rate tokens not measurable from this source; costs assume an account large enough for fees to be small.
