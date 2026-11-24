# Dollar yields on-chain, 2025-26: preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. Rules fixed and pushed before any yield, average or mix has been computed. The pool histories were downloaded first and only their sizes were looked at (to count how many pools were large on each date). Corrections go in new dated sections.
**Authority**: Alex, 2026-11-24 11:28 UTC, project thread: chose "Test dollar-yield ideas next" on the decision card "Choose where the search for a profitable strategy goes next?" (the option: lending rotation, fixed-rate tokens and stablecoin pools; realistic best case 5 to 10% a year; yields, not trading-bot strategies).
**Script**: `experiments/strategy_search/yields.py dollar` (output `results/dollar_yields.json`; universe in `results/dollar_yield_universe.json`).

## 0. What is measured

What a dollar earned in on-chain pools that take no price risk on paper: lending markets, savings and yield-bearing dollar tokens, and stablecoin-to-stablecoin exchange pools. First as a plain average, then with simple rules that move to wherever the yield has been highest. There is no price forecast anywhere in this.

## 1. Hypotheses

**H1**: a simple rule over large dollar pools earned at least 7% a year after switching costs in both 2025 and January-October 2026, against 4% for idle dollars.

**H0**: large dollar pools pay about what idle dollars pay (3 to 5%); the pools paying much more are small, short-lived or carrying a risk the yield is paying for; and chasing last month's best yield does not keep it.

## 2. Data

- **Universe**: every pool DefiLlama flags as a stablecoin pool with no impermanent-loss risk and at least 20 million dollars locked on 2026-11-23: 200 pools (118 lending, 17 real-world-asset tokens, 16 exchange pools, 11 yield tokens, 8 savings/CDP, 8 basis-trading tokens such as sUSDe, and others). The list is committed.
- **History**: each pool's daily published yield and value locked (`yields.llama.fi/chart/<pool>`).
- **Window**: 2025-01-01 to 2026-10-31. This is a measurement of realised yields; it is counted with the other uses of that window.
- **Survivors only, and this matters.** The list is today's pools. A pool that was large in 2025 and has since been drained, hacked or wound down is not in it. Only 16 of today's pools were above 50 million dollars in January 2025, against 91 in October 2026. Every figure below is therefore an upper bound: it is the yield of the pools that turned out fine.

## 3. Definitions (fixed)

- **Yield for a day**: the pool's base yield (`apyBase`; where DefiLlama gives none, total yield minus reward-token yield), as a 7-day trailing median limited to between 0% and 25%. Reward tokens are left out of the primary figure because they have to be claimed and sold; the total including them is reported beside it.
- **Eligible on the first of a month**: at least 25 days of yield history in the previous 30, and at least 50 million dollars locked the day before.
- **Daily return** = the day's yield converted from an annual rate. Principal is assumed safe (see section 6).
- **Switching cost**: 0.10% of every dollar moved between pools, including the first entry.
- **Candidates**, rebalanced on the first of each month:
  - **D1**: every eligible pool, equal weight. What large dollar pools paid on average.
  - **D2**: lending pools only (lending, savings/CDP, uncollateralised lending), the three with the highest average yield over the previous 30 days.
  - **D3**: any eligible pool, the three with the highest average yield over the previous 30 days.
  - **D4**: stablecoin exchange pools, equal weight.
  - **D5**: yield-bearing dollar tokens (basis-trading, real-world-asset, yield and allocator categories), equal weight.
  - A candidate with no eligible pool in a month holds idle dollars at 4%.

## 4. Metric and gate

**Primary**: net return in 2025 and in January-October 2026 (annualised), base yield, after switching costs.

H1 is supported for a candidate if both are at least 7%. Otherwise H0.

Reported: the same including reward tokens, what each rule held month by month, switching costs, and a table of every pool that was large for at least 300 days with its average yield in each year.

## 5. Decision per outcome

- **H1 for a candidate**: because the figure is an upper bound, a pass leads only to a forward paper record of that rule's monthly picks with the tokens' market prices recorded beside the yields, so that a loss of principal shows up. A proposal to Alex would follow that, with the practical side worked out (which chains, what moving a small account costs, where he can legally use them).
- **H0 for all**: reported with the table, including the single best place a dollar could have sat. No build.

The bot stays paused in every case.

## 6. Limits, stated before the run

- **Principal risk is not in the numbers.** A dollar token can lose its peg, a lending market can take bad debt, a contract can be drained. The published yield does not show any of that, and the survivor-only list removes the cases where it happened.
- **Fixed-rate tokens** (Pendle and similar) cannot be backtested from this source, because matured markets leave the list. Their current rates are simply whatever appears among the pools above.
- **Costs for a small account.** On Ethereum, where 120 of the 200 pools are, one deposit or withdrawal can cost more than a year of yield on 85 dollars. The 0.10% switching cost assumes an account large enough for fees to be small.
- **Published yields** are the protocols' own figures as collected by DefiLlama.
