# Staking with a hedge on 16 coins, and two mixes: results

**Date**: 2026-11-24
**Preregistration**: `2026-11-24_hedged-staking-wide-prereg.md` (commit `cd2d628`, pushed before any funding, basis or combined figure for the new coins or the mixes was computed).
**Script**: `experiments/strategy_search/yields.py hedged-wide`. Output: `results/hedged_staking_wide.json`.
**Window**: 2025-01-01 to 2026-10-31. Returns are on total capital (25 cents of idle margin per hedged dollar), after costs.

## Verdict

**H0 is supported for both mixes. No build.**

| Mix | 2025 | 2026 (Jan-Oct, annualised) | Whole window | Drawdown | Gate (7% in both years, drawdown at most 5%) |
|---|---|---|---|---|---|
| M1: every eligible coin, equal weight | +2.6% | −4.2% | −1.0% | 7.3% | fail on all three |
| M2: top three by last month's yield, idle dollars if under 4% | +8.5% | +0.2% | +8.6% | 2.8% | fail (2026) |
| Idle dollars | 4.0% | 4.0% | +7.5% | 0 | |

Holding everything lost money. Rotating into last month's best payers earned 8.5% in 2025 and nothing in 2026, which over the whole window is about what idle dollars paid, with 4.3% of capital spent on switching.

## Per coin (descriptive; held throughout, 0.30% a year in costs)

Staking and funding are per hedged dollar per year. Net is on total capital. A liquidation day is a rise of 20% or more inside a day against the short, charged at 1.5%.

| Coin (token) | 2025 staking | 2025 funding | 2025 net | 2026 staking | 2026 funding | 2026 net | Liquidation days | Drawdown |
|---|---|---|---|---|---|---|---|---|
| LINK (stLINK) | 5.2% | +5.1% | **8.2%** | 4.7% | +4.6% | **7.5%** | 0 | 0.2% |
| BNB (slisBNB, from 2025-05-23) | 13.7% | +0.6% | 11.5% | 3.6% | +2.8% | 4.9% | 0 | 0.1% |
| SOL (JitoSOL, from 2025-03-26) | 7.0% | +0.6% | 5.8% | 5.5% | −0.6% | 3.8% | 0 | 0.9% |
| ETH (stETH) | 2.7% | +4.9% | 4.8% | 2.3% | +2.1% | 1.8% | 2 | 1.2% |
| AVAX (sAVAX) | 5.3% | +0.1% | 3.9% | 4.2% | +1.3% | 2.7% | 1 | 1.2% |
| MON (shMON, from 2025-12-02) | too short | | | 13.6% | −5.5% | 4.9% | 1 | 1.4% |
| FLOW (ankrFLOW, from 2026-05-03) | none | | | 7.9% | +9.5% | 13.5% (182 days) | 0 | 0.8% |
| HYPE (kHYPE, from 2026-03-01) | none | | | 1.9% | +4.8% | 1.5% | 2 | 1.2% |
| DOT (vDOT) | 11.6% | +1.1% | 8.7% | 5.3% | −4.6% | −3.1% | 3 | 5.0% |
| ICP (nICP) | 13.2% | +2.2% | 4.4% | 12.2% | −6.7% | −4.0% | 11 | 12.9% |
| TON (hTON) | 3.8% | +4.1% | 3.5% | 9.1% | −6.0% | −11.7% (174 days) | 7 | 6.3% |
| FIL (iFIL, from 2025-07-14) | 10.2% | −0.7% | 2.5% (171 days) | 5.8% | +0.7% | −1.0% | 6 | 4.3% |
| POL (MaticX) | 2.2% | −0.5% | −0.3% | 2.5% | −8.6% | −6.3% | 2 | 6.7% |
| APT (amAPT) | 9.9% | −12.7% | −4.0% | 8.4% | −5.4% | −1.3% | 3 | 10.8% |
| STRK (xSTRK) | 9.6% | −1.2% | −3.7% | 7.0% | +0.6% | −7.2% | 17 | 16.7% |
| BERA (iBERA, from 2025-07-31) | 3.0% | −45.7% | −34.8% (154 days) | 2.9% | −65.6% | −48.0% | 8 | 49.3% |

ETH and SOL read lower here than on 2026-11-23 (ETH 6.1% and 3.4%, SOL 3.7%) because this version charges the liquidation days and takes the 7-day median of the published yield. The earlier figures stand as that file defined them.

## Reading it

- **High staking yields are paid for in funding.** The coins with the largest staking yields are the ones whose shorts pay the most: APT stakes at 10% and its short paid 13% in 2025; ICP stakes at 12% and its short paid 7% in 2026; BERA's short paid 46% to 66% a year. The market prices the hedge at roughly what the staking pays, and sometimes far more.
- **The falls made it worse in 2026.** In a year when most of these coins dropped heavily, shorts were crowded and paid to stay short. Funding turned against the hedge on DOT, ICP, TON, POL and MON.
- **Small coins squeeze.** STRK had 17 days with a rise of 20% or more inside the day, ICP 11, BERA 8. Each one costs a liquidation or a scramble for margin.
- **One coin cleared 7% in both years: LINK** (8.2% and 7.5%, no liquidation days, drawdown 0.2%). By the preregistered rule this is not a pass, because it is one of sixteen seen in hindsight, and its funding half (about 5% a year received) is the part that has vanished on ETH. It is the only figure in this table that would be worth watching forward.
- **Rotation chases what has just ended.** M2's holdings changed at 18 of the 21 month ends. Last month's best payer was often paying because of a short-lived funding spike.

## Decision (as preregistered)

H0 for both mixes: no build, and no further mixes of these series on this window.

## Limits (unchanged from the preregistration)

Staking tokens are assumed to track their coins; costs of using several chains are not counted and would exceed the yield on an account of about 85 dollars; the published yields are the protocols' own; several series have gaps or flat stretches (DOT, TON, APT and ICP are missing weeks in 2026, during which the coin is simply not counted).
