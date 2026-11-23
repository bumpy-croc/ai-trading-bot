# Resting orders that only fill in sudden spikes, on ETH, SOL and XRP: results

**Date**: 2026-11-23
**Preregistration**: `2026-11-23_spike-fade-prereg.md` (commit `c6dd8c0`, pushed before any of the test data was simulated).
**Scripts**: `experiments/strategy_search/mm_data.py`, `spike_sim.py`. Output: `results/spike_<set>.json`, `results/spike_<set>_trades.csv`, `results/spike_pooled.json`.
**The run**: every aggregated trade on Binance ETHUSDT, SOLUSDT and XRPUSDT perps from 2025-11-01 to 2026-10-31 (about 1.3 billion trades), and BTCUSDT from 2024-11-01 to 2025-10-31 as a control. Each set was run twice with identical output.

## Verdict

**By the preregistered gates: H0, no build. Two of five gates fail.** The pooled t-statistic is 2.03 against the 2.5 required, and XRP produced 60 trades against the 100 required.

It is still the first result in this search that is positive after costs in every coin tested: +12.5 bp per trade over 301 trades, and 12.5% a year on average per coin account at 1x. That is not enough evidence to act on, and it is enough to justify one replication on data this test did not touch (`2026-11-23_spike-fade-replication-prereg.md`).

## Gates (three candidates pooled)

| Gate | Requirement | Result | |
|---|---|---|---|
| 1. Pooled net per trade | above zero, t at least 2.5 by day | **+12.5 bp, t = 2.03** | **FAIL** |
| 2. Each coin | net positive | ETH +5.6 bp, SOL +2.4 bp, XRP +45.8 bp | pass |
| 3. Return | average at least 10% a year per coin account | 12.5% (ETH 7.6%, SOL 2.5%, XRP 27.5%) | pass |
| 4. Risk | worst trade no worse than −5%, drawdown at most 15% | worst −3.4%, largest drawdown 7.6% | pass |
| 5. Sample | at least 100 trades in each coin | ETH 136, SOL 105, **XRP 60** | **FAIL** |

## By set (the preregistered rule: orders 250 to 350 ms old, exit after 300 s, 8 bp round-trip cost)

| Set | Distance | Trades | Before costs, bp per trade | After costs, bp per trade | t (after costs, by day) | Net a year | Worst trade | Max drawdown | First / second half, bp | Longs / shorts, bp |
|---|---|---|---|---|---|---|---|---|---|---|
| ETH 2025-11 to 2026-10 | 43 bp | 136 | +13.6 | +5.6 | 0.93 | +7.6% | −2.2% | 7.6% | +9.9 / +0.4 | +17.2 / −7.9 |
| SOL 2025-11 to 2026-10 | 50 bp | 105 | +10.4 | +2.4 | 0.26 | +2.5% | −3.4% | 6.8% | +1.9 / +2.8 | +3.8 / +0.6 |
| XRP 2025-11 to 2026-10 | 55 bp | 60 | +53.8 | +45.8 | 2.17 | +27.5% | −2.7% | 2.1% | +32.1 / +54.9 | +46.6 / +44.3 |
| **Pooled candidates** | | 301 | +20.5 | **+12.5** | **2.03** | 12.5% avg | | | | |
| Control: BTC 2024-11 to 2025-10 | 25 bp | 278 | +8.2 | +0.2 | 0.05 | +0.5% | −3.0% | 10.5% | +1.9 / −7.5 | +3.3 / −3.1 |

What the table says:

- **The before-cost effect is there in all four sets**, including the control in a different year (+8.2 bp, t = 2.4 by day). Prices do move back after a spike through a far, fresh order. That part of the exploration held up.
- **After costs it is thin everywhere except XRP.** On BTC it is zero, as the exploration predicted. On ETH and SOL it is a few basis points with t below 1.
- **XRP carries the pooled result, on 60 trades.** One day, 22 August 2026, gave five XRP trades worth +10.5% between them, 38% of XRP's year. Without the best and worst 5% of its trades XRP still averages +44 bp, and its median trade is +33 bp, so it is not one day's luck alone. It is still a small sample.
- The control's worst day was 10 October 2025 (twelve trades, −3.4% in total). The rule survived the largest liquidation event of recent years without a large loss, because each position is small in time and the account takes one at a time.

## Sensitivities (after costs, bp per trade; trades in brackets)

| Set | Rule | Orders 50 to 150 ms old | Orders 1 to 1.1 s old | Exit after 60 s | Exit after 30 min | Rule with a maker exit (4 bp cost) |
|---|---|---|---|---|---|---|
| ETH | +5.6 (136) | +16.4 (99) | +0.4 (274) | +2.7 (156) | +3.1 (126) | +9.6 |
| SOL | +2.4 (105) | −1.0 (90) | +0.9 (152) | −4.0 (118) | +9.7 (97) | +6.4 |
| XRP | +45.8 (60) | +43.9 (42) | +17.9 (83) | +38.7 (69) | +33.2 (54) | +49.8 |
| BTC control | +0.2 (278) | +1.5 (240) | −1.0 (370) | +0.0 (329) | −0.1 (221) | +4.2 |

Older orders do worse in three of four sets, as in the exploration. Nothing here is used to change the rule.

## Fresh slice, 1 to 21 November 2026

ETH 17 trades, −1.5 bp each. SOL 2 trades, +59.5 bp each. XRP 2 trades, −1.3 bp each. Too few to say anything.

## Limits

- 301 trades. The exit at the last trade price plus 1 bp is the weakest assumption: five minutes after a spike the gap between buyers and sellers can be several basis points on XRP.
- Replacing both orders ten times a second all year is at the edge of the exchange's order-rate limit, and the simulated orders are never refused or delayed.
- The whole-account position size is for measurement. At 1x the best coin made 27.5% in the year; the dollar amounts on a small account are small.

## Decision

Per the preregistration: **no build.** Because the result is positive in every coin and misses on evidence, not on sign, the same rule is run once more, unchanged, on six sets it has not seen: ETH, SOL and XRP in the previous year, and DOGE, 1000PEPE and SUI in the same year. That test has its own preregistration and its own gates, and is judged on its own data alone.
