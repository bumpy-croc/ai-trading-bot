# Spike fade: replication on six untouched sets. Results

**Date**: 2026-11-23
**Preregistration**: `2026-11-23_spike-fade-replication-prereg.md` (commit `c8d3bd3`, pushed before any of the six sets was simulated).
**Scripts**: `experiments/strategy_search/mm_data.py`, `spike_sim.py`. Output: `results/spike_<set>.json`, `results/spike_<set>_trades.csv`, `results/spike_pooled_replication.json`.
**The run**: the unchanged rule on ETHUSDT, SOLUSDT and XRPUSDT perps from 2024-11-01 to 2025-10-31, and on DOGEUSDT, 1000PEPEUSDT and SUIUSDT from 2025-11-01 to 2026-10-31. Each set was run twice with identical output.

## Verdict

**H0. The replication fails gates 1, 3 and 4. The idea is closed on historical data, as the preregistration fixed for this outcome.**

Pooled over the six sets the rule lost 2.2 bp per trade after costs (3,422 trades, t = −0.94). Five of six sets were positive, but one set lost twice its account and the year that contained real liquidation events showed what the first test's year did not: the rule earns a little on ordinary spikes and gives back a lot when a spike is the start of a collapse.

## Gates (six sets pooled, on their own)

| Gate | Requirement | Result | |
|---|---|---|---|
| 1. Pooled net per trade | above zero, t at least 2.0 by day | **−2.2 bp, t = −0.94** | **FAIL** |
| 2. Sets net positive | at least 4 of 6 | 5 of 6 | pass |
| 3. Return | average at least 10% a year per coin account | **−12.8%** | **FAIL** |
| 4. Risk | worst trade no worse than −5%, drawdown at most 15%, in every set | worst trade −16.0%; drawdowns of 27% and 207% | **FAIL** |
| 5. Sample | at least 300 trades pooled | 3,422 | pass |

## By set (after 8 bp costs unless stated)

| Set | Distance | Trades | Before costs, bp | After costs, bp | t (by day) | Net a year | Worst trade | Max drawdown | First / second half, bp | Longs / shorts, bp |
|---|---|---|---|---|---|---|---|---|---|---|
| ETH 2024-11 to 2025-10 | 30 bp | 476 | +10.8 | +2.8 | 0.79 | +13.5% | −8.0% | 14.4% | +5.4 / −2.0 | +10.7 / −8.4 |
| SOL 2024-11 to 2025-10 | 44 bp | 328 | +8.8 | +0.8 | 0.11 | +2.7% | −10.9% | 26.9% | −5.4 / +29.9 | +15.8 / −17.4 |
| XRP 2024-11 to 2025-10 | 30 bp | 2,006 | −2.3 | **−10.3** | −3.12 | **−205.7%** | −16.0% | 206.7% | −9.8 / −13.6 | −1.2 / −21.2 |
| DOGE 2025-11 to 2026-10 | 59 bp | 129 | +29.4 | +21.4 | 2.43 | +27.6% | −3.9% | 4.5% | +27.4 / +14.8 | +25.0 / +12.6 |
| 1000PEPE 2025-11 to 2026-10 | 70 bp | 236 | +25.8 | +17.8 | 2.17 | +42.1% | −10.3% | 9.4% | +30.0 / +10.2 | +29.2 / −2.4 |
| SUI 2025-11 to 2026-10 | 61 bp | 247 | +25.5 | +17.5 | 2.52 | +43.3% | −7.3% | 8.6% | +17.9 / +17.2 | +21.4 / +10.0 |
| **Pooled** | | 3,422 | +5.8 | **−2.2** | **−0.94** | −12.8% avg | | | | |

## What happened

**The two years tell different stories.**

- **2025-11 to 2026-10** (this replication's three new coins, and the first test's three): all six coins positive after costs, four of them with t above 2. The new coins alone made +18.5 bp per trade over 612 trades.
- **2024-11 to 2025-10** (ETH, SOL, XRP here, and the BTC control from the first test): about zero on BTC, ETH and SOL, and a heavy loss on XRP. Those three sets together lost 6.7 bp per trade over 2,810 trades.

**Why the earlier year failed.**

1. **Real cascades.** On 10 October 2025 the rule lost 53.5% of the XRP account in one day (25 trades) and 5.7% of the ETH account (17 trades). On 2 March 2025 it lost 15.4% of the SOL account (14 trades). When a spike is the first leg of a collapse, buying it and holding five minutes buys the next leg, and the rule re-enters as soon as each position closes. The first test's year had no day like these.
2. **A distance that did not fit the coin.** XRP's distance was 30 bp, set from a year (2023-11 to 2024-10) in which XRP was quiet. It then quadrupled in November 2024 and 30 bp became an ordinary move: 678 trades in that month alone, 1,117 in the first two months, against 60 in the whole of the following year at 55 bp. Fills at that distance were noise, not spikes, and each paid 8 bp in costs. This is a failure of the rule as written, which fixed the distance for a year in advance.

**What survives.** Before costs, the move back after a spike showed up in nine of the ten sets across both tests (every set except XRP 2024-25). After costs it paid in one year and in the more volatile coins, and the losses when it failed were several times larger than the gains when it worked. That is the shape of selling insurance: steady small income and occasional large claims.

## Sensitivities (after costs, bp per trade; trades in brackets)

| Set | Rule | Orders 50 to 150 ms old | Orders 1 to 1.1 s old | Exit after 60 s | Exit after 30 min |
|---|---|---|---|---|---|
| ETH 2024-25 | +2.8 (476) | +3.2 (408) | −5.0 (816) | −3.3 (590) | −10.2 (392) |
| SOL 2024-25 | +0.8 (328) | +6.3 (277) | +1.3 (442) | +7.9 (415) | −1.7 (264) |
| XRP 2024-25 | −10.3 (2,006) | −7.4 (1,732) | −8.1 (2,521) | −8.5 (2,792) | +1.2 (1,254) |
| DOGE 2025-26 | +21.4 (129) | +19.7 (104) | +10.8 (185) | −1.9 (140) | +18.9 (121) |
| 1000PEPE 2025-26 | +17.8 (236) | +24.3 (182) | +8.5 (320) | +15.0 (281) | +11.9 (201) |
| SUI 2025-26 | +17.5 (247) | +25.9 (206) | +10.8 (302) | +8.2 (269) | +31.0 (226) |

Fresh slice, 1 to 21 November 2026 (new coins only): DOGE 2 trades at −127 bp each, 1000PEPE 32 trades at +21 bp, SUI 3 trades at +24 bp.

## Both tests together

| | Sets | Trades | After costs, bp per trade |
|---|---|---|---|
| First test, candidates (2025-26) | 3 | 301 | +12.5 |
| First test, control (BTC 2024-25) | 1 | 278 | +0.2 |
| Replication, new coins (2025-26) | 3 | 612 | +18.5 |
| Replication, earlier year (2024-25) | 3 | 2,810 | −6.7 |

The first result was not luck in the narrow sense: the same year, on three coins it had not seen, the rule did as well or better. It was a good year for the rule. The year before was not, and a rule that needs the right year is not something to build on.

## Decision

Per the preregistration: **H0. The idea is closed on historical data. No further variants of it are to be run on past data.**

Not tested and not claimed: a version whose distance follows current volatility and which stops for the day after a loss would, with hindsight, have avoided much of the damage. Hindsight is exactly what the preregistration rules out. The only honest test of such a version is on trades that have not happened yet, with the rule fixed beforehand, and that needs Alex's decision.
