# Spike fade: replication on six untouched sets. Preregistration

**Date**: 2026-11-23
**Status**: PREREGISTERED. Written and pushed before any of the six sets below has been simulated.
**Authority**: Alex, 2026-11-23, project thread: "merge the pr and carry on finding a profitable strategy".
**Scripts**: `experiments/strategy_search/mm_data.py`, `spike_sim.py` (`run --set <name>`, then `report --group replication`).

## 0. Why

The first test (`2026-11-23_spike-fade-results.md`) was positive after costs in all three coins and missed its gates on evidence: pooled t = 2.03 against 2.5, and too few trades in XRP, which carried most of the profit. A result like that is either a small real effect or luck concentrated in one coin. More independent data separates the two. Nothing about the rule changes.

## 1. Hypotheses

**H1**: the unchanged rule earns more than its costs on data it has not seen.

**H0**: the first result was luck, concentrated in XRP's 60 trades, and the rule does not pay on new data.

## 2. Data (none of it simulated before this file was pushed)

- **Earlier year, same coins**: ETHUSDT, SOLUSDT, XRPUSDT, 2024-11-01 to 2025-10-31. This year contains the 10 October 2025 liquidation event.
- **Same year, next coins**: the next three most traded perps by median daily volume from May to October 2025 that had been listed for at least a year at that point (which leaves out ASTER and COAI): **DOGEUSDT, 1000PEPEUSDT, SUIUSDT**, 2025-11-01 to 2026-10-31.

## 3. Rule

Exactly the rule of `2026-11-23_spike-fade-prereg.md`, section 3: orders refreshed every 100 ms and live 250 ms later, fill only on a trade through the order, one position at a time, exit after 300 seconds at the last trade price with 1 bp of slippage, 8 bp round-trip cost, whole account at 1x.

Distance by the same formula, 25 bp × the coin's daily volatility relative to BTC over the twelve months before its test window:

| Set | Volatility ratio (window used) | Distance |
|---|---|---|
| ETH 2024-11 to 2025-10 | 1.198 (2023-11 to 2024-10) | 30 bp |
| SOL 2024-11 to 2025-10 | 1.773 | 44 bp |
| XRP 2024-11 to 2025-10 | 1.219 | 30 bp |
| DOGE 2025-11 to 2026-10 | 2.348 (2024-11 to 2025-10) | 59 bp |
| 1000PEPE 2025-11 to 2026-10 | 2.814 | 70 bp |
| SUI 2025-11 to 2026-10 | 2.451 | 61 bp |

## 4. Gates (on these six sets alone, all required)

1. Pooled mean net return per trade above zero, with t at least 2.0 on daily sums. The bar is 2.0 because this is one rule fixed in advance, tested on independent data.
2. Net positive in at least four of the six sets.
3. Average net return per coin account at least 10% a year.
4. In every set: worst single trade no worse than −5%, maximum drawdown at most 15%.
5. At least 300 trades pooled.

The first test's trades are not pooled into these gates.

## 5. Decision per outcome

- **All five pass**: propose to Alex a forward shadow test on live data (orders computed and logged in real time against the live trade feed, no money), which is the only way to check the fills and the exit price. No live trading.
- **Gate 1 or 2 fails**: H0. The first result is recorded as luck and the idea is closed. No further variants of it on historical data.
- **Gates 1 and 2 pass, another fails**: reported as mixed; no build without Alex's decision.

## 6. Limits stated before the run

- The same limits as the first test: no order book, an optimistic exit, an order rate at the edge of the exchange's limit.
- DOGE and SUI trade at about 8,000 to 9,000 ticks, so one tick is more than 1 bp. Orders are rounded away from the price, which the simulator does.
