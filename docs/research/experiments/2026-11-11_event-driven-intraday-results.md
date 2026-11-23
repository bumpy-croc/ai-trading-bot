# Event-driven intraday trades (Phase 1d): results

**Date**: 2026-11-11
**Prereg**: [2026-11-11_event-driven-intraday-prereg.md](2026-11-11_event-driven-intraday-prereg.md) (three ideas, one configuration each, gates unchanged).
**Scripts**: `experiments/strategy_search/events.py` (`download`, `run`), `yields.py`. Raw numbers: `experiments/strategy_search/results/events.json`, `events_E*_trades.csv`, `yields.json`.

## Verdict

**All three ideas FAIL. H0 is supported for each.** None has a positive net return per trade; each fails four of the five gates.

| Gate | E1 funding-settlement capture | E2 buy capitulation | E3 follow a volume breakout |
|---|---|---|---|
| G1 ≥ 300 trades | pass (2,023) | pass (5,828) | pass (6,383) |
| G2 mean net > 0, clustered t ≥ 3 | **FAIL** (-143 bp, t -2.2) | **FAIL** (-16 bp, t -0.6) | **FAIL** (-52 bp, t -4.5) |
| G3 positive in each sub-period | **FAIL** (-279 / -30 / -119 bp) | **FAIL** (+34 / -23 / -73 bp) | **FAIL** (-68 / -36 / -45 bp) |
| G4 portfolio CAGR ≥ 10%, DD ≤ 20% | **FAIL** (-24%/yr, 83% DD) | **FAIL** (-11%/yr, 77% DD) | **FAIL** (-40%/yr, 97% DD) |
| G5 positive with slippage doubled again | **FAIL** (-174 bp) | **FAIL** (-66 bp) | **FAIL** (-103 bp) |

Sub-periods are 2020-22 / 2023-24 / 2025-26. Two consecutive runs produced byte-identical `events.json`. Hourly data: 408 symbols, 6,239 symbol-months (4 not in the archive).

## What each trade actually did

Per trade, in basis points (100 bp = 1%):

| | Price move | Funding | Cost | Net | Win rate | Worst trade |
|---|---|---|---|---|---|---|
| E1 | -172 | +71 | 41 | **-143** | 42% | -75% |
| E2 | +43 | +3 | 61 | **-16** | 46% | -66% |
| E3 | +9 | +1 | 61 | **-52** | 39% | -66% |

**E1, funding-settlement capture.** The payment is real (71 bp on average) and the price takes back more than twice that in the two hours around it. 1,904 of the 2,023 trades are longs in coins with deeply negative funding, which means coins that are being shorted hard, and they keep falling: the three worst trades are longs that lost 66-76% in two hours (ARIA, LIGHT, LAB, all 2026) while collecting 1-2% of funding. Extreme funding is not free money left on the table; it is the market's price for standing in front of a collapse. The opposite trade (paying the funding and going with the crowd) earns 61 bp a trade after costs, but with a t-statistic of 1.0, so it is not evidence of anything.

**E2, buy capitulation.** The rebound exists: +43 bp over four hours before costs, across 5,828 events. It is smaller than the cost of trading it in a disorderly market (61 bp with the doubled slippage the prereg set). It has also decayed: net +34 bp in 2020-22, -23 bp in 2023-24, -73 bp in 2025-26.

**E3, follow a volume breakout.** Nothing there. The average move after an extreme up-hour is +9 bp, so both following it and fading it lose the costs. This was the idea flagged as contaminated by the Phase 1b holdout; it fails anyway.

## Secondaries (reported, not decisive)

| | E2 | E3 |
|---|---|---|
| Hedged with a BTC short | -40 bp (t -2.1) | -84 bp (t -9.8) |
| Mirror (short instead of long) | -106 bp (t -4.1) | -71 bp (t -5.8) |
| 1-hour hold | -34 bp | -56 bp |
| 2-hour hold | -23 bp | -57 bp |
| 8-hour hold | -35 bp | -40 bp |
| 24-hour hold | +51 bp (t 1.2) | -19 bp |

By liquidity tier, net per trade: E1 -63 / -115 / -172 bp (rank 1-10 / 11-30 / 31+); E2 **+50** / -7 / -61 bp; E3 -24 / -30 / -89 bp.

### Post-hoc look at the one positive cell. Exploratory; cannot change the verdict.

E2 in the ten most liquid coins is the only cell above zero, so it will be asked about. Net per trade there: +78 bp in 2020-22 (557 trades, t 1.9), +27 bp in 2023-24 (350, t 0.9), +24 bp in 2025-26 (296, t 0.4); overall t 1.9. By year it swings between +224 bp (2021) and -66 bp (2025), with three losing years out of seven. That is a positive average with no statistical support and a declining trend. It was picked after seeing the results, from three tiers of three ideas, so even its t of 1.9 is flattered. It is not something to build on; at most it is a hypothesis for a forward paper test with a preregistered bar.

## Descriptive section: what passive dollar yields actually paid

Realised track records, not backtests. No gate applies.

| Source | 2023 | 2024 | 2025 | 2026 (to Oct) | Note |
|---|---|---|---|---|---|
| Hyperliquid HLP vault | +13.2% (from June) | +62.9% | +17.3% | +7.9% | 28% a year since June 2023; worst two-week interval -5.2%; deepest drawdown 5.9% at two-week sampling |
| Aave v3 USDT lending (Ethereum) | 4.1% | 6.3% | 4.0% | 3.0% | 3.6% average over 2025-01 → 2026-10 |
| Aave v3 USDC lending (Ethereum) | 3.8% | 7.2% | 4.1% | 3.4% | |
| Ethena sUSDe | | 17.5% (from Feb) | 6.5% | 4.2% | Funding carry, packaged; decayed exactly as Phases 1-1c found |

How the HLP figure was computed: return per interval = P&L earned in the interval / average capital in it (modified Dietz), from the vault's own account-value and cumulative-P&L history, 95 samples about 14 days apart. Dividing by the starting value alone would give 35% a year, which overstates it because the vault was growing. The two-week sampling hides any drawdown that happened and recovered inside an interval, so the 5.9% figure is a floor, not a measurement.

What this says:

- **The 4%/yr idle-USDT hurdle used in Phases 1b-1d was about right** (3.6-4.0% realised on Aave in 2025-26).
- **The one thing in this whole programme that made money in 2025-26 is a market-making vault**, not a directional or carry strategy. It earned in every calendar year, including the two in which every strategy tested here lost. Its return is falling (63% → 17% → about 10% annualised) as the vault and its competition grew.
- **It is not a result of this bot and it is not risk-free.** HLP is the counterparty to every trader and liquidation on the venue; it can lose (11.6% of two-week intervals were negative). It carries smart-contract and bridge risk, a withdrawal lock-up, and the venue's own operational and legal risk, none of which appears in a return series. This file records what it paid. It does not recommend depositing.

## What Phases 1 to 1d establish together

28 preregistered crypto candidates across four experiments (#1315, #1324, #1326 and this one), plus 15 non-crypto candidates in #1325:

1. **Predicting direction** (one coin, any timeframe, any model): no edge after costs.
2. **Risk premia** (carry of four kinds, trend, momentum, low-vol, alt-bleed): paid in 2020-24, paid nothing or lost in 2025-26.
3. **Event trades at hourly resolution**: the effects are either absent (breakouts), priced in more than fully (funding settlements), or real but smaller than costs and decaying (capitulation rebounds).
4. **What did earn**: being the market maker (HLP), and lending dollars. Neither is something a taker-fee bot on hourly bars can replicate.

The common cause in 1-3 is costs relative to edge. Every gross effect found is in the range of 10-50 bp per trade, and a retail taker pays 30-60 bp per round trip in the coins where the effects live.

## Decision implied by the prereg

All fail: no build. The descriptive section stands as the only positive finding.
