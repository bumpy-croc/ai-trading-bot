# Cross-venue funding arbitrage, Binance vs Hyperliquid (Phase 1c): preregistration

**Date**: 2026-11-10
**Status**: PREREGISTERED. One configuration, locked before any Hyperliquid funding data is downloaded or any result computed. Corrections go in new dated sections, never edits.
**Follows**: `2026-11-10_strategy-search-results.md` (#1324), where all eight families on the Binance price panel failed. That file's decision branch says further crypto work needs a mechanism not on its list and a clean test. This is one.
**Scripts**: `experiments/strategy_search/xvenue.py`.

## 0. Why this, and why it is different

Every strategy in Phase 1b took price risk between coins: it was long some coins and short others, and one squeeze (MYX) was enough to end it. This trade holds **the same coin on two exchanges**: long the perpetual where funding is cheap, short it where funding is expensive. A 10x move in the coin is a wash. What is left is the difference in funding, the basis between venues, costs, and the operational risk of running margin on two venues.

Why a gap should exist: Hyperliquid is an on-chain venue whose users are mostly leveraged longs, funding is paid hourly and is far less tightly capped than Binance's, and moving capital between an on-chain venue and a centralised exchange is slow enough that the gap is not instantly closed.

## 1. Hypotheses

**H1**: funding on Hyperliquid and Binance for the same coin differs persistently enough that a rule which enters on a wide trailing gap and exits when it closes earns more than twice idle USDT (8%/yr on total capital) after costs, with a drawdown under 10%, in 2025-26.
- *Supported if*: the single configuration below clears the holdout gate in section 4.

**H0**: the gap is too small, too short-lived, or too costly to harvest at retail fees.
- *Supported if*: the configuration fails any holdout gate, or the sanity check on the fit window fails (then the holdout is not read).

## 2. Data

- **Hyperliquid** public API: `meta` (all perps, including the 56 marked delisted), daily candles, hourly funding history. History starts 2023-05-12.
- **Binance**: the Phase 1b cache (all USDT perps, daily klines, funding).
- Pairing: Hyperliquid coin `X` ↔ Binance `XUSDT`; Hyperliquid `kX` ↔ Binance `1000XUSDT`. Unpaired coins are dropped. Binance non-crypto perps stay excluded.
- Daily funding on each venue = sum of that venue's prints settled in the UTC day (same convention as Phase 1b). Prices = each venue's daily close at 00:00 UTC.
- To keep the download tractable, hourly funding is fetched only for coins that are ever eligible under the liquidity rule below. Eligibility is computed from candles alone, before any funding is seen.

## 3. The rule (one configuration, no variants in the gate)

- **Eligible on day t**: at least 30 daily closes on both venues; in the top 30 Hyperliquid perps by trailing 30-day median dollar volume; Binance liquidity rank ≤ 100.
- **Signal**: m = mean over the last 7 days of (Hyperliquid daily funding − Binance daily funding), known at day t's close.
- **Slots**: 5, each one fifth of capital, split evenly between the two venues, **each leg at 1x**. Hedged notional per slot = 10% of capital per leg (total 50% of capital per side when full).
- **Entry** (reviewed daily, executed one day later at that day's close): a free slot takes the eligible coin with the largest |m|, if |m| ≥ 20%/yr annualised. Direction: short the venue with the higher funding, long the other.
- **Exit** (same delay): m changes sign, or |m| < 5%/yr, or the coin stops being eligible, or it stops trading on either venue.
- **P&L per day held** = notional × (return of the long venue − return of the short venue + funding received on the short − funding paid on the long).
- **Costs per side**: Binance 5 bps fee + the Phase 1b slippage tier; Hyperliquid 4.5 bps fee + 10 bps slippage. Both legs pay on entry and on exit.
- **Margin upkeep**: when a coin has moved more than 30% (close to close) since the slot's margin was last balanced, charge 10 bps of notional for moving collateral between venues.
- **Forced close**: if either venue's daily high or low is more than 60% away from the previous close, the slot is closed that day with an extra 2% of notional as liquidation and slippage loss.
- Missing funding print while held: the paying side is charged 3 bps/day, the receiving side gets nothing.
- Idle capital earns nothing.

## 4. Windows, metric and gates

- Sanity window: **2023-07-01 → 2024-12-31**. Holdout: **2025-01-01 → 2026-10-31**.
- The script truncates all data at 2024-12-31 for the sanity run. The holdout is read only if the sanity run's net CAGR is above 4%/yr (idle USDT); otherwise H0 is recorded and the holdout stays unread.
- **Primary metric**: net CAGR on total capital in the holdout.
- **Holdout gate** (all required): net CAGR ≥ 8%; Sharpe in excess of 4%/yr ≥ 1.0; max drawdown ≤ 10%; positive net return in both 2025 and 2026 (Jan-Oct); at least one slot filled on ≥ 50% of days.
- Secondaries, reported and never decisive: a 2x-per-leg version with forced close at ±40%; lookback 3 and 14 days; entry threshold 10% and 30%; average gap captured; number of round trips; share of P&L from funding versus basis.

**Exam-window accounting.** Phase 1b said the 2025-01 → 2026-10 window was retired. That statement was written about strategies on the Binance price panel, and it stands for them. This experiment uses a different dataset (the gap between two venues' funding for the same coin). What earlier work revealed about the window (prices fell; MYX squeezed; Binance funding was low) does not tell us the size or persistence of that gap, except that low funding levels are a hint against it. It is counted as candidate 5 on the calendar window, it is the only one this experiment sends, and after it the window is closed for every crypto strategy.

## 5. Decision each outcome triggers

- **Pass**: write a build proposal for the Board: a two-venue executor (Binance futures + Hyperliquid), paper-traded for 8 weeks first. It needs a Hyperliquid account and wallet, which is the Board's decision. Nothing is deployed from this experiment.
- **Fail**: no build. Record the result. With Phase 1, 1b and 1c all negative, recommend the Board treat systematic crypto trading at this account size as not viable for now, keep the funding watcher, and put effort where the other research thread (non-crypto markets) points.
- **Sanity check fails**: same as fail, holdout unread.

## 6. Risks of a false positive

- **Close-price asynchrony and thin Hyperliquid books** can create fake basis P&L. Handled by reporting funding and basis P&L separately; a pass that comes from basis rather than funding will be flagged as not credible.
- **Short history** (18 months sanity, 22 months holdout) and one configuration: there is no selection step, so no multiple-comparison issue, but also little power.
- **Costs**: Hyperliquid slippage for a small account is likely below 10 bps; for a large one it is not.
- **Not in the numbers**: bridge and smart-contract risk on Hyperliquid, exchange counterparty risk, withdrawal delays in a crisis, and the auto-deleveraging both venues can apply to profitable positions.
- **Determinism**: run twice, outputs must match.
