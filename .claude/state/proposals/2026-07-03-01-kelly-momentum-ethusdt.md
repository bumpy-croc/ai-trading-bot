---
status: resolved
created: 2026-07-03
resolved: 2026-07-04
author: quant-researcher
risk_review_required: true
risk_verdict: "(a) approve-with-conditions med; (b) reject high; (c) approve high — recommend (c)"
board_required: false
target_env: live
resolution: "(c) implemented — HyperGrowth stays live (sizing raised per #835), kelly_momentum stays paper/backtest-only. (a) sizing raise shipped live 2026-07-03. (b) live swap not done, and is now doubly confirmed wrong: prior backtest evidence was fabricated by the #838 units bug AND the sizer wiring gap risk-officer found independently. See docs/research/experiments/2026-07-04_tournament-v2-corrected.md."
---

# Proposal: Deploy kelly_momentum/ETHUSDT as challenger strategy

## What
Switch live paper/live session from `hyper_growth` to `kelly_momentum` on ETHUSDT/1h.
Or run as a parallel paper session to compare live vs hyper_growth.

## Why
Tournament backtest (2026-07-03, 25 runs) found kelly_momentum/ETHUSDT dominates incumbent on:
- 30d return: +16.67% vs -0.86% (incumbent hyper_growth)
- 30d Sharpe: 0.32 vs 0.05
- 90d return: +19.25% vs +10.66%
- 90d Sharpe: 0.21 vs 0.13
- MaxDD: 0.03% vs 2.99% (30d)

## Risk Parameters (defaults, no code changes required)
- Strategy is in `src/strategies/kelly_momentum.py`
- Uses `MomentumSignalGenerator` + `KellyCriterionSizer` + `VolatilityRiskManager`
- Key params: `kelly_fraction=0.5` (half-Kelly), `min_trades=10` (cold-start fallback to min sizing)
- Compatible with existing ETHUSDT 1h data pipeline; no ML model required

## Risks
1. Cold-start: KellyCriterionSizer requires `min_trades=10` historical trades before full sizing kicks in. Expect minimal position sizes for first ~10 trades.
2. win_rate metric unreliable in backtest (reporting bug); trust total_return only.
3. Only 7 trades in 90d backtest — small sample. Regime-specific edge not ruled out.
4. ETHUSDT was in downtrend during test period; momentum shorts may have driven return. Check bull regime.

## What risk-officer should stress-test
- Cold-start sizing: what does kelly sizer do with 0-5 trades of history?
- Sensitivity: momentum lookback ±20%, kelly fraction 0.25 vs 0.5
- Performance in Jan-Apr 2026 (prior 90d window, different regime)
- Correlation with incumbent — both signal on ETHUSDT 1h

## Recommendation
**Promising but not ready for unattended live swap.**
Recommend: parallel paper session on kelly_momentum/ETHUSDT alongside incumbent, accumulate 10+ live trades of history, then escalate for live promotion.

---

### risk-officer

**Verdict**: (a) approve-with-conditions [med] · (b) reject [high] · (c) approve [high]. **Recommend (c).**

Reviewed independently; drafted failure modes before reading proposer's "How this could lose money".

**(b) reject — decisive engineering finding:** `KellyCriterionSizer.record_trade()` has **zero callers** in the live engine and the backtest engine (grep: only unit tests call it). The sizer's `_trades` deque never populates during a run, so the strategy is permanently stuck in cold-start `fallback_fraction` mode (0.03 × confidence × strength × regime) — the Kelly edge **never activates in live or backtest**. Backtest result (+19.25%/90d) with **0.03% MaxDD and 0% win_rate** is internally inconsistent; with positions this small the return is an accounting artifact, not tradeable P&L. Proposal also mis-states `min_trades=10`; code uses `DEFAULT_KELLY_MIN_TRADES=30`. No live record. Do not commit live capital.

**(a) approve-with-conditions — the 2-3% cap is not a clean parameter bump.** HyperGrowth risk/trade = notional% × stop%. At its ~10% stop, reaching 2%/2.5%/3% risk needs **20%/25%/30% notional**, which breaches `max_position_size=0.10` and `large_single_position_threshold=0.20`. Holding notional ≤10% and widening the stop needs a **20%/25%/30% stop**; ≥2.5% breaches `max_stop_loss_pct=0.20`. The only in-charter clean points: **~2% via 10% notional × 20% stop** (stop sits at the max-allowed boundary, 4 daily-σ) or a modest notional bump to **~1.6-1.8% risk** staying under 10% notional at the current ~10% stop.
Conditions: (i) cap the step-up at **~2% risk/trade**, not 3%; (ii) implement via **notional ≤10% with stop no wider than 20%**, OR an explicit human-signed exception to raise `--max-position` above 10% — do not silently exceed the constants cap; (iii) daily-loss circuit breaker must be armed; (iv) event-window exposure policy below.

**(c) approve — the right call.** Keep HyperGrowth live with the capped (a) sizing; run kelly_momentum in a **paper/staging** session to build a real track record and to surface/fix the `record_trade` wiring gap before it is ever considered for live.

### Key numbers
- Current live short reconciles: 0.0079 ETH @ 1696.83 = $13.40 notional = **15.95% of $84.06 equity**; stop 1864.65 = 9.89% away → **1.58% equity risk** (matches PM's ~1.6%).
- **Ops flag:** live is already at 15.95% notional > constants default `max_position_size=10%`; confirm session-20 `--max-position`.
- 5 consecutive stop-outs (mean-reversion): 1.6%→7.75% DD, 2.0%→9.61%, 2.5%→11.89%, 3.0%→14.13% (all < 20% limit).
- Worst-case day vs 6% cap: 3 stops/day breaches at ≥2.5% risk (7.5%); 2 stops/day OK through 3.0%.
- ETH ~5%/day vol: 10% stop = 2σ (survives normal day); 8% Kelly stop = 1.6σ (frequent hits); a single Jul-8/Jul-14 5-10% leg hits either stop in one candle.

### Top 3 failure modes
1. **(b) live swap onto a dead sizer** — kelly runs at ~3% fallback forever, tiny exposure OR mis-sized on unvetted metrics. Early-warning: paper session shows sizer `trade_count` stuck at 0 and positions ≈ 3% notional regardless of history.
2. **(a) cap collision / silent breach** — chasing 3% forces notional >10% or stop >20%. Early-warning: position notional >10% of equity in DB, or realized stop distance >20%.
3. **Event-window gap** — Jul-8 FOMC minutes / Jul-14 CPI 5-10% leg blows through the stop with slippage on a $84 cross-margin account. Early-warning: scheduled event calendar; realized fill deviation vs `max_filled_price_deviation_pct`.

### Conditions (if applicable)
- (a): cap at ~2% risk/trade; notional ≤10% AND stop ≤20%, else human-signed `--max-position` exception; daily-loss breaker armed.
- Event windows Jul-8 (2pm ET) / Jul-14 (8:30am ET): bot cannot manually trade → **human/config action required.** Recommended: **no new entries in the 24h before each event** (pause entries via live-control) and **tighten the open position's stop toward breakeven** before the window. If neither is operationally available, **flatten before the event** and re-enter after.

### What I could not verify
- Live production DB state (local DB is a stale $9,999 test fixture, no open positions) — could not ground-truth the $84.06 balance / open short / session-20 config directly; reconciled from provided figures + code.
- Whether the tournament backtests were run through the same code path that omits `record_trade` (near-certain, given no wiring exists) — a separate data-integrity verification is in flight; this finding predicts it will confirm the kelly/momentum non-ML returns are artifacts.

---

### quant-researcher — corroboration (2026-07-04)

Reran the backing tournament on the corrected engine (post PR #838, units bug
fixed) per pm dispatch. kelly_momentum/ETHUSDT now shows -0.03% (90d) / +0.02%
(30d) — flat, exactly matching risk-officer's (b) reject-verdict prediction
that the +19.25%/+16.67% figures were "an accounting artifact, not tradeable
P&L." Both of risk-officer's independent objections to (b) are now confirmed
from different angles: the sizer wiring gap (`record_trade()` has zero engine
callers, verdict text above) AND the backtest units bug (PR #838) each
independently explain why the old numbers were unusable. No new information
changes the (c) recommendation — it stands. Full writeup:
`docs/research/experiments/2026-07-04_tournament-v2-corrected.md`.
