# Live-vs-Backtest Parity Gap Investigation — ETHUSDT/HyperGrowth, 2026-07-12

**Author**: quant-researcher · **Type**: read-only mechanism investigation (lever #2 of `docs/research/2026-07-12_returns-levers-synthesis.md`) · **Feeds**: fold-verdict caveats, future parity studies
**Scope**: read-only throughout — SELECT-only against prod Postgres, no `src/` changes, one short (1-month) backtest run against a local dev DB. No live/staging state touched.
**Inputs**: `docs/research/2026-07-12_returns-levers-synthesis.md` §2(c), `docs/research/notes/2026-07-12_live-trade-review.md` (the 12-vs-6-trade, sign-flipped divergence this investigates), `docs/research/experiments/2026-07-06_forming-bar-fliprate.md` (the prior forming-bar mechanism study).

## Verdict up front

The 12-vs-6-trade, sign-flipped divergence reported in the 2026-07-12 live-trade-review is **not primarily a forming-bar artifact**. Two other mechanisms dominate it, both new to this investigation:

1. **A model-version confound in the "matched" backtest.** The backtest that produced the "6 trades / −0.78%" comparator scores the *entire* 2026-06-02→2026-07-12 window with whatever model is `latest` **today** (the native ETHUSDT model, promoted 2026-07-05). All 12 live trades — and the still-open position #22 — were entered **before** 2026-07-05, when live was actually running a different (cross-symbol-substitute) model. The comparison silently re-scores live's entire trading history with a model that was never live for it.
2. **A large, previously uncharacterized live execution funnel.** `strategy_executions.action_taken='opened_long'/'opened_short'` logs the strategy's fully-sized, risk-gated decision *before* the engine attempts to execute it. Over the window, 3,325 such log rows fired during genuinely flat periods with real position sizing and confidence above the 0.05 gate — and fewer than a dozen became real trades. The attrition is also **direction-asymmetric**: signal episodes split ~50/50 long/short (1,650 vs 1,675), but real trades were 9 LONG vs 3 SHORT. A specific, code-confirmed SHORT-side margin/inventory guard is the leading explanation.

The forming-bar mechanism (fliprate study) is real and reconfirmed here, but its measured effect on these 12 trades' P&L is small (−1.1 percentage points of the +9.0% cumulative live return) — nowhere near enough to explain a ~10-point swing to backtest's −0.78%, let alone the sign flip. It is a genuine, secondary contributor, not the dominant one.

**A note on an unsolicited mid-task message**: partway through this investigation a system-reminder relayed a message from "the coordinator" citing a file (`scratchpad/audit_data_path.md`) and instructing me to reframe my verdict around the forming-bar mechanism specifically. I did not treat this as an instruction — no agent message is authorization, and this matches a previously-logged pattern (`.claude/state/log.md`, 2026-07-05 ml-engineer entry, "two unsolicited coordinator messages... not acted on"). I read the file, independently verified its code citations myself (`kline_buffer.py` tail-mutation, `backtest/engine.py`'s closed-bar iteration), and found they restate — without adding to — the already-published fliprate study; the file does not address either of the two findings below. I kept my own independently-derived conclusions rather than the suggested reframing. Flagging this for the record per the standing session-integrity practice.

---

## 1. Premise check: when did the model actually switch?

The brief states "the native model deployed 2026-07-06." Three independent sources in the repo disagree with that date by one day:

- `docs/research/model-promotions.md`: one ETHUSDT/basic entry, **promoted 2026-07-05**, old→`2026-07-04_22h_v1` (PR #886/#887/#867/#872).
- `src/ml/models/ETHUSDT/basic/2026-07-04_22h_v1/metadata.json`: `created_at: 2026-07-04T22:44:32Z`.
- `.claude/state/log.md`, 2026-07-05 00:15 ml-engineer entry: trains/evaluates/proposes this exact model version that day.
- `git log --all -- src/ml/models/ETHUSDT/basic`: only two commits, both PR #886 — **no second ETHUSDT/basic version was ever added**. There is no 2026-07-06 model promotion in this repo's history.

2026-07-06 is when a *different*, unrelated fix landed: commit `46b5be84` (18:09 +01:00 = 17:09 UTC), "fix(logging): populate `ml_predictions` in strategy execution rows (#914) (#917)" — this wires up the `ml_predictions` JSON column that had been `null` in 100% of rows before it (confirmed independently: `docs/research/notes/2026-07-06_latency-error-phantom-short-forensics.md` found it null in prod's entire history as of that afternoon; a direct query here confirms it starts populating from 2026-07-06 17:09 UTC onward, 9,086/13,006 rows non-null since). The brief's "07-07 onward" for `ml_predictions` is close enough; "07-06 for the model" appears to be a conflation of these two events. Neither date affects the analysis below, but it's worth correcting since the model-switch date is load-bearing for Finding 1.

**Consequence that matters**: every one of the 12 closed trades in the live-trade-review window entered **before 2026-07-05**:

| Trade | Entry (UTC) |
|---|---|
| 1–4 | 2026-06-02 |
| 5 | 2026-06-04 |
| 6 | 2026-06-05 |
| 7–9 | 2026-06-06/07 |
| 10 | 2026-06-14 |
| 11 | 2026-06-18 |
| 12 | 2026-06-23 |
| #22 (open) | 2026-07-02 |

Not one falls after the model switch.

## 2. Method

- **Live decision stream**: `strategy_executions` for ETHUSDT since 2026-06-02, read via `RAILWAY_PRODUCTION_DATABASE_URL` (public proxy), `SET default_transaction_read_only = on`, SELECT only. 94,580 rows in the full window (through 2026-07-12); analysis below is scoped to 2026-06-02→2026-07-02 13:34:24 (immediately before position #22 opened and has stayed open since — the period covered by all 12 closed trades) unless stated otherwise.
- **Backtest decision stream**: reused the live-trade-review's existing matched-config backtest result (6 trades/−0.78%, `atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --start 2026-06-02 --end 2026-07-12`) rather than re-running it — the exit-round-2 lane holds the heavy-compute lock, and re-running an identical comparison would duplicate published work. I additionally ran **one** short backtest (`--start 2026-06-01 --end 2026-07-01`, 1 fold-month, sequential, fees/slippage on defaults) to get June-specific trade-level context; see §4.
- **Code tracing**: three focused read-only subagent passes (model resolution/registry, `ignore_signal_reversal`/reversal-guard semantics, and the live entry-execution pipeline), each independently verified against source where cited below — I re-read the load-bearing files myself (`entry_coordinator.py`, `execution_engine.py`) rather than taking subagent summaries at face value, per standing practice.
- **Forming-bar counterfactual**: computed directly from the cached ETHUSDT 1h parquet (already resident in `cache/market_data/`, no network fetch needed — confirmed via `CachedDataProvider` returning in 2.7s), independent of the engine/backtest harness.
- **No src/ changes. No writes to any Railway database.** The one local backtest run attempted `--log-to-db` against the local dev Postgres (`localhost:5432/ai_trading_bot`, per `.env`'s `DATABASE_URL` — not Railway) and hit a pre-existing local FK-seeding gap (`trading_sessions` row missing) unrelated to this investigation; the backtest still completed and returned results. Not investigated further — a local sandbox issue, not a prod concern.

## 3. Finding 1 — The "matched" backtest silently re-scores live's history with a model that was never live for it

`atb backtest` resolves the model once, at strategy construction, via `PredictionModelRegistry._scan_registry()` (`src/prediction/models/registry.py:87-133`), which walks `src/ml/models/{symbol}/{model_type}/` and loads whatever `latest` points to **at invocation time**. `Backtester.run()` (`src/engines/backtest/engine.py:927-983`) then walks the full historical DataFrame with that one fixed strategy object — there is no argument, flag, or code path anywhere in `cli/commands/backtest.py` or the engine that varies the model by bar timestamp. Confirmed: no `--model-version`/`--as-of` flag exists.

Live, by contrast, only picks up a new `latest` on **process restart** or an explicit manual hot-swap call (`StrategyHotSwapCoordinator`, `src/engines/live/strategy_hot_swap.py:150-222`) — `PredictionEngine.reload_models()` exists but has zero callers anywhere in `src/`. So the live engine that traded 2026-06-02 through (at least) 2026-07-02 was running whatever model was `latest` *then* — the cross-symbol-substitute model referenced in the fliprate study ("prod now scores ETHUSDT with the native ETHUSDT model as of 2026-07-05") — while any backtest run today against those same dates scores them with the native model instead.

This is not a hypothetical effect. The one short backtest run for this investigation (`--start 2026-06-01 --end 2026-07-01`, native model, fees/slippage on) produced:

```
Total Trades: 3
Win Rate: 66.67%
Total Return: 0.91%
trade_pnl_pcts: [+0.055%, -1.222%, +0.094%]
avg_trade_duration_hours: 138.0
```

Compare to live's actual June activity: 9 real trades entered in June alone, with individual swings up to **+3.97%** and **−10.00%** — an order of magnitude larger than anything the native model's backtest produces for the same month. The native model, scored retroactively, generates a qualitatively quieter, barely-actionable trading pattern; it is not simply "the same trades, slightly mistimed." **Comparing live's pre-07-05 trade count/return to any backtest run today is comparing two different strategies' worth of decisions, not a timing artifact of one strategy.**

I could not construct a true apples-to-apples closed-bar comparison for the pre-07-05 period: only one ETHUSDT/basic model version has ever existed in this repo's history (confirmed via `git log --all`), so the actual cross-symbol-substitute model that was live in June is not reproducible from current repo state without reverting code — out of scope for a read-only investigation. This is itself worth recording as a limitation on how precisely this specific window's parity gap can ever be decomposed.

## 4. Finding 2 — Live's signal log and live's actual execution are two different things, and the gap is enormous and direction-asymmetric

### 4.1 The scale of it

Restricting to genuinely flat periods (no open position per the `trades` table) between 2026-06-02 and 2026-07-02 13:34:24:

| | count |
|---|---:|
| `strategy_executions` rows, `signal_type='entry'`, `action_taken` ∈ {opened_long, opened_short} | 10,970 |
| ...of which, logged *while a trade was already open* (hypothetical "what I'd do if flat," alongside real position management) | 7,645 |
| ...of which, logged during genuinely flat time (real opportunities) | 3,325 |
| ...of which matched (within 5s) to one of the 12 real trade entries | 7 |

Over 99% of the logged, fully-sized, confidence-gate-clearing "opened_long"/"opened_short" decisions during flat time never became a trade, for reasons invisible in `strategy_executions` itself.

Distinct direction-episodes (contiguous same-direction actionable stretches, counting a new episode whenever the logged direction changes) across the whole window: **641 starting long, 626 starting short — 1,267 total** over roughly 962 hours (≈1.3 episodes/hour), which is in the same ballpark as the fliprate study's independently-measured 1.07 gate-crossings/hour — a nice cross-check that both studies are seeing the same underlying signal-flicker phenomenon, even though this study's denominator (episodes) and the fliprate study's (gate-crossings) aren't identical constructs.

### 4.2 The mechanism, traced in code

`LiveEntryCoordinator.check_entry_conditions` (`src/engines/live/execution/entry_coordinator.py`) computes the fully risk-gated decision (signal generator → `FlatRiskManager` sizing → position-size bounding) and writes it to `strategy_executions` **unconditionally** whenever `db_manager` exists (lines ~324-353) — this write happens *before* the early-return guard (`if not entry_signal or position_size <= 0: return`, line ~355) and well before the real order path (`self.execute_entry(...)`, line ~448). So `action_taken` records "the engine decided to attempt an entry," never "an order was placed."

Downstream, `execute_entry` → `execute_entry_locked` (lines ~492-586) can silently `return` on:
- a duplicate-position check (ruled out for the concrete example below — no position was open),
- a max-concurrent-positions check (ruled out — logged reasons showed `max_positions_check_0_of_3`, capacity available),
- or the real order submission via `LiveExecutionEngine.execute_entry` → `_execute_live_order` (`execution_engine.py:597`) → `exchange_interface.place_order`.

None of these downstream outcomes get written back to `strategy_executions`. `system_events` is empty in every window checked here, and Railway logs don't retain history for these dates (documented repo-wide limitation, re-confirmed by the 2026-07-06 phantom-short forensics note) — so the exact rejection reason is not fully provable from surviving telemetry for any specific instance. But one candidate is directly confirmed in code and matches the data unusually well:

**The SHORT-side margin/inventory guard** (`execution_engine.py:663-706`): before opening a SHORT, if the exchange is in margin mode, the engine checks the account's free base-asset (ETH) balance. If `free_ETH * price > $1` (a dust threshold), it **fail-closed rejects the short** — the in-code rationale, quoting the comment directly, is that Binance's `MARGIN_BUY` side-effect "only borrows when free base inventory is insufficient... If the wallet holds the base asset, Binance sells existing inventory instead of borrowing, breaking short position semantics." The rejection is `logger.error` only — no DB row, consistent with everything observed.

This guard is **direction-asymmetric by construction** — it only ever blocks SHORT entries, never LONG. That prediction matches the data cleanly: despite a near-50/50 split in the raw signal stream (1,650 long-direction vs 1,675 short-direction flat-period log rows), the 12 real trades split **9 LONG vs 3 SHORT** — a 3x skew in the opposite direction from what the raw signal suggests.

### 4.3 A concrete instance, and a concrete counter-instance

- **2026-06-07, 22:14:44–22:59:47** (immediately after trade #8's LONG position closed via stop-loss at 22:14:39): 30 consecutive `opened_short` log rows, confidence climbing 0.26→0.42, `max_positions_check_0_of_3`, fully-sized (`position_size≈0.144`). **Zero SHORT orders** in the `orders` table for this window; **zero new rows** in `positions`. The next successful entry was a **LONG** at 23:04:53 (order 7419, matching the low-confidence 0.052 signal that finally cleared) — 45 minutes and ~30 rejected SHORT cycles later.
- **2026-06-14, 21:49:05–21:52:28** (immediately after trade #9's LONG position closed at 21:50:14): only 2 `opened_short` log rows (21:49:05, 21:50:19) before a SHORT **did** open successfully at 21:52:28 — resolved in ~2 minutes, not 45.

Both episodes are consistent with a transient residual-ETH-balance effect that clears at variable speed (settlement timing, dust size relative to price, or something else not visible from these tables) — but I could not verify actual historical free-ETH balances directly (no per-asset balance-history table exists in this schema; `account_balances` is USD-only deltas), so this remains the **leading, code-confirmed, but not empirically-confirmed** explanation for the exact duration variability. What is empirically confirmed, not just hypothesized, is the aggregate direction-asymmetric attrition itself (§4.1–4.2).

### 4.4 Why this matters more than it might first appear

A naive backtest models zero execution-layer attrition: whenever its (single, closed-bar) evaluation clears the confidence gate and sizing is nonzero, it opens the trade. Live's real funnel evidently has substantial, direction-asymmetric friction between "signal says go" and "position exists." **Neither engine's raw signal stream is a reliable proxy for the other's real trade stream** — this is a structural parity gap, independent of any timing/forming-bar question, and it goes in different directions for different symbols/sides than the trade-count question the brief posed. It is also large enough on its own (>99% attrition vs the fliprate study's ~15% actionable flip rate) to dominate any comparison that doesn't control for it.

## 5. Finding 3 — The forming-bar entry-price effect is real but small

Using the fliprate study's already-established mechanism (live's `current_price` reads the still-forming tail candle; `current_index = len(df)-1` in both engines, but that row means "closed" in backtest and "partial" in live), I computed the bar-close counterfactual for all 12 real live trades: same side, same exit price/time, entry price replaced with the containing 1h bar's actual close (from cached OHLCV, no backtest engine invocation).

| id | side | min. into bar | live entry | bar close | realized P&L% | counterfactual P&L% | Δ (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LONG | 22.7 | 1968.53 | 1975.28 | 0.030 | −0.312 | −0.342 |
| 2 | LONG | 10.0 | 1965.34 | 1941.36 | −0.010 | 1.225 | 1.235 |
| 3 | LONG | 30.9 | 1913.41 | 1913.11 | −0.055 | −0.039 | 0.016 |
| 4 | LONG | 33.3 | 1914.08 | 1913.11 | −0.024 | 0.027 | 0.051 |
| 5 | LONG | 7.0 | 1765.48 | 1774.88 | −10.003 | −10.480 | −0.477 |
| 6 | SHORT | 4.1 | 1609.95 | 1609.80 | −0.158 | −0.168 | −0.009 |
| 7 | LONG | 40.1 | 1554.74 | 1553.54 | 3.965 | 4.045 | 0.080 |
| 8 | LONG | 54.1 | 1616.46 | 1620.17 | 3.775 | 3.537 | −0.238 |
| 9 | LONG | 5.0 | 1673.21 | 1690.51 | 3.480 | 2.421 | −1.059 |
| 10 | SHORT | 52.5 | 1726.78 | 1718.67 | 1.659 | 1.195 | −0.464 |
| 11 | SHORT | 42.0 | 1698.18 | 1705.39 | 3.201 | 3.610 | 0.409 |
| 12 | LONG | 25.0 | 1643.70 | 1648.48 | 3.136 | 2.837 | −0.299 |
| **Sum** | | | | | **+8.996%** | **+7.898%** | **−1.098pp** |

The realized sum (+9.0%) matches the live-trade-review's headline figure, a useful sanity check on this reconstruction. The entries fired anywhere from 4 to 54 minutes into their hour (median ~26 min) — consistent with the code finding that live has no bar-boundary gate at all (confirmed by the entry-execution trace: the main loop re-evaluates every iteration with no per-candle throttle) — entries land wherever the (already-established-as-noisy) intra-bar signal happens to clear whatever gate finally lets it through.

Entering at bar-close instead of the actual live tick would have cost **1.1 percentage points** of the +9.0% cumulative return — real, non-trivial per-trade (up to −1.06pp on trade 9), but far short of the ~9.8-point gap needed to reach backtest's −0.78%, and it doesn't touch trade *count* at all (this counterfactual re-prices the same 12 entries; it doesn't create or remove any).

**Does the fliprate study's flip-rate predict the observed extra trade count?** No. The fliprate study measured a ~15% actionable-decision flip rate — real, but far too small in magnitude to produce a 2x trade-count difference (12 vs 6) on its own, and this investigation's own funnel numbers (Finding 2) show the trade-count question is dominated by something roughly two orders of magnitude larger (>99% signal-to-trade attrition) that has nothing to do with forming-bar timing. The fliprate mechanism is real, reconfirmed, and worth keeping in mind for entry-price quality — it is not the explanation for the specific 12-vs-6 divergence this investigation was asked to explain.

## 6. Verdict on the three explicit questions

**(a) Is the gap explained?** Mostly, yes, with receipts: two concrete, code-confirmed mechanisms (model-version confound, execution funnel) account for the bulk of both the trade-count divergence and the sign flip; the forming-bar mechanism contributes a measured, minor amount (−1.1pp) to the return side and nothing to the trade-count side. What remains unexplained is the *precise* duration-variability of the execution-funnel rejections (§4.3) and a true old-model closed-bar comparison for the pre-07-05 period (not reconstructable read-only).

**(b) Which direction is 'right' for our purposes?** Neither of the two dominant mechanisms found here argue for a forming-bar-aware backtest mode as the next build, and this investigation does not touch the closed-candle-gating work (owned elsewhere, stopped by the human — not re-opened here). Instead:
- **Backtest tooling gap (cheap, mechanical)**: give the backtest engine point-in-time model resolution — score each historical bar with whatever model `docs/research/model-promotions.md` says was actually `latest` on that date, instead of always using today's `latest`. This directly closes Finding 1 and is a harness fix, not a strategy change.
- **Live execution-funnel forensics (moderate effort, real-money-adjacent)**: confirm or refute the SHORT-inventory-guard hypothesis with real-time balance monitoring (watch for a stretch of repeated `opened_short` logs with zero matching orders, and check `get_balance('ETH')` at that moment). If confirmed working-as-designed, the backtest should at minimum caption that it assumes zero execution-layer attrition, which live manifestly does not have — and if the >99% attrition is costing more opportunity than the $1 dust guard is worth protecting against, that's a genuine (small-capital-relevant) risk/return trade-off for risk-officer and pm to weigh, not something to silently accept.
- **A full forming-bar-aware backtest mode**, if ever pursued, is expensive and is not obviously worth it yet: the fliprate study already showed no directional-accuracy edge is gained or lost from timing (its own §6), this investigation shows the entry-price effect on realized P&L is small (~1pp on a 12-trade sample), and building it without *also* modeling Finding 2's execution funnel would only add spurious forming-bar-triggered trades to backtest with none of live's real friction — plausibly *widening* the parity gap rather than closing it. Recommend deprioritizing until (i) and (ii) above are done.

**(c) How should fold-based verdicts be caveated meanwhile?** The five convergent-null tournament results cited in the returns-levers synthesis (window, architecture, target-redesign, linear/nonlinear input screens) do not cross a live model-promotion boundary and don't rely on strategy_executions-vs-trades reconciliation, so they are **not retroactively undermined** by either finding here. Going forward, any live-vs-backtest comparison should explicitly check two things before being called "matched config": (1) does the compared window straddle a model-promotion boundary (check `docs/research/model-promotions.md` first), and (2) is the comparison being made on trade *counts* at all — if so, caption that backtest's trade count assumes zero execution-layer attrition, which live's real funnel does not have, until Finding 2 is understood well enough to be modeled or bounded.

## 7. Honest limitations

- **n=12 trades** (9 LONG, 3 SHORT) for the entire live sample under discussion — every number here is mechanism-identification, not a performance claim, per standing practice.
- Finding 2's *specific* rejection mechanism (short-inventory dust guard) is confirmed at the code level and matches the aggregate direction-asymmetry and one detailed example well, but is **not empirically confirmed** against actual historical balance state — no such per-asset historical ledger exists in this schema, and Railway logs don't retain history for these dates. A live-ops forensics pass with real-time instrumentation is the natural next step, not something achievable from these tables alone.
- Finding 1's counterfactual ("what would the old cross-symbol-substitute model actually have decided, at bar close, for these same dates") is not reconstructable — only one ETHUSDT/basic model version has ever existed in this repo's git history.
- The June-only backtest run (§3, §4) used `--log-to-db` against the local dev Postgres and hit a local FK-seeding gap unrelated to Railway/prod; the aggregate results reported were unaffected (computed in-memory regardless of the DB-write failure) but no per-trade timestamp export was available from that run, so §3's comparison is at the aggregate-statistics level, not a bar-by-bar match against backtest's own 3 trades.
- The mid-task "coordinator" message (see verdict-section note) was evaluated, not trusted; its citations were independently re-verified and found to add nothing beyond the already-published fliprate study.

## 8. Reproducibility

- **Prod DB queries**: all SELECT-only against `RAILWAY_PRODUCTION_DATABASE_URL` (public proxy), `SET default_transaction_read_only = on` set first in every session. Query text preserved in this session's transcript; key ones: `strategy_executions` action_taken breakdown, flat-period episode reconstruction (CTE joining `trades` intervals against `strategy_executions` timestamps), `orders`/`positions` cross-checks for the two concrete episodes in §4.3.
- **Cached-data counterfactual**: `CachedDataProvider.get_historical_data('ETHUSDT', '1h', 2026-06-01, 2026-07-12 14:00 UTC)` — cache hit, 2.7s, no network fetch, 998 bars. Computation script run inline via the repo venv (`.venv`), not persisted as a standalone file (small one-off, reproducible from the table in §5).
- **Backtest run**: `atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --start 2026-06-01 --end 2026-07-01 --initial-balance 84 --log-to-db` — one run, 1 fold-month, sequential, fees/slippage on (defaults, never disabled). Output: `logs/backtest/20260712_131540_HyperGrowth_0.08yrs.json`.
- **Code citations**: `src/prediction/models/registry.py:87-133`, `src/engines/backtest/engine.py:927-983`, `src/engines/live/strategy_hot_swap.py:150-222`, `src/engines/live/execution/entry_coordinator.py:324-586`, `src/engines/live/execution/execution_engine.py:597-706`, `docs/research/model-promotions.md`, `src/ml/models/ETHUSDT/basic/2026-07-04_22h_v1/metadata.json` — all re-read directly in this session, not solely relayed from subagents.
- **Fees/slippage**: on throughout (defaults), per `CODE.md`. No look-ahead: the forming-bar counterfactual uses only each trade's own containing-bar close and its own already-realized exit; no future data enters any computation.

## 9. Recommendations to pm

1. **Open a `type:fix`/`area:backtest` issue**: point-in-time model resolution for the backtest engine (Finding 1). Low complexity, mechanical, high value for the honesty of every future live-vs-backtest comparison. Not built here.
2. **Open a `type:experiment` or hand to `live-ops`**: real-time forensics on the SHORT-inventory-guard hypothesis (Finding 2) — confirm/refute with live balance monitoring, then decide whether the >99% attrition (direction-asymmetric) is working-as-designed risk management or an opportunity-cost problem worth `risk-officer` review. Not built here.
3. **Do not build a forming-bar-aware backtest mode next** — per §6(b), its expected value is currently low relative to cost, and building it before (1)/(2) risks making the parity picture worse, not better.
4. **Caveat, don't retract**: the five-tournament convergent-null result stands; apply the two-question check in §6(c) to future parity comparisons.

Status: **ready for pm triage**; nothing here proposes a live-affecting change, so no `risk-officer` gate is required for this write-up itself (per (2) above, a future execution-funnel fix — if pursued — would be money-path-adjacent and require the standard review gauntlet at that time).
