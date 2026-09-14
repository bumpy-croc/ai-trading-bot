# Landing #1106 — closed-candle gating (parity decision D1)

**Date**: 2026-08-25
**Branch**: `fix/closed-candle-gating`
**Issue**: #1106
**Scope**: rebase the recovered implementation onto current `develop`, re-verify the
design against what has landed since, add a backtest↔live decision-parity test, open a PR.

---

## 1. Rebase notes

Base moved from `b6dc4e25` (pre-#1073) to `f64ac4ca` (merge of #1108).

**No textual conflicts.** `git rebase origin/develop` replayed the single commit cleanly.
That was not the expectation going in, so the important part is the semantic check, not
the mechanical one — the conflict-free replay only proves the *lines* did not collide.

What actually landed in between, and why each was benign:

| Change | Touches | Interaction with the gate |
|---|---|---|
| #1073 risk hydration | `policy_hydration`, risk manager wiring | Hydration runs *inside* `runtime_process_decision`, downstream of the index the gate chooses. Gating changes *which index* is evaluated, never whether hydration runs. |
| #1080 source-root guard | `src/_source_root.py`, import path | Import-time only; no loop interaction. |
| #1102 backtest truncation fix | backtest driver | Backtest-side only. Relevant to parity *claims*, not to live gating; the new parity test pins the two sides against each other so a future truncation regression on either side now fails a test. |
| #1108 SL re-placement loop | `execution_engine.py`, `stop_loss_manager.py`, `order_tracker.py`, `binance_provider.py`, +4 lines in `trading_engine.py` | **The safety-critical one.** All of it lives in the execution/order-poll layer, which the gate does not touch. The 4 lines in `trading_engine.py` are outside `_trading_loop`'s decision block. Verified by reading the post-rebase loop (below). |

The only real hazard the rebase could have introduced is a protective call being moved
*inside* the gated branch, and it was not.

## 2. Design re-verification (D1 still correct)

### 2.1 The mechanism the gate closes

Per `docs/research/experiments/2026-07-06_forming-bar-fliprate.md`, the flip mechanism is
**not** the model seeing malformed features — `ml_signal_generator._get_ml_prediction`
slices `iloc[index - sequence_length : index]`, exclusive of `index`, so the feature window
was already closed-bars-only in both engines. The model's raw prediction is *constant* for
the whole forming hour.

What floats is `current_price = df["close"].iloc[index]`, the denominator of
`predicted_return`. A fixed numerator over a moving denominator is the entire source of the
43.2%-at-minute-5 flip rate.

Gating at the last closed bar's index therefore freezes exactly the right variable, and
does so by evaluating at an index whose close is final. This is the minimal correct fix for
the measured mechanism — confirmed, not assumed.

### 2.2 Protective paths are ungated (the PM condition on D1)

Post-rebase, inside `_trading_loop`, the following all run on `current_index` /
`current_price` — the forming tail bar — on **every tick, in both flag states**:

- `live_position_tracker.update_pnl(...)`
- `live_exit_handler.update_trailing_stops(df, current_index, ...)`
- `live_position_tracker.update_mfe_mae(...)`
- `_check_exit_conditions(df, current_index, current_price, ...)`
- `live_exit_handler.check_partial_operations(df, current_index, ...)`
- `_check_max_drawdown()`, `_check_inference_health()`, `_log_periodic_account_state()`

Only two call sites consume the gate's `entry_index` / `allow_entries`:
`_check_entry_conditions` and `entry_coordinator.process_legacy_short_entry`. Both are
entry paths. Stop-loss placement, re-placement (#1108), reconciliation, and account
monitoring are on other threads entirely and never consult the gate.

`_check_exit_conditions` receives `runtime_decision`, which between bar closes is the
*cached* last closed-bar decision rather than `None`. That is deliberate and correct: the
strategy-exit signal keeps flowing every tick (never delayed), and its value is the one
backtest would hold for that bar. The cache is `None` only for the ticks before the first
closed-bar evaluation of a session, during which SL/TP/trailing still fire normally.

### 2.3 Seam choice

Gating by *index* rather than by truncating the frame is the right seam:

- backtest itself calls `process_candle(df, i)` mid-frame with future rows present, so
  strategies must already be causal w.r.t. `index` — index-gating is input-equivalent to
  truncation and structurally identical to what backtest does;
- it keeps one feature-pipeline pass per tick (a truncated decision frame would force a
  second indicator/ML computation on the hot path);
- protective paths keep the exact frame/index/price they get today.

### 2.4 Residual divergences NOT closed by this change

Recording these so the PR is not over-claimed. D1 closes divergence #1 only. Still open:
quantization/dust (#2), margin interest (#3), funding (#4), warmup gate (#5), multi-position
(#6), resting-SL mechanics (#7), spread/impact (#8), data source (#9), partial-op cadence
(#10), sentiment freshness (#11). Parity claims remain scoped accordingly.

## 3. New test: `tests/unit/engines/live/test_closed_candle_parity.py`

The pre-existing 58 tests cover the gate's *internal* behavior (frontier resolution,
idempotence, flag inertness, protective-path tick-drive). None of them pinned the actual
claim — that backtest and gated live produce the same decision on the same bar. That test
now exists.

Construction:

- a deterministic, close-sensitive component strategy (up-bar → BUY, down-bar → SELL), so
  reference-price contamination is directly observable — the same mechanism the flip-rate
  study identified;
- **backtest side**: `StrategyRuntime.prepare_data` + `process(index, ctx)` per closed bar
  — this is literally what `BacktestEngine._get_runtime_decision` does;
- **live side**: the real `_trading_loop` with only its periphery stubbed (I/O, exits,
  entries, metrics). Gate resolution and `runtime_process_decision` run for real. Frames
  are fed as three ticks per bar, with a forming tail whose close deliberately walks
  *opposite* to the closed bar, so an ungated loop must disagree.

Assertions:

- `test_gated_live_matches_backtest_bar_for_bar` — the `(bar_time, direction, reference
  price)` sequences are equal, and each bar is evaluated exactly once. Measured: 10
  decisions, alternating `sell/buy`, exact match on all three fields.
- `test_ungated_live_diverges_from_backtest` — the control. Flag OFF, the sequences differ
  and at least one decision used a forming-bar reference price. Without this, the parity
  test could pass vacuously if the fixture stopped contaminating the tail.
- `test_decision_reference_price_is_the_bars_final_close` — the frozen-denominator
  property, stated directly.

## 4. Verification performed

- `pytest tests/unit/engines/live/test_closed_candle_gate.py
  test_closed_candle_gating_loop.py test_kline_buffer.py` — 58 passed.
- New parity module — 3 passed.
- Full unit suite — see PR body.
- `atb dev quality --changed` (never bare `atb dev quality`; that runs `black .` in place
  across the whole repo).

## 5. Recommendation

Land it. Flag stays OFF, so the merge is inert. The next step per the issue is a staging
soak with the flag ON, measuring realised flip-rate against the 43.2% study figure — that
is a separate change (staging env flag), not part of this PR.

Do **not** treat a parity-passing backtest as authorization to size up; the parity plan's
sizing guardrail (§1) still applies and the residual is biased optimistic.

---

# Review round (2026-08-25)

Four reviewers (two architecture, two code) across two waves. **No blockers**; the merge was
safe with the flag OFF. What follows is what had to be true before the flag could be flipped
ON in staging, plus two test defects.

## The parity test was weaker than it looked

Two of its three assertions could not fail for the right reason:

- `assert actual == expected[: len(actual)]` — the slice makes the comparison vacuous under
  the failure that matters. A reviewer demonstrated it: cut the stream to one bar and
  **1 decision out of 10 expected passes all three assertions**. Now `assert actual ==
  expected`, plus a separate `test_gated_live_evaluates_every_closed_bar`. Verified by
  replaying the jam: the old assertion returned True, both new ones return False.
- The control asserted `actual != expected[: len(actual)]`, which was true on **length
  alone** (30 ungated entries vs 10). It now compares content bar-by-bar on shared bars.

A third assertion pinned the wrong value — see "the half-dead parameter" below.

## Defects fixed

| # | Defect | Fix |
|---|---|---|
| P1 | **Frontier race.** The frame was copied, indicator/ML prep ran, and only then was `last_closed_bar_time` read. A bar closing inside that window let `min(frontier, df.index[-1])` certify the frame's own forming tail as closed — contamination reintroduced silently, reported as `decision_bar_closed=True`. | `KlineBuffer.snapshot()` returns frame and frontier from **one** lock acquisition; the coordinator records it on `_buffer_frontier`, cleared on REST fallback/resync/error. |
| P1 | **A jammed gate is invisible.** `_last_evaluated_bar` is an unbounded high-water mark; one bad timestamp silences evaluation permanently while the heartbeat keeps firing. Only symptom: an absent log line. | `ClosedCandleGate.stall_observation()` + a `CLOSED_CANDLE_GATE_STALLED` condition in `LatchedConditionMonitor` — a positive assertion, like #1103. |
| P1 | **The monotonic guard could raise.** `_closed_frontier` caught `TypeError` for mixed tz-awareness; `bar_time <= self._last_evaluated_bar` two lines later repeated the same comparison unguarded, so it escaped into the loop's handler and counted toward `consecutive_errors`. | Shared `_le` helper that fails closed; latched WARNING (was DEBUG, which understated a parity downgrade). |
| P2 | **Index desync.** `view.index` is a post-`dropna` frame position; `runtime.process` indexes `dataset.data` positionally and ignores the frame it is handed. | `runtime_index_for(bar_time)` resolves the bar by **timestamp** against the dataset actually indexed. |
| P2 | **Bar consumed before execution.** A raising `_check_entry_conditions` was swallowed, and the bar was already marked evaluated — up to a full timeframe of lost entry where the ungated path retried next tick. | `LoopSignalDecision.commit_bar`; the loop calls `mark_evaluated` **after** the entry block. |
| P2 | **Hot-swap kept stale state.** The retired strategy's decision drove signal-reversal exits and its high-water mark blocked the new strategy from deciding the current bar. | `_reset_closed_candle_gate()` on a successful swap. |
| P2 | **Write-only observability.** Nothing read the `decision_bar_*` keys, so the staging A/B and flip-rate soak had no data source. | Added to `_ML_SIGNAL_METADATA_KEYS`, so they land in `strategy_executions.ml_predictions`. |
| P2 | **No-WS path degraded silently.** Against a provider whose REST tail is already closed, frame-shape evidence lags one bar, making gating quietly *worse* for parity than leaving it off. | Latched WARNING naming the degradation. See "declined" below. |
| P2 | **The `_tail_closed` latch protected the flag, not the data.** A late duplicate still rewrote a closed bar's OHLCV through `_update_current_candle`, contradicting the latch's own comment. | Once closed, non-`x` events for that bar are dropped; a repeated `x: true` still applies (identical values, idempotent). |
| P3 | `stamp_decision_signal` unguarded on the flag-OFF path — a raising `pd.Timestamp` would break inertness. | Wrapped; never raises. |

## Declined, with reasons

**Bar-clock promotion of the REST tail.** Suggested as an alternative to warning about the
no-WS path. It is unsound: after a REST frame's tail bar passes its close time, the row still
holds the *partial* snapshot captured when it was fetched. Promoting it by wall clock would
certify incomplete data as final — precisely the defect this change exists to remove. The
degradation is now loud instead; the frame-shape fallback remains sound, only possibly stale.

**Strict-inequality frontier fix** (`frontier < df.index[-1]`), offered as a cheaper
alternative to the atomic snapshot. It does kill the race, but it also makes the frontier
*entirely redundant*: since the index is sorted, `frontier < tail` implies `frontier <=
df.index[-2]`, so the merged evidence collapses to frame shape alone and the `x: true` signal
stops contributing anything. The snapshot keeps that evidence and its lower latency, and with
the `_tail_closed` data fix above the clamp is now sound. Flagged for the record because it is
a deliberate divergence from the suggested fix.

## Correction to a review finding: `current_price` is NOT dead

The review reported `decision_price`/`decision_time` as dead, on the grounds that
`build_runtime_context` discards both. It discards `current_time` only. `current_price` flows
into `build_component_positions` and sets `ComponentPosition.current_price` — the value a
strategy reads for anti-pyramiding and correlation-aware sizing. Freezing it to the closed
bar's close is therefore a **real parity property**, matching how backtest values open
positions at that bar, not a no-op.

Acting on the finding as written would have removed a live parameter. The dead half
(`current_time` and its tz gymnastics) is gone; `current_price` stays, with the reasoning in
the test that pins it.

The reviewer's underlying point still landed: `test_decision_reference_price_is_the_bars_final_close`
was asserting on the argument passed *in*, not on what the strategy read. It now asserts on
the close recorded in the probe generator's signal metadata — the value the strategy actually
consumed.

## Precise protection claim

The blanket "protection is never delayed" is not literally true and has been corrected in the
PR body. Precisely:

- **Hard protective paths are tick-driven and ungated** in both flag states: stop-loss,
  trailing stops, exit checks, partial operations, PnL/MFE-MAE, drawdown guard, account
  monitoring, reconciliation.
- **Policy hydration** (trailing-stop / partial-exit / dynamic-risk config) and
  **strategy-signal exits** ride on the decision, so with the flag ON they refresh at *bar*
  cadence rather than tick cadence. Both are parity-correct — backtest behaves identically —
  so this is a parity gain, not a protection loss. But it is a real cadence change and should
  not be described as "unchanged".

## Known limitation for the prod flip

"Exactly once per bar" does not survive a restart: `_last_evaluated_bar` is in-memory, so a
process restart mid-bar can re-evaluate the current bar once. Harmless (entry guards prevent
duplicate positions) but it should be understood before the flag is flipped in production.

## Test coverage added

`tests/unit/engines/live/test_closed_candle_gate_hardening.py` — 28 tests, one class per
defect: atomic snapshot (including a concurrent-writer consistency loop), closed-bar
immutability, tz fail-closed, degraded-frontier warning, stall observation + monitor wiring,
hot-swap reset, commit-after-execution, cached-decision non-mutation, runtime index by
timestamp, and an end-to-end class driving a **real** `KlineBuffer` through the real
coordinator into the gate — the path no previous test exercised, which is why both P1s
survived a 61-test suite.

Three fixes were verified load-bearing by reverting the source and confirming the matching
test fails: the closed-bar rewrite (`assert 99.0 == 150.0`), commit-after-execution
(`assert 1 == 3`), and the tz guard (`TypeError: Cannot compare tz-naive and tz-aware`).

## Second correction round

**The gate's own unit test codified the unsound inference.**
`test_stale_buffer_frontier_ahead_of_frame_clamps_to_tail` justified the clamp as "a stale
REST fallback frontier proves the tail closed". That is exactly the reasoning that is unsound
for a non-atomic frame. Note the test did **not** fail after the race fix, because the fix
removes the unsound *input* at the source (the frontier is either captured with the frame or
None) rather than changing the clamp. So it would have sat there asserting a true outcome for
a false reason. Rewritten as `test_frontier_ahead_of_frame_clamps_to_tail`, driving the case
that can actually still arise — a frame truncated downstream by `dropna` — and stating the
invariant that makes the clamp sound.

**Parity test: only the count-blindness was real.** Two reviewers disagreed; both were right
about different properties. The test detects *wrong* decisions well (eight mutations caught)
and was blind to *missing* ones, because `expected[: len(actual)]` slices the count away. Only
that was changed. Re-ran four mutations against the final code — decision index reverted to
the forming bar, `_closed_frontier` using `df.index[-1]`, `searchsorted` losing its `-1`, and
`mark_evaluated` no-op'd — all still caught (3/4, 3/4, 3/4, 2/4 tests failing respectively).
The test was strengthened, not weakened.

**Flag-OFF is inert, not byte-identical.** `stamp_decision_signal` writes three keys into
`Signal.metadata` every tick in both modes, and `Strategy._extract_indicators` fans metadata
into the indicator snapshot, so they reach persisted rows. Nothing reads them to make a
decision — that is what inertness means here. Claims corrected in the gate docstring, the loop
test module docstring, and the PR body.

## Suite result

6008 passed, 1 skipped, 2 failed. **Both failures are pre-existing or environmental, neither
attributable to this PR:**

- `test_models_tft::test_create_model_lightgbm_dispatches_to_directional_classifier` — the
  known lightgbm failure; reproduced with the branch stashed.
- `test_indicators::test_indicators_performance` — wall-clock assertion (`< 3.0s`) that
  measured 6.5s under 4-way parallel load; **0.32s in isolation**. Timing flake in an
  untouched file.
