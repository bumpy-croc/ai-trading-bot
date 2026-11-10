# Architecture review — #1106 closed-candle gating (`fix/closed-candle-gating`)

**Reviewer**: architecture-reviewer subagent
**Date**: 2026-08-25
**Base**: `origin/develop` @ f64ac4ca (post #1108, #1073, #1102)
**Verdict**: architecturally sound, no blockers. Four should-fix items, several nits.
Flag defaults OFF and the OFF path is behaviourally inert apart from one metadata write,
so the merge is safe. The should-fix items all bite only once the flag is turned ON in
staging, and #3 blocks the staging soak the issue calls for.

---

## Blockers

None. Specifically: **every protective path is genuinely ungated in both flag states.**
See "Verified correct" §1.

---

## Should-fix

### S1. Closed-bar frontier is read non-atomically with the frame snapshot
`trading_engine._evaluate_signal_decision` → `closed_candle_gate.ClosedCandleGate._closed_frontier`

`df` is snapshotted in `_get_latest_data` (a locked copy of `KlineBuffer._df`), then the
whole prep pipeline runs (sentiment merge, `_prepare_strategy_dataframe` indicators/ML
features, dropna, readiness), and only then `_evaluate_signal_decision` reads
`self._kline_buffer.last_closed_bar_time`. The frontier is therefore *newer* than the frame
it is being applied to.

`_closed_frontier` clamps with `min(buffer_frontier, df.index[-1])`, so if the tail bar
receives its `x: true` (or a successor bar arrives) inside that window, the gate declares
`df`'s tail closed and evaluates it — but `df`'s tail row still holds the **pre-final**
OHLC snapshot taken before the closing tick landed. The decision is then made on a floating
close, `mark_evaluated` consumes the bar, and the true final close is never re-evaluated.

That is precisely the contamination D1 exists to remove, reintroduced silently on a
fraction of bars ≈ (prep duration / check interval), and it degrades without any signal —
`decision_bar_closed` will read `True`.

Suggested fix: have `KlineBuffer` expose one atomic snapshot (e.g.
`snapshot() -> tuple[pd.DataFrame, pd.Timestamp | None]` under a single lock acquisition)
and thread the frontier from `LiveMarketDataCoordinator.get_latest_data` alongside the
frame, instead of re-reading it later. A cheaper variant: only honour the buffer frontier
when the buffer's tail timestamp still equals `df.index[-1]` *and* the buffer revision
counter is unchanged since the copy.

### S2. Cached closed-bar decision survives a strategy/model hot-swap
`trading_engine._last_closed_bar_decision` (init at trading_engine.py:835)

`_apply_pending_strategy_update` runs at the top of `_trading_loop`, before the decision
block. With gating ON, `_last_closed_bar_decision` is not cleared, so the **retired**
strategy's decision keeps being replayed into `_check_exit_conditions` →
`StrategyExitChecker` for up to one full bar (an hour on 1h) after the swap, and the gate's
`_last_evaluated_bar` suppresses re-evaluation of the current closed bar by the new
strategy. Flag OFF, the new strategy takes effect on the very next tick.

Suggested fix: in `hot_swap_coordinator.refresh_strategy_dependencies` (or right after
`_apply_pending_strategy_update` returns True), clear `self._last_closed_bar_decision` and
decide deliberately whether to also clear `ClosedCandleGate._last_evaluated_bar` so the new
strategy re-decides the current closed bar immediately. Add a `reset()` method on the gate
rather than reaching into the private field.

### S3. `decision_bar_*` stamping is write-only — the P1.0 observability deliverable is not delivered
`closed_candle_gate.stamp_decision_signal`

The three keys are written into `Signal.metadata` and nothing ever reads them.
`extract_ml_predictions_from_signal` (src/tech/adapters/row_extractors.py:141) persists only
a whitelist (`_ML_SIGNAL_METADATA_KEYS`) and returns `{}` early unless the signal carries a
prediction; `strategy_executions` never sees `decision_bar_open_time` /
`decision_bar_close_time` / `decision_bar_closed`. A repo-wide grep finds these keys only in
the new tests.

That means the staging(ON)/prod(OFF) A/B the module docstring and the landing note both
promise — "measure realised flip-rate against the 43.2% study figure" — has no data source.
Suggested fix: add the three keys to the persisted set for `strategy_executions` (or a
dedicated small column/JSON field), and pin it with a test that asserts they reach the DB
row builder, not just the metadata dict.

### S4. Without a WS kline buffer, the frontier can lag a full bar
`ClosedCandleGate._closed_frontier` (frame-shape branch)

When `self._kline_buffer is None` the only evidence is `df.index[-2]`. That is correct for
providers whose REST tail is the in-progress candle (Binance klines), but a provider or
cache path that returns **closed bars only** makes `df.index[-1]` already closed — the gate
then decides one whole bar late, i.e. entries are systematically a bar behind backtest.
This is the one shape where gating ON is worse than OFF for parity, and it fails silently.

Suggested fix: derive closed-ness for the REST path from the bar clock rather than frame
shape (`df.index[-1] + timeframe_to_ms(timeframe) <= now_utc` ⇒ tail closed), or refuse to
enable gating when no WS buffer is active and log a warning once. `timeframe_to_ms` is
already available in `kline_buffer`.

---

## Nits

- **N1** `closed_candle_gate`: `bar_time: Any`, `buffer_frontier: Any`, `decision: Any`.
  These are `pd.Timestamp | None` and `TradingDecision | None`. CODE.md (Types) says avoid
  `Any` on public APIs; the `Any` here also hides the tz-comparability hazard S1/N2 turn on.
- **N2** `_closed_frontier` swallows the mixed-tz `TypeError` at `logger.debug`. Mixed
  tz-awareness between the buffer and the frame is a real data-integrity defect that
  silently downgrades the parity guarantee to frame-shape evidence; log it at WARNING once
  (latched), per CODE.md "errors not silenced without explicit justification".
- **N3** The docstring calls flag OFF "byte-identical" behaviour, but OFF now mutates
  `Signal.metadata` on every tick via `stamp_decision_signal`. Harmless today (all known
  consumers read by key), but the claim should be softened to "decision-identical".
- **N4** The flag is resolved once in `__init__`, so rollback requires a process restart.
  That matches `enable_exposure_governor` / `enable_macro_event_guard`, so it is a
  convention, not a defect — but the staging runbook should state that turning gating OFF
  is a restart, unlike `entry_pause`.
- **N5** `_evaluate_signal_decision` takes six positional params and returns a bare
  3-tuple `(decision, entry_index, allow_entries)`. A small frozen dataclass (mirroring
  `GateDecision`) would keep the call site self-documenting and stop the tuple from drifting.
- **N6** Informational, not a regression: entries are now closed-bar while **scale-ins**
  (`check_partial_operations`) remain tick-driven on the forming bar. Scale-ins are
  price-threshold driven, not signal driven, so the gate genuinely does not apply — this is
  residual divergence #10 in the plan, correctly left open by the landing note.

---

## Verified correct

**1. Protective paths ungated in both flag states.** Read post-rebase `_trading_loop`
(trading_engine.py:1789–1866). All of `live_position_tracker.update_pnl`,
`live_exit_handler.update_trailing_stops`, `update_mfe_mae`, `_check_exit_conditions`,
`check_partial_operations`, `_update_performance_metrics`, `_check_max_drawdown`,
`_check_inference_health`, `_log_periodic_account_state` are called with
`current_index` / `current_price` (the forming tail) unconditionally. Only
`_check_entry_conditions` and `process_legacy_short_entry` consume `entry_index` /
`allow_entries`, and both are entry paths. `_system_halt_enforcer.check()`,
`_latched_condition_monitor.check()`, `_drain_pending_fill_exits()` run before the decision
block. Reconciliation, order tracking, SL placement/re-placement (#1108) and account
monitoring are on other threads and never touch the gate.

**2. The one skipped safety call is provably redundant.** With gating ON and no new closed
bar, `_check_entry_conditions` — and therefore its in-line `state._refresh_drawdown_gate()`
(#807 pattern) — does not run. That refresh exists solely to close the one-iteration leak
where an entry executes before close-only latches; with no entry evaluated there is no leak,
and `_check_max_drawdown()` still runs every tick further down the loop. No protection delay.

**3. Cached-decision replay is safe from mutation.** Traced every consumer of the replayed
`TradingDecision`: `exit_coordinator.check_exit_conditions` → `exit_handler.check_exit_conditions`
→ `_check_strategy_exit` → `StrategyExitChecker.check_exit`, which reads
`runtime_decision.metadata.get("ignore_signal_reversal")`, `.signal.direction`, `.regime`
and constructs fresh `ComponentPosition`/`ComponentMarketData` objects. No write to the
decision or its signal anywhere on the exit path. `stamp_decision_signal` is the only
mutator and runs only on freshly produced decisions. Reuse is read-only and idempotent.

**4. Cache is populated on the first gated tick, not after a bar wait.** On the first
iteration `_last_evaluated_bar is None` and the frame-shape frontier `df.index[-2]` exists,
so evaluation fires immediately; signal-reversal exits are `None`-gated only for ticks
before that first evaluation (and `safety_mode` ticks, where the loop already passes `None`
today). No post-restart protection gap.

**5. Thread safety holds as claimed.** `ClosedCandleGate` and `_last_closed_bar_decision`
are touched only from `_trading_loop` (grep of `trading_engine.py` confirms lines 830/835,
1789, 1826, 1859 and the two methods — no other thread, no callback path). `KlineBuffer`
writes `_tail_closed` only inside `with self._lock` in `on_kline` and `resync_from_rest`;
`last_closed_bar_time` takes the same non-reentrant `threading.Lock` and is not called from
any already-locked method (checked all five `with self._lock` sites), so no re-entrancy
deadlock. The `if kline.get("x"): self._tail_closed = True` latch on the `event_ts == tail_ts`
branch correctly prevents a late duplicate forming event from regressing a closed bar, and
the new-bar branch resets it from that event's own `x`.

**6. Frontier logic across reconnect / resync / backfill / gap.**
- `resync_from_rest` sets `_tail_closed = False` — correct, REST tails are in-progress.
- Gap detection returns before appending, leaving `_tail_closed` matched to the retained
  tail — no false "closed" claim.
- `mark_evaluated` + `bar_time <= self._last_evaluated_bar` makes evaluation monotonic:
  a rewound/backfilled frame can never re-trigger an already-decided bar (pinned by
  `test_backfill_to_older_data_does_not_reevaluate`).
- `df.index.searchsorted(frontier, side="right") - 1` correctly lands on the newest row at
  or below the frontier when the frontier row itself was removed by the essential-column
  `dropna`; `pos < 0` is guarded. The `df.index[-2]` evidence stays valid after dropna
  because "a successor row exists" is what proves closure, not adjacency.
- Mixed tz-awareness raises `TypeError` from `min()` and falls back to frame-shape evidence
  rather than crashing the loop (see N2 on the log level).

**7. SL/TP arithmetic is not affected by the index/price split.** With gating ON,
`_check_entry_conditions` receives `index=entry_index` (closed bar) but `current_price` =
live tail close. `entry_handler._calculate_sl_tp` anchors on `current_price` and computes
percentage offsets via `PriceTargetCalculator.sl_tp(entry_price=current_price, ...)`, so the
stop distance is measured from the price the order actually fills near — no closed-bar/live-price
mixing in the risk arithmetic. `index` is used only for indicator/ML/sentiment extraction
and correlation control, all of which *should* come from the decision bar. Formula check:
long SL = entry_price × (1 − sl_pct) with entry_price = live price ⇒ realised risk fraction
= sl_pct as intended, unchanged from flag OFF.

**8. Rebase interaction claims hold.** #1108's changes live in
`execution_engine`/`stop_loss_manager`/`order_tracker` (order-poll layer, other threads);
the four `trading_engine.py` lines it added are outside the decision block. #1073 hydration
runs inside `runtime_process_decision`, downstream of the index the gate picks — gating
changes which index, never whether hydration runs. Confirmed by reading the post-rebase loop.

**9. Tests and quality gates.** 61 tests pass across the four modules
(`test_closed_candle_gate.py`, `test_closed_candle_gating_loop.py`, `test_kline_buffer.py`,
`test_closed_candle_parity.py`). The parity module is the valuable addition: it drives the
real `_trading_loop` and the real `StrategyRuntime`, and `test_ungated_live_diverges_from_backtest`
is a proper control that stops the parity assertion passing vacuously. Ruff and black clean
on all changed files.
