# Code review — closed-candle gating (#1106, parity decision D1)

**Branch**: `fix/closed-candle-gating` @ `cce8191f` (impl `49a0bb99` + test/docs `cce8191f`), rebased onto `f64ac4ca`.
**Reviewer**: code-review subagent, 2026-08-25.
**Verification run**: `pytest tests/unit/engines/live tests/unit/engines/shared` → **863 passed**. The four new/changed
test modules → 61 passed.

**Verdict: correct.** No blockers. The flag-OFF path is behaviourally inert modulo the intended
`Signal.metadata` stamping, the gate's `searchsorted` boundary handling is sound, and the
`KlineBuffer._tail_closed` latch is correct across the WS event orderings I could construct. Findings
below are should-fix / nit.

---

## Verified-good (so the findings are read in context)

- **Flag-OFF inertness.** `_evaluate_signal_decision` OFF-branch calls `_runtime_process_decision` with
  the identical `(df, current_index, self.current_balance, float(current_price), current_time)` tuple,
  unconditionally (it ignores `view.evaluate`), and returns `entry_index == current_index`,
  `allow_entries=True`. The only additions on that path are the metadata stamp and one buffer-lock read.
- **`_closed_frontier` boundary handling.** `searchsorted(frontier, side="right") - 1` is the correct
  "newest row at or below the frontier"; `pos < 0`, empty frame, and single-row frame are all guarded and
  tested. The `max(frame-shape, min(buffer_frontier, tail))` merge is sound in both directions: a stale
  buffer frontier can never *demote* a bar (max), and a buffer frontier ahead of the frame can never
  promote past the frame tail (min). The tz-mismatch `TypeError` is caught and degrades to frame-shape
  evidence.
- **`KlineBuffer._tail_closed`.** Latch-on-only for `event_ts == tail_ts` correctly survives a late
  forming duplicate; stale events (`event_ts < tail_ts`) return before touching it; a gap event returns
  before appending and correctly leaves an already-closed tail marked closed; `resync_from_rest` resets to
  forming, which is right for Binance REST (tail is the in-progress candle).
- **Parity test is not tautological.** `_reference_decisions` builds an independent `StrategyRuntime` over
  closed bars only; `_run_live` drives the real `_trading_loop` and the real `_runtime_process_decision`
  over frames whose forming tail deliberately contradicts the closed bar. Because
  `build_runtime_context` discards `current_price`/`current_time`, the *only* functional difference between
  the gated and ungated live runs is the evaluated index — and the control test shows that difference does
  change the decision. So the parity assertion is measuring a real property, not re-running one code path.

---

## Should-fix

### [P2] `mark_evaluated` consumes the bar before entry execution runs
`src/engines/live/trading_engine.py:_evaluate_signal_decision` (the `gate.mark_evaluated(view.bar_time)`
call) fires while still inside the decision helper, but `_check_entry_conditions` /
`process_legacy_short_entry` run ~60 lines later in `_trading_loop`. If entry execution raises (transient
exchange error, order-submission failure) the loop's `except` swallows it and the bar is already marked
evaluated — with gating ON there is no retry until the *next* bar closes, i.e. up to a full timeframe of
lost entry on a 1h/4h symbol. Ungated, the very next tick retried. Consider marking the bar evaluated only
after the entry pipeline returns, or recording a "decision made, entry not yet attempted" state.

### [P2] `_last_closed_bar_decision` is never invalidated
`src/engines/live/trading_engine.py` (init at ~line 834, replay in `_evaluate_signal_decision`). The cached
decision is replayed to `_check_exit_conditions` on every tick between closes — intended — but it is never
cleared. After `_apply_pending_strategy_update()` hot-swaps the strategy/model, the exit path keeps
receiving a decision produced by the *previous* strategy until the next bar close. Same for a long
safety-mode stretch or a stuck frontier. Clearing it in `_apply_pending_strategy_update` (and on
`finalize_runtime`) would bound the staleness to the swap itself.

### [P2] Gating ON can stop evaluating with no log
`ClosedCandleGate.resolve` returns `evaluate=False` silently whenever the frontier is `None` or at/below
`_last_evaluated_bar`. Combined with a REST resync that rewinds the buffer (`resync_from_rest` skips the
overwrite when the buffer tail is newer, leaving `_needs_resync=True`) or a frozen WS frame, the engine can
go a long time making no entry decisions and the logs look identical to normal between-bar quiet — the
`"decided on closed bar"` INFO simply stops appearing. Given the existing observability posture (#853), a
warn when no evaluation has occurred for > N intervals would make this failure mode visible.

### [P2] Observability-only stamping is unguarded on the flag-OFF path
`closed_candle_gate.stamp_decision_signal` calls `pd.Timestamp(bar_time)` with no try/except, on every tick,
in both modes. Live frames always carry a `DatetimeIndex` today, so this is latent — but if a degraded
provider ever hands back a non-datetime index, a pure-observability call would raise into `_trading_loop`'s
handler and count toward `consecutive_errors`, i.e. the flag-OFF path would no longer be inert. Wrapping the
body in `try/except Exception: logger.debug(...)` costs nothing and preserves the "merge is inert"
guarantee unconditionally.

### [P2] `view.index` is a position in the post-dropna frame, consumed against `_runtime_dataset`
`_trading_loop` does `df = self._prepare_strategy_dataframe(df)` (which returns `dataset.data` and stores it
as `_runtime_dataset`) and *then* `df = df.dropna(subset=essential_columns)`. `runtime_process_decision`
ignores the `df` argument and indexes `_runtime_dataset` positionally. The gate maps a *timestamp* frontier
to a position in the dropna'd frame — so if the essential-column dropna ever removes rows, `view.index`
lands on an earlier bar of the runtime dataset than the one the gate resolved. The gate's own comment
("the frontier bar itself may have been dropped by essential-column dropna upstream") shows the scenario is
contemplated. The desync is pre-existing in kind for the tail index, but the timestamp→position→other-frame
hop makes it easier to hit silently. Cheap mitigation: resolve the runtime index by `get_loc(bar_time)` on
the runtime dataset, or assert the two frames are the same length.

### [P2] Parity test: prefix slice masks a decision-count regression
`tests/unit/engines/live/test_closed_candle_parity.py:test_gated_live_matches_backtest_bar_for_bar`
asserts `actual == expected[: len(actual)]`. I instrumented the run: `len(expected) == 10` and
`len(actual) == 10`, so the slice is a no-op today — but it means a regression that evaluates only the first
bar (1 decision) would still pass, since `assert actual` and the uniqueness assertion both hold. The
accompanying comment ("the last reference bar has no live counterpart") is factually wrong for the current
fixture. Assert the exact count: `assert actual == expected`.

### [P2] Control test's primary assertion is vacuous
`test_ungated_live_diverges_from_backtest` asserts `actual != expected[: len(actual)]`. Ungated produces one
decision per tick — measured 30 entries against `expected`'s 10 — so `expected[:30]` is the full 10-element
list and the inequality is true on **length alone**, regardless of any content divergence. The control's real
weight is carried entirely by the later `contaminated` assertion. Either compare like-for-like (dedupe the
ungated stream to one decision per bar before comparing) or drop the misleading first assertion.

---

## Nits

### [P3] `decision_price` / `decision_time` are dead computations
In `_evaluate_signal_decision`, both are computed (including a tz-normalisation branch) and passed to
`_runtime_process_decision` — but `LiveStrategyRuntimeCoordinator.build_runtime_context`
(`src/engines/live/strategy_runtime.py:448`) and `BacktestEngine._build_runtime_context`
(`src/engines/backtest/engine.py:891`) both construct `RuntimeContext(balance=..., current_positions=...)`
and discard `current_price`/`current_time` entirely. The reference-price freeze the docstring describes is
achieved solely by the *index* into `_runtime_dataset`. The values are harmless but they are dead per
CODE.md, and the docstring ("with that bar's final close as the reference price") reads as if the argument
is the mechanism.

### [P3] Parity test's price assertions measure the discarded argument
Following from the above, the third tuple element on the live side (`float(current_price)`) and
`test_decision_reference_price_is_the_bars_final_close` both assert on a parameter the strategy never
consults. The direction/bar-timestamp comparison is the part with teeth. Reading the close out of the
runtime dataset at the evaluated index would make the price claim real.

### [P3] "Exactly once per bar" does not survive a process restart
`_last_evaluated_bar` is in-memory only. On restart the gate immediately evaluates `index[-2]` — a bar the
previous process may already have acted on — and can re-fire that bar's entry. The `ClosedCandleGate`
docstring's idempotence claim is scoped to "ticks, reconnects, and backfills" so it is not inaccurate, but
the restart case is worth an explicit note (or seeding the marker from the last persisted decision) before
the flag is flipped ON in prod.

### [P3] Parity test never exercises the ML path
The parity module uses a 12-bar frame and a hand-written close-delta signal generator. The module docstring
of `closed_candle_gate.py` and the changelog both make an ML-specific claim (`predicted_return`'s floating
denominator). A parity case over `ml_basic` with a stubbed ONNX prediction would pin the claim actually
being made.

### [P3] Small consistency / coverage items
- `kline_buffer.timeframe_to_ms`: `_TIMEFRAME_MS.get(timeframe) or None` — the `or None` only matters for a
  zero value that cannot occur; `.get(timeframe)` says the same thing.
- `kline_buffer.on_kline`: `bool(kline.get("x", False))` in two branches vs bare `if kline.get("x"):` in the
  `event_ts == tail_ts` branch — same semantics, inconsistent form.
- `_evaluate_signal_decision` reads `self._kline_buffer.last_closed_bar_time` (a lock acquisition) on every
  tick even with the flag OFF, where the value is used only for the `bar_closed` metadata field.
- No test covers the `TypeError` branch of `_closed_frontier` (tz-naive frontier vs tz-aware frame), a frame
  with duplicate timestamps, or a non-monotonic index (`Index.searchsorted` returns silently wrong positions
  on an unsorted index). All three are unlikely in practice; the tz one is the only branch with an explicit
  `except` and zero coverage.
