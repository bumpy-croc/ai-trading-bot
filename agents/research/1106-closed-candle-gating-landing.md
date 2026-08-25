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
