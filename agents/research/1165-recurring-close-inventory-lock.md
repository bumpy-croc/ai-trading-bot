# #1165: the recurring `CLOSE_INVENTORY_LOCKED` loop on ETHUSDT

Read-only investigation, 2026-09-14 ~21:10 UTC. Prod DB (`main`), code at
`origin/main` = `3a73bfe5` ("Promote to production: develop @ b46fe700"), which
contains #1108, #1112 and #1126.

**Verdict in one line:** the inventory is *not* locked by an untracked or
external order, and the close path *does* cancel the resting stop first. The
close aborts because the free-base-balance read used to size it is taken
milliseconds after that cancel, and Binance's cross-margin wallet endpoint has
not yet released the `locked` amount — so the close sizes itself off a
pre-cancel snapshot, shrinks to dust, and trips #1108's 98% gate.

Hypothesis (a) — the trading loop's own exit re-evaluation — is **CONFIRMED**.
Hypothesis (b) — a close that never cancels the resting stop — is **REFUTED**.

---

## 1. What is firing, and why every ~66 s

`strategy_executions` shows an unbroken run of `signal_type=exit`,
`action_taken=closed_position`, `reasons[0] = "Stop loss"` at the trading-loop
cadence, from 20:18:58 to 20:57:58 UTC (and 08:32–08:57 on 2026-08-27). It is
the strategy loop, not the reconciler, not a self-check.

The stop test is `src/engines/live/execution/exit_handler.py:714-717`:

```python
if self.use_high_low_for_stops and candle_low is not None and candle_high is not None:
    if position.side == PositionSide.LONG:
        # For long SL, check if candle_low breached the stop
        return candle_low <= position.stop_loss
```

Position 29 (ETHUSDT LONG, entry 2513.57 @ 11:00:07) has
`trailing_stop_activated = t` and `stop_loss = trailing_stop_price =
2568.41705`. `update_trailing_stops` ratcheted the stop up off the *current,
still-forming* candle's high. That same candle's low is below 2568.417, so
`candle_low <= position.stop_loss` is true, and stays true on every iteration
for the remainder of that candle. It re-arms on any later candle whose low dips
back through the (only-ever-rising) trail — which, with the trail sitting ~0.1%
under market, is most of them.

#1108's own issue body reached the identical conclusion from the 2026-08-19
data: "~66s is the trading loop (`DEFAULT_CHECK_INTERVAL = 60`), not the
reconciler... `strategy_executions` shows exit reason `"Stop loss"` on *every*
loop iteration from 15:44:12 to 15:59:44, then `hold_position` at 16:00:49 — on
the hour, when the candle whose low had dipped through the trailing stop rolled
over." Both episodes here sit inside a single hourly candle (08:32–08:57 on
Aug 27; 20:18–20:58 today).

In a backtest this is invisible: the bar closes the position on first
evaluation. Live, the loop re-evaluates the same forming candle every ~66 s, so
**any close that fails re-fires the identical exit decision for the rest of the
candle, and again on the next one.** The
engine already knows this — `execution_engine.py:171-173` says so in a comment
("The exit signal that triggers a close re-fires every trading-loop iteration,
so an un-latched abort would page once per ~66s").

## 2. The close path DOES cancel the resting stop

`src/engines/live/execution/exit_coordinator.py:415-455` is the only route to
`ExecutionEngine._close_live_order` (verified: `_close_live_order` has exactly
one caller, `execution_engine.py:866`; `live_exit_handler.execute_exit` has
exactly one caller, `exit_coordinator.py:457`). That block is:

```python
if (state.enable_live_trading and state.exchange_interface and position.stop_loss_order_id):
    pre_filled = state._stop_loss_filled_quantity(position)
    if pre_filled is None or pre_filled > BORROW_DUST_EPSILON:
        return                      # defer — no close attempted
    if not state._cancel_stop_loss_order(position):
        return                      # unconfirmed cancel — no close attempted
    protective_order_cancelled = True
    post_filled = state._stop_loss_filled_quantity(position)
    if post_filled is None or post_filled > BORROW_DUST_EPSILON:
        return                      # no close attempted
```

Every escape hatch `return`s **without** reaching `_close_live_order`. Emitting
`CLOSE_INVENTORY_LOCKED` therefore *proves* the cancel ran and the exchange
confirmed it. Hypothesis (b) as stated is refuted by control flow alone.

Four independent pieces of evidence agree:

1. **Stop id churn with a 3-second signature.** `positions.stop_loss_order_id`
   moved `49814050339` → `49814325117` → `49814364654` between 20:50 and
   20:59. The write of `49814325117` landed at `last_update = 20:56:56.56`,
   **3 s after** the abort at `20:56:53.52`. A new stop id immediately after a
   failed close is the signature of `_reprotect_position`
   (`exit_coordinator.py:468-475`), which runs *only* when
   `protective_order_cancelled is True`.
2. **The re-placement succeeds, so the balance is fresh a moment later.**
   `BinanceProvider.place_stop_loss_order` carries the *same* holdings cap and
   the same `HOLDINGS_CAP_MIN_RATIO` gate (`binance_provider.py:2111-2137`) and
   emits `STOP_LOSS_PLACEMENT_FAILED` / `UndersizedProtection` into
   `system_events` when it trips. There are **zero** such rows today. So a few
   hundred ms after the abort — at reprotect time — the free base was full.
   Only the close-path read, taken immediately post-cancel, sees it locked.
3. **The Aug 27 control case.** The single cycle where the close *succeeded* —
   order 7442, `FULL_EXIT`, quantity 0.0037, `filled_quantity` 0.0037, filled
   `08:59:03` — came 71 s (one cycle) after four consecutive
   `STOP_LOSS_PLACEMENT_FAILED` (`-1111`, price precision) at 08:57:50–08:57:59
   and a `RECONCILE_CRITICAL` "ETHUSDT unprotected — SL re-placement failed".
   That is the one cycle in which **no stop was resting at all**. A full-size
   sell filled instantly. The inventory was never genuinely unsellable.
4. **No orphan is needed to explain anything.** `reconciliation_audit_events` is
   empty for today, and there are no `UNPROTECTED` alerts — the reconciler keeps
   finding a healthy, tracked, resting stop every cycle. There is exactly one
   stop, it is ours, and it is tracked.

## 3. The actual defect

`ExecutionEngine._free_base_for_close`,
`src/engines/live/execution/execution_engine.py:1294-1320` — and specifically
its stated assumption:

> ``free`` is the amount available to sell — reported identically in spot and
> margin mode, **and the resting stop-loss is cancelled before the close (#710),
> so its previously locked inventory is free again by the time this reads it.**

That last clause is false. The call chain is
`get_balance(base)` → `BinanceProvider._call_get_account()` →
`client.get_margin_account()` (`binance_provider.py:931-949`; no local cache —
it is a live REST call). Binance's `/sapi/v1/margin/account` is
eventually-consistent and does not reflect a just-confirmed `DELETE
/sapi/v1/margin/order` within the few milliseconds between the two calls.

(This does not contradict #1108's "a stale read is not the cause". That ruled
out staleness in the per-*order* `get_order` reads on the SL-verification path,
which are live and uncached. This is a different endpoint — the *wallet*
aggregate — and a different consistency guarantee.)

The close therefore reads the **pre-cancel** wallet:

```
free = 0.00027109        locked ≈ 0.0034 (the stop we just cancelled)
```

Then, in `_close_live_order` (`execution_engine.py:1249-1296`):

```
intended_quantity = 0.0035008407259148715
free_base         = 0.00027109        ->  quantity = 0.00027109
_normalize_quantity(floor=True)       ->  quantity = 0.0002       (LOT_SIZE 0.0001)
0.0002 < 0.0035008407 * 0.98 = 0.0034308  ->  ABORT
```

which is the observed row verbatim:

```
{"symbol": "ETHUSDT", "intended_quantity": 0.0035008407259148715,
 "sellable_quantity": 0.0002, "free_base_balance": 0.00027109,
 "min_ratio": 0.98, "consecutive_aborts": 6}
```

The loop is then self-sustaining and perfectly stable:
abort → reprotect places a fresh stop (balance now settled) → next iteration
re-fires the same trailing-stop exit → cancel → stale read → abort.

**Why `free_base` is bit-identical for 40 minutes across a process restart:**
it is not a flaky race value, it is a steady state — `total_holdings −
one_resting_stop`. Same on Aug 27 (`0.00017459`, unchanged for 25 minutes). The
relative fraction differs (2.7% vs 5.7%) only because the position sizes
differed; the absolute residue is leftover dust from the prior position.

This is fail-safe: #1108 refuses to sell a fraction and book a full close, the
position stays protected (reprotect re-places the stop each cycle), and no
capital is lost per cycle. But it will not self-clear, and it latches
close-only within three cycles — exactly the 17-day halt of #1121.

## 4. Recommended fix

**Primary (fixes the root cause).** Do not let a stale wallet snapshot shrink a
close. Thread the "we just cancelled a resting stop" fact into the close and
either skip or retry the cap:

- Add a `stop_just_cancelled: bool = False` parameter to
  `ExecutionEngine.execute_exit` / `_close_live_order`, set from
  `exit_coordinator.execute_exit_locked`'s `protective_order_cancelled`.
- When set, re-read the balance with a short bounded poll before deciding —
  e.g. up to 3 attempts, ~250 ms apart, stopping as soon as
  `free >= intended_quantity * HOLDINGS_CAP_MIN_RATIO`. Abort only if the budget
  expires. This keeps #1108's guarantee intact (a genuinely locked inventory
  still aborts, ~750 ms later) while removing the false positive entirely.
- Equivalent and slightly cleaner: poll `get_margin_account_asset(base)` in
  `exit_coordinator` right after the confirmed cancel, until `locked` drops or a
  ~1 s budget expires, and only then call the exit handler. That keeps the
  settlement wait in the layer that owns the cancel.

Either way, **update the `_free_base_for_close` docstring** — the false
assumption in it is what made this invisible for three weeks.

**Secondary (stops the storm even if the primary regresses).** Latch the exit
decision per position + candle. `_check_stop_loss` re-evaluates the *forming*
candle every iteration, so one ratcheted trailing stop produces an unbounded
re-fire. Evaluating stops on closed candles only (backtest parity) — or
suppressing a repeat exit decision for the same `(position, candle_open_time,
reason)` — turns "every 66 s forever" into "once per candle".

## 5. Other defects found while tracing (file separately, not blockers)

- **`LiveExitHandler._execute_partial_exit` bypasses the whole cancel-then-close
  sequence.** `src/engines/live/execution/exit_handler.py:1112` calls
  `self.execute_exit(...)` directly when partials fully close a position. That
  skips `exit_coordinator.execute_exit_locked` entirely — so it skips the #710
  cancel block *and* the `#703` base-asset lock. A position closed out by
  partials submits a market close with its stop still resting: a guaranteed
  `-2010` (or, post-#1108, a guaranteed `CLOSE_INVENTORY_LOCKED`). Latent —
  position 29 has `partial_exits_taken = 0` — but real.
- **`update_trailing_stops` never moves the exchange stop.**
  `src/engines/live/execution/exit_handler.py:780-831` updates
  `position.stop_loss` (memory + DB) and nothing else. The resting exchange
  order stays at the original entry stop while the engine believes protection
  sits at 2568.417. The reconciler's step-2 audit checks the order's *status*,
  never its price, so this reads as protected forever. This is both a real
  protection gap and the reason the engine-side stop keeps firing while the
  exchange-side stop never triggers.
- **`_place_missing_stop_loss` does not track the order it places.**
  `src/engines/live/reconciliation.py:4560-4566` sets
  `position.stop_loss_order_id = new_sl_id` but — unlike
  `stop_loss_manager.reprotect` — never calls
  `order_tracker.track_order(new_sl_id, symbol)`. A stop placed by that path has
  no real-time fill/cancel detection.

## 6. Live-operations flag (separate from #1165)

**Prod has been silent since 2026-09-14 20:59:04 UTC.** As of 21:10:29 UTC:
last `strategy_executions` row 20:57:58, last `system_events` row 20:58:00, last
`account_history` row 20:59:04, `positions.last_update` frozen at 20:59:04.
Session 21 has no `end_time`, so it was not a clean shutdown. Position 29 is
still OPEN (0.0035 ETH LONG, entry 2513.57) with `stop_loss_order_id =
49814364654`. Needs a liveness check by someone authorised to touch Railway —
this investigation was read-only and did not act.

## 7. Evidence appendix (all read-only, `SET default_transaction_read_only = on`)

- `system_events` where `error_code='CLOSE_INVENTORY_LOCKED'`: 58 rows,
  2026-08-27 08:32:05 → 2026-09-14 20:58:00. Two episodes only — 24 rows on
  Aug 27, 34 today from 20:18:58.
- Today's episode: `free_base_balance` = `0.00027109` on **every** row.
  Aug 27's: `0.00017459` on every row.
- `positions` id 29: ETHUSDT LONG, OPEN, entry 2513.57 @ 2026-09-14 11:00:07,
  `quantity` 0.0035, `stop_loss` = `trailing_stop_price` = 2568.41705,
  `trailing_stop_activated` = t, `current_price` 2570.62 (stop **below**
  market — the exchange stop is not triggerable; only the candle-low test fires).
- `orders`: only two rows are relevant — 7443 (today's ENTRY, FILLED 0.0035)
  and 7442 (Aug 27 FULL_EXIT, **filled 0.0037** at 08:59:03). **No FULL_EXIT row
  exists for today** — consistent with the abort returning before the
  order-journal write at `execution_engine.py:1305`.
- `reconciliation_audit_events`: zero rows since 2026-09-14 10:00.
- `railway logs` unavailable ("Unauthorized" — the recurring staging/prod CLI
  auth issue), so no application-log corroboration; every conclusion above rests
  on Postgres + `origin/main` source.
