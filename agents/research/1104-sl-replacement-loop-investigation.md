# #1104 — Stop-loss re-placement loop on ETHUSDT (2026-08-19): root-cause investigation

Status: root cause identified with prod evidence.
Investigated: 2026-08-24. Code read at `origin/develop` @ `f7d87e9e`; prod ran `main` @ `8da478a3`
(the paths below are byte-identical on both for every file cited except where noted).
Prod DB accessed strictly read-only (`SET default_transaction_read_only = on;`).

---

## 1. Verdict

**H1 (degraded user stream causes false/stale order-state reads) — PARTIALLY CONFIRMED, but not as
stated.** The order-state reads themselves are correct and fail-closed. What the degraded stream did
was *enable* a detection path (`OrderTracker` REST polling) and drive extra off-cadence reconciler
cycles. The actual defect is a **read-modify-write race on
`LivePosition.stop_loss_order_id` between the deliberate pre-close SL cancel and the
OrderTracker cancel callback**, which (a) fires a false `STOP_LOSS_CANCELLED`/UNPROTECTED alert for a
cancel *we* issued, and (b) nulls the tracked SL id so the reconciler places a **second**
stop-loss while the first still rests.

**H2 (genuine exchange rejection) — RULED OUT as the cause of the SL dying.** The stops did not die
on the exchange spontaneously; **we cancelled them ourselves**, once per trading-loop iteration, from
the `#710` pre-close cancel path. An exchange rejection *does* occur in this incident, but on the
**close order**, not the stop — and it is a downstream symptom (§4).

**H3 (quantity/precision drift, LESSONS §1.1) — RULED OUT.** Evidence in §6.

**The capital-risk headline is not the alert. It is this:** while the loop ran, an orphaned resting
stop-loss locked the position's entire base inventory, so every close attempt was silently
**re-sized down to dust** (0.00009419 ETH ≈ $0.20 of a 0.0087 ETH ≈ $18 position) by the free-balance
cap at `src/engines/live/execution/execution_engine.py:1181-1191`. Had that dust order filled, the
engine would have booked a **full position close** while ~99% of the inventory remained on the
exchange. It did not fill only because $0.20 is under Binance's `MIN_NOTIONAL`.

---

## 2. Prod evidence (all read-only queries against the production DB)

### 2.1 `system_events` — the loop

```
2026-08-19 15:41:54.462  ALERT   critical connectivity  USER_WS_DEGRADED
2026-08-19 15:46:28.257  ERROR   critical order_tracker STOP_LOSS_CANCELLED  order 49075377000
2026-08-19 15:46:32.009  WARNING info     connectivity  USER_WS_RECOVERED     <-- 4s AFTER event #1
2026-08-19 15:47:34.911  ...     order 49075492446
2026-08-19 15:48:41.548  ...     order 49075591432
2026-08-19 15:49:46.922  ...     order 49075673082
2026-08-19 15:50:54.956  WARNING warning  reconciler    RECONCILE_HIGH
2026-08-19 15:52:05.040  ...     order 49075821431
2026-08-19 15:53:10.398  ...     order 49075907043
2026-08-19 15:54:16.510  ...     order 49075984024
2026-08-19 15:55:22.355  ...     order 49076049798
2026-08-19 15:56:28.224  ...     order 49076124513
2026-08-19 15:57:34.501  ...     order 49076214926
2026-08-19 15:58:39.629  ...     order 49076281975
2026-08-19 15:59:46.043  ...     order 49076317709
2026-08-19 17:19:38.547  ...     order 49076355916
```

**Correction to the issue text:** `USER_WS_RECOVERED` fired at **15:46:32**, i.e. the user stream was
back to WS-primary *four seconds after the first* `STOP_LOSS_CANCELLED` and stayed primary for the
remaining twelve. The WS degradation therefore cannot be sustaining the loop. This is the first
piece of evidence that H1-as-stated is wrong.

**Base rate:** `STOP_LOSS_CANCELLED` has fired **13 times ever**, all on 2026-08-19.
`USER_WS_DEGRADED` has fired **29 times** since 2026-07-05. WS degradation is therefore *not
sufficient*; something else specific to 2026-08-19 15:44–16:00 is required.

### 2.2 `strategy_executions` — the missing ingredient (decisive)

```sql
SELECT timestamp, signal_type, action_taken, price, reasons::text
FROM strategy_executions
WHERE symbol='ETHUSDT' AND timestamp BETWEEN '2026-08-19 15:44' AND '2026-08-19 16:02';
```

| timestamp | signal | action_taken | price | reason[0] |
|---|---|---|---|---|
| 15:44:12.80 | exit | closed_position | 2082.78 | **"Stop loss"** |
| 15:45:19.12 | exit | closed_position | 2084.21 | **"Stop loss"** |
| 15:46:25.84 | exit | closed_position | 2082.96 | **"Stop loss"** |
| 15:47:32.60 | exit | closed_position | 2080.10 | **"Stop loss"** |
| 15:48:39.71 | exit | closed_position | 2087.26 | **"Stop loss"** |
| 15:49:45.23 | exit | closed_position | 2083.46 | **"Stop loss"** |
| 15:50:56.19 | exit | closed_position | 2087.27 | **"Stop loss"** |
| 15:52:02.53 | exit | closed_position | 2083.68 | **"Stop loss"** |
| 15:53:08.38 | exit | closed_position | 2081.01 | **"Stop loss"** |
| 15:54:14.49 | exit | closed_position | 2081.16 | **"Stop loss"** |
| 15:55:20.22 | exit | closed_position | 2078.20 | **"Stop loss"** |
| 15:56:26.33 | exit | closed_position | 2082.57 | **"Stop loss"** |
| 15:57:32.01 | exit | closed_position | 2084.94 | **"Stop loss"** |
| 15:58:37.79 | exit | closed_position | 2086.15 | **"Stop loss"** |
| 15:59:44.19 | exit | closed_position | 2087.03 | **"Stop loss"** |
| **16:00:49.74** | exit | **hold_position** | 2086.15 | **"holding_position"** |
| 16:01:53.48 | exit | hold_position | 2091.03 | "holding_position" |

Two things fall out of this table and they are the crux of the whole incident:

1. **The engine issued an exit decision on EVERY trading-loop iteration**, once per ~66 s
   (`DEFAULT_CHECK_INTERVAL = 60`, `src/config/constants.py:244`, wired at
   `src/engines/live/trading_engine.py:280`; 60 s + REST latency = the observed 66 s).
   The "~66 s matches the reconciliation cadence" reading in the issue is **wrong** — the periodic
   reconciler runs at `DEFAULT_RECONCILIATION_INTERVAL_SECONDS = 120`
   (`src/config/constants.py:433`, wired without override at `src/engines/live/startup.py:435-446`).
   The 66 s cadence is the **trading loop**, not the reconciler.
2. **The signal flips to `hold_position` exactly at 16:00**, on the hour. The strategy evaluates the
   stop against the *in-progress* candle's low (`use_high_low_for_stops`, see
   `src/engines/live/execution/exit_handler.py:243-260` `_apply_stop_loss_gap_pricing` and the
   strategy-side stop check). Once the 15:00 hourly candle's low dipped through the trailing stop,
   **every** subsequent iteration inside that same candle re-evaluated the same low and re-emitted
   "Stop loss" — for 15 consecutive minutes. The new candle at 16:00 cleared it. The isolated
   17:19:38 event is the same thing recurring inside the 17:00 candle, and that one finally closed.

So the trigger condition is: *an intra-candle stop touch that the live close cannot service*,
repeating once per loop for the remainder of the candle.

### 2.3 `positions` id=25 — the position

```
symbol ETHUSDT | side LONG | status CLOSED | entry_price 1918.51 | entry_time 2026-07-21 17:55:02
quantity 0.00870000 | size 0.20 | original_size 0.20 | current_size 0.20 | partial_exits_taken 0
stop_loss 2072.36120000 | trailing_stop_activated t | breakeven_triggered t
stop_loss_order_id 49076355916 | entry_order_id 48553879892
current_price 2071.98 | last_update 2026-08-19 17:20:56
```

### 2.4 `orders` — the smoking gun

Only three order rows exist in the whole window:

```
id 7433 | pos 25 | FULL_EXIT | status UNKNOWN   | exchange_order_id (null) | qty 0.00009419
       | created 2026-08-19 15:49:46.800 | client_order_id atbx_1a01ab6dae0_f67129af
id 7434 | pos 25 | FULL_EXIT | status CONFIRMED | 49079988284 | qty 0.00860000 @ 2071.98
       | created 2026-08-19 17:20:52.237  (the close that finally worked)
id 7435 | pos 26 | ENTRY     | status FILLED    | 49079995684 | qty 0.00810000 @ 2071.35
```

**Do the arithmetic:**

```
tracked position quantity                     = 0.00870000 ETH   (approx; true holding 0.00869419)
free base at 15:49:46 (order 7433's quantity) = 0.00009419 ETH
=> locked base                                = 0.00869419 - 0.00009419 = 0.00860000 ETH
successful close quantity at 17:20:52         = 0.00860000 ETH   <-- exact match
```

`0.0086` is **exactly** the size of one resting stop-loss order. At 15:49:46 a stop-loss for the full
position size was **still resting on the exchange, locking the entire base inventory**, while the
engine was simultaneously submitting a close it believed was unobstructed. That resting stop was not
the one the exit path had just cancelled — it was an orphan the engine had lost track of.

`0.00009419 × $2083 = $0.196` — far below Binance's $5 `MIN_NOTIONAL`, so the order was rejected and
journalled `UNKNOWN` with no `exchange_order_id`. That is why 12 of the 13 iterations produced no
order row at all and this one produced a dead one.

### 2.5 `reconciliation_audit_events`

Zero rows in the window. #1097's `ExchangeOrderError` capture shipped **after** the incident
(`main` @ `8da478a3`, promoted 2026-08-24), so it recorded nothing here. It is live now and will
capture the Binance code if this recurs — but per §4 the informative capture would be on the
**close** path, and `_record_order_error` is only wired into
`BinanceProvider.place_stop_loss_order`, not into the close path. See recommendation R4.

---

## 3. The traced code path

### 3.1 What emits `STOP_LOSS_CANCELLED`

`src/engines/live/execution/order_fill_coordinator.py:196-232` —
`LiveOrderFillCoordinator.handle_stop_loss_cancelled()`:

```python
196  def handle_stop_loss_cancelled(self, order_id: str, symbol: str) -> bool:
207      for pos_key, position in state.live_position_tracker.positions.items():
208          if getattr(position, "stop_loss_order_id", None) == order_id:
209              matched_key = pos_key
213      state.live_position_tracker.set_stop_loss_order_id(matched_key, None)   # <-- CLOBBER
214      message = (f"Stop-loss order {order_id} for OPEN {symbol} position {matched_key} was "
215                  "cancelled/rejected/expired on the exchange — position is UNPROTECTED ...")
220      state._record_event(..., error_code="STOP_LOSS_CANCELLED", alert=True)
```

Reached from `handle_order_cancel` (`order_fill_coordinator.py:234-247`), registered as the
`OrderTracker` `on_cancel` callback (`src/engines/live/trading_engine.py:715`, wrapper at
`trading_engine.py:2089-2091`).

Its docstring at `order_fill_coordinator.py:202-204` asserts:

> "Deliberate cancels from the close path don't reach here — that path stops tracking the SL order
> before the callback can fire."

**That assertion is false.** See §3.3.

The terminal status itself is read correctly from the exchange —
`src/engines/live/order_tracker.py:620-673`, fed either by the 5 s REST poll
(`order_tracker.py:224-234` → `:253-255` `self._circuit_breaker.call(self.exchange.get_order, ...)`;
`poll_interval=5` set at `trading_engine.py:712`) or by the user-data WS `executionReport`
(`order_tracker.py:679-745`, Binance `X` mapped at `:757-763`,
`"CANCELED"/"REJECTED"/"EXPIRED"/"EXPIRED_IN_MATCH"`). Both are exchange truth. **There is no stale
read and no cache** — `get_order` is an uncached live REST round-trip, and the periodic reconciler's
verification is explicitly fail-closed via `lookup_order_fail_closed`
(`src/engines/live/reconciliation.py:114-142`, consumed at `:3838-3852`). This is what rules out
H1-as-stated.

### 3.2 Who was placing the 13 stop-losses

`process_execution_event` and the poll loop both only act on ids in `OrderTracker._pending_orders`
(`order_tracker.py:695-697`). Only four call sites ever add one:

- `src/engines/live/execution/stop_loss_manager.py:126` — `place_protection` (entry time)
- `src/engines/live/execution/stop_loss_manager.py:317` — `reprotect` (after a failed close)
- `src/engines/live/execution/entry_coordinator.py:908` — entry order
- `src/engines/live/recovery.py:428` — startup recovery

`src/engines/live/reconciliation.py` contains **zero** references to `order_tracker` — neither the
step-2 re-placement (`reconciliation.py:3949-3960`) nor `_place_missing_stop_loss`
(`reconciliation.py:4249-4330`) registers the stop it places.

**Therefore all 13 events came from stops placed by `stop_loss_manager.reprotect`** — the
failed-close re-protect path (`exit_coordinator.py:460-466` → `trading_engine.py:2188` →
`stop_loss_manager.py:249-335`). That independently corroborates the trading-loop cadence and
confirms the closes were failing.

### 3.3 The race — `LiveStopLossManager.cancel()`

`src/engines/live/execution/stop_loss_manager.py:129-165`:

```python
143      cancelled = False
144      try:
145          cancelled = bool(
146              state.exchange_interface.cancel_order(position.stop_loss_order_id, position.symbol)
147          )
...
161      # Only stop tracking when the cancel is confirmed; otherwise the order may
162      # still be live on the exchange and must remain watched.
163      if cancelled and state.order_tracker:
164          state.order_tracker.stop_tracking(position.stop_loss_order_id)
165      return cancelled
```

Two independent defects in eight lines:

**(a) The cancel is issued before `stop_tracking`.** Binance emits the `executionReport` with
`X=CANCELED` on the already-open user socket as soon as it processes the cancel — routinely *before*
the `DELETE /api/v3/order` HTTP response gets back to us. During that window the id is still in
`_pending_orders`, so `UserDataProcessor` → `order_tracker.process_execution_event` →
`_process_order_status` → `on_cancel` → `handle_stop_loss_cancelled` fires **for a cancel we issued
ourselves**. This is why 12 of 13 events landed while the WS was primary and healthy. The REST poll
has the same hole with a wider window: `_check_orders` re-checks `_pending_orders` membership
*before* `get_order` (`order_tracker.py:244-248`) but **not after**, so a ~100-400 ms
`get_order` round-trip that overlaps our `stop_tracking` still fires the callback.

**(b) `position.stop_loss_order_id` is read twice — line 146 and line 164 — with a network
round-trip between them, on an object shared across threads.** `LivePositionTracker.positions` is a
shallow `dict(self._positions)` copy (`src/engines/live/execution/position_tracker.py:125-131`), so
every consumer holds the *same* `LivePosition` object. When (a) fires, line 213 of
`order_fill_coordinator` sets that shared field to `None`, and line 164 then executes
`stop_tracking(None)` — a no-op that leaves the genuine id in `_pending_orders`.

### 3.4 How the orphaned, base-locking stop is created

Per iteration, `exit_coordinator.py:415-466`:

```python
416   if (state.enable_live_trading and state.exchange_interface
              and position.stop_loss_order_id):        # <-- gate on the field that just got nulled
421       pre_filled = state._stop_loss_filled_quantity(position)
422       if pre_filled is None or pre_filled > BORROW_DUST_EPSILON: ... return
429       if not state._cancel_stop_loss_order(position): ... return
436       protective_order_cancelled = True
450   exit_result = state.live_exit_handler.execute_exit(...)
460   if not exit_result.success and protective_order_cancelled:
466       state._reprotect_position(position)
```

Once `handle_stop_loss_cancelled` has nulled the shared field, three things follow:

1. **The next iteration's cancel is skipped entirely** — line 416's gate is `False`, so the engine
   proceeds straight to `execute_exit` believing nothing is resting. Any stop actually resting on
   the exchange keeps its lock on the base asset.
2. **The periodic reconciler places a duplicate.** `reconciliation.py:3824-3828`:
   `if not sl_order_id: self._place_missing_stop_loss(position, order_key)`. It sees `None` and
   places a *second* stop while the first still rests. `reprotect` (`stop_loss_manager.py:293-317`)
   does the same from the trading loop. Neither checks the exchange for an existing resting
   protective order first. Whichever writes `position.stop_loss_order_id` last wins; the other
   becomes an untracked orphan that nothing will ever cancel — and it holds 0.0086 ETH locked.
   That is precisely the 0.0086 in §2.4.
3. **The WS degradation adds extra chances to hit (2).** `ws_health.py:845` and `:891` call
   `_periodic_reconciler.reconcile_once()` on user-stream disconnect and after each reconnect, and
   `ws_health.py:552` / `:836-837` call `order_tracker.enable_polling()`. The 15:41:54 degrade
   therefore fired off-cadence reconciler cycles and switched on the 5 s REST poll right as the
   exit/re-protect churn began at 15:44. **This is the real, and only, causal role of
   `USER_WS_DEGRADED` — it widened the race window and multiplied the duplicate-placement
   opportunities. It is not a stale or misinterpreted read.**

### 3.5 The close is then silently shrunk to dust

`src/engines/live/execution/execution_engine.py:1178-1200`:

```python
1180  if order_side == OrderSide.SELL:
1181      free_base = self._free_base_for_close(symbol)
1182      if free_base is not None and free_base < quantity:
1183          logger.warning("Close sell qty %.8f for %s exceeds free base balance %.8f "
1185                         "— capping to holdings to avoid -2010.", ...)
1190          quantity = free_base
1191      quantity = self._normalize_quantity(symbol, quantity, position_notional, floor=True)
1194  if quantity <= 0:
1199      return None
```

The cap's own docstring (`execution_engine.py:1294-1302`) states the assumption that makes it safe:

> "the resting stop-loss is cancelled before the close (#710), so its previously locked inventory is
> free again by the time this reads it."

**This incident falsifies that assumption.** The cap was designed to shave a fee-rounding sliver
(fractions of a percent). Here it shaved **98.9%** of the order — from 0.0087 to 0.00009419 — and
the only guard is `quantity <= 0`. There is no floor on how much of the requested close may be
silently discarded.

---

## 4. The latent P1 hidden underneath: dust close booked as a full close

`exit_handler.execute_exit` (`src/engines/live/execution/exit_handler.py:493-520`) calls
`execution_engine.execute_exit(...)` and, **on success**, unconditionally calls
`self.position_tracker.close_position(order_id=..., exit_price=execution_result.executed_price, ...)`
— closing the position at **full tracked size**, booking full realized P&L, and closing the DB row.

There is no check that the quantity actually executed matches the quantity intended.

So if the free-base cap shrinks a close to, say, 30% of the position — anything at or above
`MIN_NOTIONAL` — the engine sells 30%, books a 100% close, marks the DB position CLOSED, and
abandons ~70% of the inventory on the exchange with no tracked position and no stop-loss. That is
the same capital-integrity class as #648/#653 (phantom balance) and it is **currently latent in
production**. In this incident it was masked purely by luck: $0.196 < $5 `MIN_NOTIONAL`, so the
order was rejected instead of filled. A larger position with the same fault would have filled.

I consider this a more serious finding than the alert loop itself, and it is the one I would fix
first.

---

## 5. Reconstructed timeline

```
2026-07-21 17:55  ETHUSDT LONG opened, 0.0087 ETH @ 1918.51, SL placed & tracked
...
2026-08-19 15:41:54  user data stream circuit-opens -> REST-degraded
                     -> order_tracker.enable_polling()          (ws_health.py:552)
                     -> reconcile_once() off-cadence            (ws_health.py:845)
2026-08-19 15:44:12  15:00 hourly candle's low dips through the trailing stop.
                     Strategy emits exit reason "Stop loss" and will keep doing so
                     every ~66s until the candle closes at 16:00.
      each iteration: exit_coordinator cancels the tracked SL (#710 path)
                      -> Binance pushes executionReport X=CANCELED on the open socket
                      -> OrderTracker fires on_cancel BEFORE stop_tracking runs
                      -> handle_stop_loss_cancelled: STOP_LOSS_CANCELLED alert
                         + position.stop_loss_order_id = None  (shared object)
                      -> cancel() line 164 executes stop_tracking(None): no-op
                      -> reconciler and/or reprotect each see "no SL" and place one;
                         one of them becomes an untracked orphan resting on the exchange
                      -> orphan locks 0.0086 ETH
                      -> next close is capped to free base = 0.00009419 ETH ($0.20)
                      -> rejected under MIN_NOTIONAL -> exit fails -> reprotect -> repeat
2026-08-19 15:46:32  user stream RECOVERS. Loop continues unaffected for 12 more events.
2026-08-19 15:59:46  last event of the run
2026-08-19 16:00:49  new hourly candle -> signal flips to hold_position -> loop stops
2026-08-19 17:19:38  same mechanism recurs once inside the 17:00 candle
2026-08-19 17:20:52  close finally succeeds: 0.0086 ETH @ 2071.98 (order 7434)
                     -> the locked inventory was free by then; the orphan had been
                        consumed or cancelled
2026-08-20 16:48:51  CLOSE_ONLY latched; RECONCILE_CRITICAL "ETHUSDT unprotected"
```

The safety guards behaved correctly throughout. Capital was preserved. But the position was
genuinely unprotected for stretches of ~66 s at a time, and the close path was silently disarmed.

---

## 6. Hypotheses — disposition with evidence

### H1 — degraded user stream causes false/stale order-state reads: **PARTIALLY CONFIRMED (mechanism differs)**

Ruled out as stated:
- `USER_WS_RECOVERED` at **15:46:32**, four seconds after the first event; 12 of 13 events occurred
  with the stream healthy and WS-primary (`system_events`, §2.1).
- The degraded read path is `OrderTracker._check_orders` → `exchange.get_order(order_id, symbol)`
  (`order_tracker.py:253-255`) — a per-order live REST lookup. It is **not** `get_open_orders`, so
  LESSONS §1.8's fail-open class does not apply here.
- No caching layer exists on order reads. `CachedDataProvider` caches OHLCV only; `binance_provider`'s
  only cache is `_margin_symbol_verified` (`binance_provider.py:956-960`). Every order read is live.
- The reconciler's runtime SL verification is explicitly fail-closed and *refuses* to re-place on an
  unverifiable lookup (`reconciliation.py:3838-3852`).
- `USER_WS_DEGRADED` has fired 29 times since 2026-07-05 with zero `STOP_LOSS_CANCELLED` events.

Confirmed, in the narrower sense: the degradation **is** causally involved, via
`order_tracker.enable_polling()` (`ws_health.py:552`) and the off-cadence
`reconcile_once()` calls (`ws_health.py:845`, `:891`), which widened the race window of §3.3 and
multiplied duplicate-SL opportunities. Self-inflicted: **yes** — but by a read-modify-write race,
not by a misread.

### H2 — genuine exchange-side rejection: **RULED OUT for the stop; CONFIRMED for the close**

- The stops were cancelled by us: `exit_coordinator.py:429` → `stop_loss_manager.py:146`
  `cancel_order(...)`, once per trading-loop iteration, in lockstep with the 15 "Stop loss" exit
  decisions in `strategy_executions` (§2.2). The cadence, count and timing all match the trading
  loop; nothing on the exchange side produces a clean 66 s period.
- No `reduceOnly` is ever sent: `place_stop_loss_order` builds `sl_params` at
  `binance_provider.py:2036-2050` with `symbol/side/type/quantity/stopPrice/price/timeInForce`
  (+ optional `sideEffectType`) only.
- Margin/borrow state is not implicated: the position is a **LONG**, no borrow involved; the
  reconciler's margin-close branch (`reconciliation.py:3610`) never ran (the position stayed
  tracked and OPEN throughout, and would have been popped had it run).
- The genuine rejection is on the **close**: order 7433, notional $0.196, under `MIN_NOTIONAL`. That
  is a consequence of §3.5, not a cause.
- "Insufficient free balance" is real but is *our own resting stop's lock*, not an account shortfall.

### H3 — quantity/precision drift on re-placement (LESSONS §1.1): **RULED OUT**

- Both snap sites in `place_stop_loss_order` are correctly quantized:
  `binance_provider.py:1969-1974` (`stop_price`/`limit_price` via `quantize_to_step(round(x/tick)*tick, tick)`)
  and `:2011-2015` (`quantity` via `quantize_to_step(quantity, step_size)`). #695/#699 hardening intact.
- A precision rejection (`51077` / `-1111`) would mean **placement failed** and returned `None`. The
  observed behaviour is the opposite: 13 successful placements with 13 distinct exchange order ids.
- `place_stop_loss_order` returns `None` on every rejection path
  (`binance_provider.py:2081-2093`), which would have produced `RECONCILE_CRITICAL`
  "exchange returned no order id" audit rows. `reconciliation_audit_events` is empty for the window.
- Grep of `round(... / ...) * ...` across `src/` shows no unquantized survivor on any exchange-bound
  numeric.

---

## 7. Recommended fixes

Ordered by capital risk, not by proximity to the reported symptom.

### R1 (P0-adjacent, latent in prod now) — never silently shrink a close

`src/engines/live/execution/execution_engine.py:1178-1200`. The free-base cap must not discard an
arbitrary fraction of the intended close. Require the capped quantity to be within a tight tolerance
of the requested quantity (the cap exists only to shave a fee-rounding sliver); otherwise **abort the
close, escalate, and let the reconciler resolve exchange truth** rather than submitting a partial the
caller will book as a full close.

```suggestion
            if order_side == OrderSide.SELL:
                free_base = self._free_base_for_close(symbol)
                if free_base is not None and free_base < quantity * CLOSE_HOLDINGS_CAP_TOLERANCE:
                    logger.critical(
                        "Close sell qty %.8f for %s exceeds free base %.8f by more than the "
                        "fee-rounding tolerance — inventory is locked (orphaned stop?); "
                        "ABORTING close rather than selling a fraction.",
                        quantity, symbol, free_base,
                    )
                    return None
```

Independently, `exit_handler.execute_exit` must not call `position_tracker.close_position(...)` at
full size when the executed quantity is materially below the intended quantity. `ExecutionResult`
already carries both `quantity` and `requested_quantity` (`execution_engine.py:88-89`) — compare
them and convert a short fill into a partial-exit adjustment or a hard escalation, never a full
close.

### R2 (P1) — make the deliberate SL cancel race-free

`src/engines/live/execution/stop_loss_manager.py:129-165`. Capture the id **once**, and untrack
**before** issuing the cancel; re-track on a failed cancel so the "order may still rest → keep
watching" invariant is preserved.

```suggestion
        sl_id = position.stop_loss_order_id
        if state.order_tracker:
            state.order_tracker.stop_tracking(sl_id)
        cancelled = False
        try:
            cancelled = bool(state.exchange_interface.cancel_order(sl_id, position.symbol))
```

with a `if not cancelled and state.order_tracker: state.order_tracker.track_order(sl_id, position.symbol)`
on the failure path. This removes the false `STOP_LOSS_CANCELLED` alert, removes the
`stop_tracking(None)` no-op, and — most importantly — stops the alert handler from nulling
`stop_loss_order_id` out from under a live resting order.

Belt-and-braces: `order_tracker._check_orders` should re-check `_pending_orders` membership *after*
`get_order` returns and before firing `on_cancel` (it already re-checks before; see
`order_tracker.py:244-248`).

### R3 (P1) — never place a second stop while one may still rest

`reconciliation._place_missing_stop_loss` (`reconciliation.py:4249`), the step-2 re-placement
(`reconciliation.py:3949`) and `stop_loss_manager.reprotect` (`stop_loss_manager.py:293`) all place
unconditionally. Each should first consult the fail-closed
`has_open_orders` accessor (`binance_provider.py:1161-1176`, the LESSONS §1.8-compliant one) for the
symbol and, on `None` (unknown) or `True`, **adopt or cancel the existing protective order rather
than stacking a second**. Stacking is what locked the base and starved every close.

### R4 (P2) — close the observability gap #1097 left

`_record_order_error` is wired into `place_stop_loss_order` only. The rejection that actually
mattered here was on the **close** path. Wire the same durable capture into
`execution_engine._close_live_position` so the next occurrence records the Binance code for the
close, not just for the stop.

### R5 (P2) — re-register reconciler-placed stops with the OrderTracker

`src/engines/live/reconciliation.py` has zero `order_tracker` references, so a reconciler-replaced
stop is invisible to fill detection until the next 120 s cycle. Additionally
`ws_health.check_user_stream_health` returns early when `get_tracked_count() == 0`
(`ws_health.py:416-417`), so if the only live orders are reconciler-placed the user-stream watchdog
goes permanently idle. Not causal here, but it is a real hole on the same code path.

### R6 (P3) — fix the false docstrings that encode the broken assumptions

- `order_fill_coordinator.py:202-204` — claims deliberate cancels cannot reach the handler.
- `execution_engine.py:1294-1302` — claims the resting stop is always cancelled before the close.
- `reconciliation.py:3130` — says "default 60s"; the constant is 120 s.

---

## 8. Relationship to #723/#724

**Different root cause; do not merge them.** #723/#724 concern the user stream degrading to REST and
not returning to WS-primary until restart. Here the stream recovered on its own in 4.6 minutes
(`USER_WS_RECOVERED` 15:46:32). The degradation's only role in #1104 was to enable REST polling and
fire two off-cadence reconciler cycles. Fixing #723/#724 would not have prevented this incident;
fixing R1–R3 would have.

There is, however, a shared aggravator worth noting on both issues: `ws_health` reacting to stream
transitions by driving `reconcile_once()` means connectivity flapping injects extra reconciler cycles
into whatever the trading loop is doing at that moment. R3 makes that safe.

---

## 9. What was verified correct (no action needed)

- Order-status reads are exchange truth, uncached, per-order (`order_tracker.py:253-255`).
- The reconciler's SL verification is fail-closed and refuses to re-place on an unverified lookup
  (`reconciliation.py:3838-3852`) — #713 hardening intact.
- `lookup_order_fail_closed` / `has_open_orders` correctly implement LESSONS §1.8.
- Precision quantization on both quantity and price is intact (#695/#699).
- `_notify_tracking_lost` is correctly kept distinct from `on_cancel` (`order_tracker.py:330-345`).
- The orphaned-order sweep (`reconciliation.py:4044-4060`) cannot cancel a stop-loss: SLs carry an
  exchange-generated client id (no `newClientOrderId` is ever passed), so the `atb`-prefix filter
  never matches them. I checked this specifically as a candidate self-cancel mechanism and ruled it out.
- `account_sync` issues no exchange cancels (`account_sync.py:581-694` only marks DB rows).
- Trailing-stop updates never touch exchange orders (`exit_handler.py:769-820`).
- The close-only halt, the drawdown guard and the `RECONCILE_CRITICAL` escalation all fired as
  designed. The last line of defence worked.
