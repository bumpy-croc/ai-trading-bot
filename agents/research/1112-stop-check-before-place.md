# #1112: stop-loss placement never checks for an existing resting stop

## Why this matters

Root-caused the incident that latched prod close-only for 17 days
(2026-08-27 → 2026-09-14, #1121, closed): a stop got orphaned — a
duplicate/untracked resting stop locked 97%+ of the base inventory. #1108
made the *consequences* loud and safe (abort-and-latch + page). This issue
fixes the *cause*: nothing checked whether a protective stop was already
resting before placing another, so duplicates could accumulate and lock
inventory in the first place. Same root enabler as the original #1104
incident (2026-08-19).

## Where the gap actually was

Grepping `self.exchange.place_stop_loss_order(` in
`src/engines/live/reconciliation.py` turned up **seven** call sites, not the
three named in the issue body (which cited pre-#1108 line numbers):

1. `_reconcile_filled_entry_order` recovery path (~line 995) — placing a stop
   for a position recovered at startup.
2. `_resize_stop_loss_after_partial_exit` (~1393) — cancel-then-replace after
   a partial exit.
3. Startup `_verify_entry_and_stop` "missing SL" branch (~1675).
4. `_verify_stop_loss` "SL order not found" branch (~1988).
5. `_verify_stop_loss` "SL cancelled/expired/rejected" branch (~2157).
6. Periodic step-2 re-placement inside `_reconcile_cycle` (~4093).
7. `_place_missing_stop_loss` (~4453) — the exact #1104 shape (no
   `stop_loss_order_id` at all → place one).

Plus two sites in `src/engines/live/execution/stop_loss_manager.py`:
`place_protection` (fresh entry) and `reprotect` (after a failed close).

Every one of these follows the identical shape: `if sl_order_id: <success> else:
<critical log, unprotected escalation>` — none of them first asked the
exchange "is something already resting here?"

## Design

### Root-cause choke point, not nine bespoke checks

Rather than duplicating a resting-stop check at nine call sites (or, worse,
baking it into `BinanceProvider.place_stop_loss_order` itself — rejected,
see below), a single pair of functions was added to `reconciliation.py`
(co-located with the existing `lookup_order_fail_closed` guard, which solves
the analogous problem for order *lookups*):

- `guard_stop_placement(exchange, symbol, side) -> StopPlacementDecision`
  — the primitive. Classifies into `PROCEED` / `ADOPT` / `REFUSE`.
- `place_or_adopt_stop_loss(exchange, *, symbol, side, quantity, stop_price,
  side_effect_type=None) -> str | None` — the call-site replacement for a
  bare `exchange.place_stop_loss_order(...)` call. `None` on `REFUSE` is
  deliberately indistinguishable from `place_stop_loss_order` itself
  returning `None`: every call site already treats a falsy result as
  "placement failed, escalate" — so swapping in this wrapper required zero
  other changes to any of the seven `reconciliation.py` call sites.

`stop_loss_manager.py`'s two call sites (`place_protection`, `reprotect`)
have bespoke 3-attempt retry loops around the raw exchange call, so they
consult `guard_stop_placement` directly, once, before entering the loop
(ADOPT/REFUSE short-circuit the loop entirely; PROCEED runs the existing
retry loop unchanged).

### Why not push the check into `BinanceProvider.place_stop_loss_order`?

That was the first design considered — it would protect every current *and
future* call site with zero risk of a tenth site being added without the
check. Rejected because:

- It would make `place_stop_loss_order` fail-closed by default for **every**
  test double that doesn't explicitly mock `get_open_orders_checked` /
  `_call_get_open_orders` — including the ~30 existing tests in
  `test_binance_provider.py` that construct `mock_client = Mock()` and
  never touch open-orders at all. Verified: `mock_client.get_open_orders(...)`
  on an unconfigured `Mock()` returns another `Mock()`, and iterating that
  is `TypeError: 'Mock' object is not iterable` — every one of those ~30
  tests would need updating to keep asserting a successful placement.
- It conflates two different responsibilities: the provider's job is "talk
  to Binance"; "should we place a NEW one, or is one already resting" is a
  call-site/orchestration decision (the caller knows whether it's placing
  fresh protection vs. handling a confirmed-missing/expired one).

The chosen design instead required updating far fewer fixtures — only the
callers that go through the *new* guard (`stop_loss_manager.py`'s two sites)
use plain `Mock()` without magic-method defaults; the `reconciliation.py`
call sites' tests use `MagicMock()`, whose default `__iter__` returns `iter([])`,
so unconfigured `get_open_orders_checked` calls transparently resolve to
`PROCEED` and every pre-existing reconciliation test kept passing unmodified.

### Classification logic (`guard_stop_placement`)

Uses the new fail-closed accessor `get_open_orders_checked(symbol)` (added
to `ExchangeInterface`/`BinanceProvider`, mirroring the existing
`has_open_orders`/`get_order_checked` fail-closed pattern — LESSONS.md
§1.8: `get_open_orders` fails OPEN and must never be used for a safety
decision).

```
orders = exchange.get_open_orders_checked(symbol)
if orders is None:                      -> REFUSE (lookup unconfirmed)
resting = [o for o in orders if o.stop_price is not None]
if not resting:                         -> PROCEED
if len(resting) > 1:                    -> REFUSE (already ambiguous/duplicated)
if resting[0].side == side:             -> ADOPT(resting[0].order_id)
else:                                    -> REFUSE (wrong side, don't touch)
```

Detection deliberately keys on **`stop_price is not None`**, not
`order_type == OrderType.STOP_LOSS`. While auditing this, found (and filed
as **#1152**, non-blocking) that `_convert_order_type` never mapped
Binance's actual `"STOP_LOSS_LIMIT"` type string (the one `place_stop_loss_order`
actually sends) to `OrderType.STOP_LOSS` — it silently fell through to the
`OrderType.MARKET` default. Had the guard filtered by `order_type`, it would
have found **zero** resting stops, always — defeating the entire fix.
`stop_price` is unaffected by that bug and is a strictly stronger signal of
"this is a stop order" regardless of type-string quirks.

### The three cases from the task brief

- **The resting order IS the tracked one (must not false-positive):** by
  construction, every guarded call site only fires when the position's own
  `stop_loss_order_id` is already `None`/confirmed-gone (that's *why* it's
  trying to place a new one). So "the resting order actually is still our
  order, we just lost track of it" and "ADOPT" are the same case — this is
  exactly the #1104 race (tracked id nulled while the order still rests).
  ADOPT recovers it instead of duplicating.
- **A genuine orphan** (unrelated leftover order, e.g. from a previous
  position on the same symbol): guarded against by the **side match**.
  Adopting *any* resting stop blindly is dangerous — it might be a stale
  order from a closed position with the wrong price/side entirely, which
  would be worse than a duplicate (silently mis-protecting the position
  while reporting itself as protected). Requiring `resting[0].side == side`
  is a cheap, meaningful filter: a resting stop for the *opposite* direction
  cannot be the order we're trying to place, so it's REFUSE, not ADOPT.
  (Quantity match was considered too, but partial-exit sizing makes exact
  quantity comparison brittle; side match already catches the worst failure
  mode — protecting the wrong direction.)
- **The exchange query itself fails (must fail closed):** `orders is None`
  → REFUSE. Also REFUSE when the exchange has no `get_open_orders_checked`
  accessor at all (`getattr(..., None)` not callable) — never silently
  degrades to "assume none and place anyway". `place_or_adopt_stop_loss`
  and the two `stop_loss_manager.py` sites all funnel REFUSE into the
  pre-existing "placement failed" / "UNPROTECTED, MANUAL REVIEW REQUIRED"
  escalation paths — no new failure mode, just routed earlier.

## Files changed

- `src/engines/live/reconciliation.py` — `StopPlacementCheck`,
  `StopPlacementDecision`, `guard_stop_placement`, `place_or_adopt_stop_loss`;
  all 7 `place_stop_loss_order` call sites route through the wrapper.
- `src/engines/live/execution/stop_loss_manager.py` — `place_protection` and
  `reprotect` consult `guard_stop_placement` before their retry loops.
- `src/data_providers/binance_provider.py` — new `get_open_orders_checked`
  (fail-closed variant of `get_open_orders`, same shape as `has_open_orders`).
- `src/data_providers/exchange_interface.py` — `get_open_orders_checked`
  added to the interface with a fail-closed default (`None`); documented the
  new invariant on `place_stop_loss_order`'s docstring (route new placements
  through the guard, don't call the provider directly).
- Tests: `tests/unit/live/test_reconciliation.py` (guard unit tests +
  reconciler-level ADOPT/REFUSE/multiple-resting/unconfirmed-lookup
  scenarios — the issue's definition-of-done), `tests/unit/engines/live/
  test_stop_loss_manager.py` (place_protection ADOPT/REFUSE), `tests/unit/
  engines/live/test_close_cancel_stop_710.py` (reprotect ADOPT/REFUSE; also
  updated its exchange fixture to default `get_open_orders_checked` to `[]`
  so existing reprotect tests keep exercising the real placement path).

## Related, not done here

- **#1152** (filed): `_convert_order_type` doesn't map `STOP_LOSS_LIMIT` /
  `TAKE_PROFIT_LIMIT` → silently reports every stop-loss order as
  `OrderType.MARKET`. Found while designing this fix's detection logic;
  doesn't block #1112 (which uses `stop_price`, not `order_type`), dispatched
  as a separate small PR.
- **#740** (orphan-order sweep can't match stop-losses — `atb`-prefix client-id
  filter never fires for exchange-generated stop ids): still open, still a
  real gap for a stop that becomes orphaned by a path *other* than "we were
  about to place a new one" (e.g. one that was already resting, untracked,
  when the reconciler was flat for that symbol). This fix prevents new
  duplicates at placement time; it does not add a periodic sweep to find and
  clean up an already-existing orphan through this path (#740's own fix).
- **#739** (periodic reconciler steps 1/1b vs step 2 stale-snapshot re-arm):
  same "place a stop we should not have placed" class, different mechanism
  (stale snapshot vs. missing tracked id). Not addressed here; the new guard
  incidentally also protects step 2's re-placement call site (it's one of
  the seven), but the stale-snapshot iteration bug itself (iterating removed
  positions) is untouched.
