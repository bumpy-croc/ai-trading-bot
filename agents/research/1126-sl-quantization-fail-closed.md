# Stop-loss quantization fails open when get_symbol_info() fails (GH #1126)

Date: 2026-09-14 · Branch: `fix/1126-sl-quantization` · Status: implemented

## 1. The defect

`src/data_providers/binance_provider.py::place_stop_loss_order` gated all tick/lot
quantization behind `if symbol_info:`:

```python
symbol_info = self.get_symbol_info(symbol)
step_size = 0.0
if symbol_info:
    tick_size = ...
    if tick_size > 0:
        stop_price = quantize_to_step(round(stop_price / tick_size) * tick_size, tick_size)
        limit_price = quantize_to_step(round(limit_price / tick_size) * tick_size, tick_size)
    step_size = ...
```

When `get_symbol_info(symbol)` returned `None` (transient API failure or a cache miss),
`stop_price`/`limit_price` kept their raw float value — `stop_price * (1 ± SLIPPAGE_FACTOR)` —
with no tick snap, and were sent to Binance verbatim. That is exactly the shape Binance
rejects with **-1111** ("Parameter 'price' has too much precision") per LESSONS.md §1.1.
`quantize_to_step(value, step_size)` is a documented no-op when `step_size <= 0`
(`src/trading/precision.py:16-17`), so calling it unconditionally would not have helped —
there is no way to quantize a value to an unknown tick.

Twelve lines below, the SELL free-balance cap runs unconditionally with a comment claiming
it must (correctly) — but the *quantity* lot-rounding (`round(qty/step)*step` /
`floor(qty/step)*step`) was itself still gated on `step_size > 0`, which is `0.0` exactly
when `symbol_info` is missing. So the same asymmetry existed on the quantity side too: a
missing symbol lookup silently skipped LOT_SIZE rounding as well as PRICE_FILTER rounding.
Point 3 of the issue ("grep every other consumer of `symbol_info` … for the same
conditional-hardening asymmetry") is confirmed — both prices and quantity were unprotected,
not just price.

Net effect: a transient `get_symbol_info` hiccup produced a stop-loss placement that either
got rejected by Binance for precision (-1111/51077) or, worse, was accepted with a
tick/lot-violating value and rejected later — in both cases the position was left
unprotected with **no loud, distinguishing error**. The existing `error_params["symbol_info_available"]`
flag captured this after the fact for forensics, but nothing in the control flow ever
refused to send.

## 2. Why "fail loud" over "conservative fallback"

The issue proposed two options: (1) quantize to a conservative fixed decimal count when
`tick_size` is unknown, or (2) refuse to submit and fail loud. Went with (2), matching the
issue's own stated preference and the shape used elsewhere in this function
(`ClientUnavailable`, `InvalidStopPrice`, `ZeroQuantityAfterSizing` all return `None` +
`_record_order_error`):

- A "conservative fixed decimal count" is still a guess about a specific exchange's
  PRICE_FILTER/LOT_SIZE for a specific symbol. Get it wrong (e.g. a low-tick-size asset the
  bot has never fetched) and you still ship an out-of-tolerance value — silently, with no
  signal that the fallback ever fired.
- Every other precondition failure in this function already fails closed and returns `None`.
  A conservative-quantize path would have been the *one* branch in the function that tries
  to proceed anyway, inconsistent with the function's own established contract.
- `place_stop_loss_order` returning `None` is not a new failure mode for callers — the
  reconciler's unprotected-position escalation (`_audit_unprotected` → `on_critical` →
  close-only) already exists and already fires on every other `None` return from this
  function. Fail-closed here needs zero new escalation plumbing.

## 3. The fix

`src/data_providers/binance_provider.py`, `place_stop_loss_order`: when `get_symbol_info`
returns falsy, log an error, call `self._record_order_error(..., error_type="SymbolInfoUnavailable", params=error_params)`
(same durable sink used by every other failure branch — routes through
`order_error_sink` → `LiveExecutionEngine._record_exchange_order_error` →
`system_events` at `severity="critical"` for stop-loss operations, confirmed lowercase per
LESSONS.md §1.1/§weekly-retro precedent), and return `None` — **before** doing any
price/quantity arithmetic. The subsequent `if symbol_info:` block was flattened to
unconditional code (symbol_info is guaranteed present past the guard), which also fixes the
quantity-side asymmetry described above as a side effect of the same change, with no
additional branch.

This follows the same shape as #1097's `ExchangeOrderError` capture (durable, non-blocking to
the sink, `last_order_error` still set) and the fail-closed pattern used across this function.

## 4. Verification

- `atb test unit` (full suite) and `atb test integration` — see PR test-plan section for
  results.
- `atb dev quality --changed` — black/ruff/mypy/bandit on changed files only.
- Updated `tests/unit/data_providers/test_binance_provider.py`:
  - Renamed/rewrote `test_sell_stop_loss_caps_without_symbol_info` →
    `test_sell_stop_loss_fails_closed_without_symbol_info`: asserts `result is None`,
    `create_order` is never called, and the durable `order_error_sink` receives exactly one
    `ExchangeOrderError` with `error_type == "SymbolInfoUnavailable"` and
    `params["symbol_info_available"] is False`.
  - `test_place_stop_loss_sell_order_success` / `test_place_stop_loss_buy_order_success` /
    `test_auto_limit_price_calculation_sell` / `test_auto_limit_price_calculation_buy` /
    `test_custom_limit_price_used` / `test_order_exception_returns_none` /
    `test_generic_exception_returns_none` / `test_missing_order_id_returns_none` previously
    relied on the *real* `get_symbol_info` against an empty mocked `exchange_info` (which
    silently returns `None` for any symbol) — they were incidentally exercising the buggy
    "skip quantization" path rather than the behavior their names describe. Each now mocks
    `get_symbol_info` with real tick/step data so they exercise the code path they're
    actually named for.

## 5. What did NOT change

- The reconciler's existing unprotected-position escalation path
  (`_audit_unprotected` → `on_critical` → close-only) — unchanged, and is what actually pages
  for this failure once #1126's `SymbolInfoUnavailable` row lands in `system_events`.
- `quantize_to_step` itself — already correct (LESSONS.md §1.1); this fix only ensures it is
  never bypassed by an absent tick/step.
- Coinbase provider — checked; `CoinbaseProvider.place_stop_loss_order` is an unimplemented
  stub that always returns `None` (no quantization logic exists there to have the same gap).
