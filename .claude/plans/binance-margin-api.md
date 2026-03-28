# Plan: Migrate Binance Spot API to Cross Margin API

## Context

The bot is live on production trading ETHUSDT with $100, but the funds are in the Cross Margin wallet and the bot only knows the Spot API. It can't see the balance or place margin orders (needed for shorts). The HyperGrowth strategy is predominantly bearish — without margin support, the bot sits idle.

**Goal:** Default to Margin API for all account/order operations. Add env var to switch back to Spot if needed.

## Files to Modify

| File | Changes |
|------|---------|
| `src/data_providers/binance_provider.py` | Primary: dispatch methods, balance normalization, startup checks, fail-fast |
| `src/data_providers/exchange_interface.py` | Add `side_effect_type` to `place_order()` and `place_stop_loss_order()` |
| `src/data_providers/coinbase_provider.py` | Add `side_effect_type` optional param to match interface (ignored) |
| `src/engines/live/execution/execution_engine.py` | Pass explicit `side_effect_type` based on position side + entry/exit |
| `src/engines/live/trading_engine.py` | Pass `side_effect_type` on SL placement calls (~line 2677) |
| `src/engines/live/reconciliation.py` | Pass `side_effect_type` on SL placement calls (~lines 523, 825, 2330); skip spot-only checks in margin mode |
| `src/engines/live/account_sync.py` | Skip position sync from exchange in margin mode |
| `tests/unit/data_providers/test_binance_provider.py` | New margin tests + fixture updates |

## Implementation

### 1. Add `_use_margin` flag and fail-fast to BinanceProvider.__init__

Read `BINANCE_ACCOUNT_TYPE` from config (default `"margin"`). Read `TRADING_MODE` to determine live vs paper.

```python
self._use_margin = config.get("BINANCE_ACCOUNT_TYPE", "margin").lower() == "margin"
self._is_live = config.get("TRADING_MODE", "paper").lower() == "live"
```

**Fail-fast rule:** If `_use_margin=True` AND `_is_live=True` AND client init fails → raise immediately. Do NOT fall back to offline stub. A fake client placing dummy margin orders is a fund-loss path.

In paper mode, offline stub fallback is still acceptable.

### 2. Startup capability checks

When `_use_margin=True` and client init succeeds:

**Account-level:** Call `get_margin_account()`, verify `tradeEnabled=True` and `borrowEnabled=True`. Log `marginLevel`.

**Symbol-level (lazy, on first order):** BinanceProvider is instantiated without a symbol in many places (dashboards, data commands, fallback provider). So the per-symbol check can't happen at init. Instead, on the first `_call_create_order()` call in margin mode, call `get_margin_symbol(symbol=...)` to verify `isMarginTrade=True`, `isBuyAllowed=True`, `isSellAllowed=True`. Cache the result so subsequent orders skip the check. This catches "account has margin but this pair doesn't support it."

If any check fails, raise with clear error before placing the order.

### 3. Add 7 private dispatch methods in binance_provider.py

Thin wrappers calling margin or spot client methods based on `_use_margin`. See table in prior section. All margin calls include `isIsolated='FALSE'` for cross margin.

**Balance normalization** in `_call_get_account()` for margin mode — preserve debt fields:
```python
raw["balances"] = [
    {
        "asset": a["asset"],
        "free": a["free"],
        "locked": a["locked"],
        "borrowed": a.get("borrowed", "0"),
        "interest": a.get("interest", "0"),
        "netAsset": a.get("netAsset", a["free"]),
    }
    for a in raw.get("userAssets", [])
]
# Normalize account info fields
raw["canTrade"] = raw.get("tradeEnabled", False)
```

### 4. Explicit margin intent on ALL order paths

Add `side_effect_type: str | None = None` to both `place_order()` AND `place_stop_loss_order()` in:
- `exchange_interface.py` (abstract interface)
- `binance_provider.py` (passes to `_call_create_order`)
- `coinbase_provider.py` (accepts param, ignores it)

The `_call_create_order` wrapper injects `sideEffectType` into margin order params only when provided and `_use_margin=True`.

**Callers that must pass `side_effect_type`:**

| Caller | File | Lines | Intent |
|--------|------|-------|--------|
| Entry execution (long) | execution_engine.py | ~669 | `None` (buy with existing USDT) |
| Entry execution (short) | execution_engine.py | ~669 | `'MARGIN_BUY'` (borrow asset to sell) |
| Exit execution (close long) | execution_engine.py | ~800 | `'AUTO_REPAY'` |
| Exit execution (close short) | execution_engine.py | ~800 | `'AUTO_REPAY'` (buy back, repay) |
| SL placement after entry | trading_engine.py | ~2677 | `'AUTO_REPAY'` (SL closes position) |
| SL re-placement (reconciliation) | reconciliation.py | ~523, ~825, ~1094, ~1351, ~1501, ~2330, ~2545 | `'AUTO_REPAY'` |
| Emergency market close (reconciliation) | reconciliation.py | ~580 | `'AUTO_REPAY'` (closes recovered short) |

When `_use_margin=False`, `side_effect_type` is ignored (not passed to Binance).

### 5. Disable exchange-position sync in margin mode

In `account_sync.py`, when margin mode is active, skip both `_sync_positions()` AND `_sync_balances()`:

- **`_sync_positions()`**: The spot position model (positive balance = long) doesn't map to margin. Building a proper margin position model is a separate concern.
- **`_sync_balances()`**: Treats USDT exchange balance as authoritative account balance. In cross margin after a short, USDT rises from sale proceeds while liability sits in the borrowed asset — syncing this would corrupt DB balance and break sizing/risk logic.

The bot's internal position tracker (which tracks orders it placed) still works correctly — this only disables exchange-state-driven overrides. Order sync (`_sync_orders()`) still runs.

### 6. Skip spot-only reconciliation checks in margin mode

In `PeriodicReconciler`, accept a `use_margin` flag (passed from trading engine config):
- **Skip `_verify_asset_holdings()`** — spot-specific (assumes holdings = positions)
- **Skip `_verify_balance()`** — spot-specific (assumes USDT = DB balance - notional). Cross-margin equity is account-wide, not USDT-only.
- **Keep all order-based checks** — `_verify_entry_order()`, SL status checks, orphan detection all work the same regardless of spot/margin

### 7. Fix cancel_all_orders

Replace `self._client.cancel_all_orders(symbol=symbol)` with:
```python
orders = self._call_get_open_orders(symbol=symbol)
for order in orders:
    self._call_cancel_order(symbol=symbol, orderId=order.order_id)
```
This works for both spot and margin.

### 8. Update offline client stub

Add margin method stubs to `_OfflineClient` for paper mode. But remember: live + margin mode never uses the stub (Step 1).

### 9. Tests

Existing tests: set `provider._use_margin = False` in fixtures so they pass unchanged.

New tests:
- `test_margin_flag_from_env` — flag reads correctly
- `test_margin_live_mode_no_offline_fallback` — raises on init failure
- `test_margin_startup_checks` — verifies tradeEnabled/borrowEnabled/symbol checks
- `test_margin_create_order_injects_params` — isIsolated and sideEffectType
- `test_margin_get_balances_normalizes_with_debt` — userAssets → balances with borrowed/interest
- `test_side_effect_type_short_entry` — SELL + MARGIN_BUY
- `test_side_effect_type_short_exit` — BUY + AUTO_REPAY
- `test_side_effect_type_long_entry` — BUY + None
- `test_side_effect_type_stop_loss` — AUTO_REPAY
- `test_spot_mode_unchanged` — all spot methods called

### 10. Config & Documentation

- Add `BINANCE_ACCOUNT_TYPE=margin` to Railway production env
- Add to CLAUDE.md essential variables
- Update docs/live_trading.md

## Verification

1. `atb test unit` — all existing + new tests pass
2. `atb dev quality` — clean
3. Deploy to dev (paper mode) — verify startup logs:
   - "Margin mode enabled"
   - `tradeEnabled=True, borrowEnabled=True`
   - Symbol ETHUSDT margin-enabled
   - Correct USDT balance from margin wallet (via `get_balances()`)
4. Place a test order (paper mode) — verify symbol margin check fires and logs result
5. Deploy to production — verify:
   - Bot sees $100 USDT
   - First SELL signal → `create_margin_order` with `sideEffectType='MARGIN_BUY'`
   - SL placed with `AUTO_REPAY`
   - No accidental borrows on exits
5. Monitor Railway logs for margin-specific errors

## Branch

`feat/binance-margin-api`
