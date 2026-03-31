# Plan: Migrate from REST Polling to WebSocket Streams

## Context

The bot repeatedly gets IP-banned by Binance (-1003 "Way too much request weight used") because it polls REST APIs across 4 parallel cycles: main trading loop (60s), order tracker (10s), periodic reconciler (120s), and account sync. Each deploy triggers startup API calls that stack with ongoing polling, causing bans that prevent the bot from trading.

Binance recommends WebSocket streams for live updates. The python-binance library (v1.0.19) includes `ThreadedWebsocketManager` with full support for kline streams, user data streams (order fills, balance changes), and margin-specific streams. Railway supports long-lived WebSocket connections with no special config.

**Goal:** Replace REST polling with WebSocket event streams, eliminating ~73% of API weight usage initially (kline + order streams), with further reductions to ~97% in follow-up PRs (balance cache, reconciler interval). Sub-100ms event latency for order fills instead of 10s polling delay. Keep REST as a fallback.

## Architecture Overview

### Cache-First Design

WebSocket streams update an **in-memory state cache**. The existing trading engine **heartbeat loop continues** on a short cadence, polling the local cache instead of hitting the exchange REST API. This preserves all current trading semantics (PnL updates, trailing stops, partial exits, account snapshots, hot-swap checks) while eliminating network REST calls.

```
WebSocket Streams:                        Engine (unchanged heartbeat):

Kline Stream ──push──► KlineCache         Trading Loop ──60s──► read KlineCache
User Stream  ──push──► OrderEventRouter   │ update PnL, trailing stops, partials
  ├─ executionReport ──► OrderTracker     │ account snapshots, model hot-swap
  ├─ outboundAccountPosition ──► (future: BalanceCache)  │ check positions
  └─ margin events ──► OrderTracker       └─ strategy.process_candle(cached_df)

REST Reconciler ─2min─► Unchanged initially  OrderTracker ──event-fed──► fill/cancel callbacks
```

**Key principle:** WebSockets are a data source, not a scheduler. The engine heartbeat remains the sole scheduler for trading decisions.

### Connection State Machine

Each provider instance has its own `WebSocketState`. Possible states:

```
DISCONNECTED ──(start_stream)──► PRIMARY
PRIMARY ──(stale/disconnect/error)──► RESYNCING
RESYNCING ──(reconnect success → start_stream sets PRIMARY)──► PRIMARY
RESYNCING ──(reconnect failure)──► REST_DEGRADED
REST_DEGRADED ──(reconnect success → start_stream sets PRIMARY)──► PRIMARY
PRIMARY/RESYNCING ──(API ban -1003)──► SUSPENDED ──(ban expired)──► RESYNCING
```

- **WS_PRIMARY:** Normal operation. WebSocket streams update caches. REST used only for validation.
- **RESYNCING:** Gap detected. Behavior depends on mode:
  - **Live mode:** Full trading freeze — no new entries, no strategy evaluation, no exit monitoring from stale cache. Exchange-side stop-loss orders and the reconciler provide position safety during the brief resync window (typically seconds). Full REST reconciliation runs before resuming WS.
  - **Paper mode:** Immediate REST fallback — since there are no exchange-side SL orders or reconciler in paper mode, a trading freeze would leave positions unmonitored. Instead, paper mode skips the freeze and falls back to REST polling immediately (the data is simulated anyway, so gap risk is minimal).
  The engine checks `kline_provider.ws_state == RESYNCING` and branches on `self.enable_live_trading`.
- **REST_DEGRADED:** WebSocket unavailable for this stream. Falls back to current polling behavior for that stream's data. Note: kline and user streams degrade independently — kline WS can remain PRIMARY while user stream falls back to REST polling (or vice versa). "Only one mode owns state mutation" applies per-stream, not globally.

### Dual Processing Architecture

User-data events (fills, cancels, balance changes) are processed on a **dedicated thread** separate from kline/trading signals, preventing head-of-line blocking:

```python
# Thread 1: User-data event processor (high priority)
class UserDataProcessor(threading.Thread):
    """Processes executionReport and balance events with minimal latency."""
    def run(self):
        while running:
            event = user_data_queue.get(timeout=5)
            if event.type == "execution":
                order_tracker.process_execution_event(event.data)
            # Balance events logged for future BalanceCache integration
            elif event.type == "balance":
                logger.debug("Balance event received: %s", event.data.get("a", ""))

# Thread 2: Trading engine heartbeat (existing loop, unchanged cadence)
# See Phase 4 for detailed pseudocode. High-level flow:
while running:
    # 1. Check WS state — live mode freezes during resync, paper falls back to REST
    # 2. Get data from cache (WS active) or REST (fallback)
    # 3. Process strategy, check positions, update snapshots (all unchanged)
    # 4. Sleep with interrupt
```

### Idempotency & Deduplication

All fill/cancel processing is idempotent, keyed on `(orderId, executionType, I)` where `I` is Binance's execution ID field (NOT `t` which is -1 for non-trade events like cancels/rejects/expires):

```python
class EventDeduplicator:
    """Thread-safe tracker for processed events to prevent duplicate state mutations."""
    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._lock = threading.Lock()
        self._seen: OrderedDict[tuple[str, str, str], datetime] = OrderedDict()

    def is_duplicate(self, order_id: str, exec_type: str, exec_id: str) -> bool:
        key = (order_id, exec_type, exec_id)
        with self._lock:
            if key in self._seen:
                return True
            self._seen[key] = datetime.now(UTC)
            # Evict oldest entries when capacity exceeded
            while len(self._seen) > self._max_size:
                self._seen.popitem(last=False)
            return False
```

**Dedup key rationale:** Binance `executionReport` events include:
- `i` = orderId
- `x` = executionType (NEW, TRADE, CANCELED, REPLACED, REJECTED, EXPIRED)
- `I` = execution ID (unique per execution event; `t` is -1 for non-trade events)

Using `(i, x, I)` correctly deduplicates both trade fills and lifecycle events.

## Implementation Phases

### Phase 1: WebSocket Manager Layer (binance_provider.py)

Add WebSocket lifecycle to `BinanceProvider` as two independent streams (kline + user data), reusing the existing client configuration path (tld, testnet, margin settings). Kline stream works for both paper and live mode. User data stream requires credentials and is live-only. Both streams handle WS error events and track per-stream timestamps:

```python
class BinanceProvider:
    def _ensure_twm(self):
        """Lazily create the ThreadedWebsocketManager from existing config."""
        if self._twm is not None:
            return
        twm_kwargs = {"api_key": self.api_key, "api_secret": self.api_secret}
        if self.testnet:
            twm_kwargs["testnet"] = True
        api_endpoint = get_binance_api_endpoint()
        if api_endpoint == "binanceus":
            twm_kwargs["tld"] = "us"
        self._twm = ThreadedWebsocketManager(**twm_kwargs)
        self._twm.start()

    def start_kline_stream(self, symbol: str, timeframe: str, on_kline) -> bool:
        """Start kline stream only. Safe for paper mode (no credentials needed)."""
        try:
            self._ensure_twm()
            self._active_symbol = symbol
            self._active_timeframe = timeframe
            self._on_kline_cb = on_kline  # Store for reconnect

            def _kline_callback(msg):
                # Handle WS error events before routing to buffer
                if msg.get("e") == "error":
                    logger.error("Kline WS error: %s", msg.get("m", "unknown"))
                    self._on_ws_disconnect()
                    return
                self._last_kline_event_time = datetime.now(UTC)
                on_kline(msg)

            self._kline_socket_key = self._twm.start_kline_socket(
                callback=_kline_callback, symbol=symbol, interval=timeframe
            )
            self._ws_state = WebSocketState.PRIMARY
            self._last_kline_event_time = datetime.now(UTC)
            return True
        except Exception as e:
            logger.error("Failed to start kline stream: %s", e)
            return False

    def start_user_stream(self, on_user_event) -> bool:
        """Start user data stream. Requires credentials. Live mode only."""
        try:
            self._ensure_twm()
            self._on_user_event_cb = on_user_event  # Store for reconnect

            def _user_callback(msg):
                if msg.get("e") == "error":
                    logger.error("User data WS error: %s", msg.get("m", "unknown"))
                    self._on_ws_disconnect()
                    return
                self._last_user_event_time = datetime.now(UTC)
                on_user_event(msg)

            if self._use_margin:
                self._user_socket_key = self._twm.start_margin_socket(callback=_user_callback)
            else:
                self._user_socket_key = self._twm.start_user_socket(callback=_user_callback)
            self._last_user_event_time = datetime.now(UTC)
            self._ws_state = WebSocketState.PRIMARY  # User stream provider state

            # python-binance TWM handles listen key keepalive internally.
            return True
        except Exception as e:
            logger.error("Failed to start user data stream: %s", e)
            return False

    def stop_streams(self):
        """Stop all WebSocket streams. Uses manager-wide stop (recreates on reconnect)."""
        if self._twm:
            self._twm.stop()
            self._twm = None
            self._kline_socket_key = None
            self._user_socket_key = None
            self._ws_state = WebSocketState.DISCONNECTED

    @property
    def ws_state(self) -> WebSocketState:
        """Public read access to WebSocket connection state."""
        return self._ws_state

    @property
    def ws_healthy(self) -> bool:
        """Kline stream must be alive. User-data idleness is normal."""
        if self._ws_state != WebSocketState.PRIMARY:
            return False
        kline_age = (datetime.now(UTC) - self._last_kline_event_time).total_seconds()
        return kline_age < 120
```

**WebSocketState enum:**
```python
class WebSocketState(Enum):
    DISCONNECTED = "disconnected"
    PRIMARY = "primary"        # WS active, normal operation
    RESYNCING = "resyncing"    # Gap detected, running REST reconciliation
    REST_DEGRADED = "degraded" # WS failed, using REST polling
    SUSPENDED = "suspended"    # API ban active, waiting for ban expiry
```

**Files:** `src/data_providers/binance_provider.py`

### Phase 2: Kline Cache (kline_buffer.py)

WebSocket kline events update a local cache. The trading engine reads from this cache instead of calling `get_live_data()`.

```python
class KlineBuffer:
    """Thread-safe rolling kline history maintained by WebSocket events."""

    def __init__(self, symbol: str, timeframe: str, provider):
        self._lock = threading.Lock()
        # One-time REST fetch at startup (10 weight)
        self.df = provider.get_live_data(symbol, timeframe, limit=500)
        self._last_update = datetime.now(UTC)

    def on_kline(self, event: dict) -> None:
        """Process kline WebSocket event. Thread-safe.

        Handles both open-candle updates and candle-close transitions.
        On close: replaces the current tail row if timestamps match,
        then rolls the window only when a genuinely new timestamp appears.
        """
        kline = event.get("k", {})
        if not kline:
            return

        event_ts = pd.Timestamp(kline["t"], unit="ms", tz="UTC")

        with self._lock:
            tail_ts = self.df.index[-1] if len(self.df) > 0 else None

            if kline.get("x"):  # Candle closed
                closed_row = self._parse_kline(kline)
                if tail_ts is not None and event_ts == tail_ts:
                    # Replace current tail with final closed values
                    self.df.iloc[-1] = closed_row.iloc[0]
                else:
                    # New candle we haven't seen — roll window
                    self.df = pd.concat([self.df.iloc[1:], closed_row])
            else:
                # Open candle update — update current tail in-place
                if tail_ts is not None and event_ts == tail_ts:
                    self._update_current_candle(kline)
                elif tail_ts is not None and event_ts > tail_ts:
                    # First tick of a new candle — roll window to maintain fixed size
                    new_row = self._parse_kline(kline)
                    self.df = pd.concat([self.df.iloc[1:], new_row])
                # else: stale event for older candle, ignore

            self._last_update = datetime.now(UTC)

    def get_dataframe(self) -> pd.DataFrame:
        """Get current kline history. Thread-safe read."""
        with self._lock:
            return self.df.copy()

    @property
    def is_fresh(self) -> bool:
        """Check if cache has been updated recently."""
        age = (datetime.now(UTC) - self._last_update).total_seconds()
        return age < 120  # 2 min threshold

    def resync_from_rest(self, provider, symbol, timeframe):
        """Full REST resync after reconnection."""
        with self._lock:
            self.df = provider.get_live_data(symbol, timeframe, limit=500)
            self._last_update = datetime.now(UTC)
```

**Important:** The buffer updates the current (open) candle in-place so that intra-candle high/low values are available for SL/TP detection via `use_high_low_for_stops`. On candle close, it replaces the tail row (same timestamp) rather than blindly appending, preventing duplicate rows that would corrupt indicators.

**Files:** New file `src/engines/live/kline_buffer.py`

### Phase 3: Feed WebSocket Events into Existing OrderTracker (order_tracker.py)

**Do NOT rewrite OrderTracker.** Instead, add a method to accept WebSocket `executionReport` events and translate them into the same status/fill flow the polling path uses. The WS path constructs an `Order` object from WS fields and looks up the `TrackedOrder`, then delegates to the existing `_process_order_status(order_id, tracked, order)` method:

```python
class OrderTracker:
    def __init__(self, ..., event_deduplicator: EventDeduplicator | None = None):
        # ... existing init ...
        self._dedup = event_deduplicator or EventDeduplicator()

    def process_execution_event(self, event: dict) -> None:
        """Process an executionReport from WebSocket user data stream.

        Constructs an Order object from WS fields and delegates to the
        existing _process_order_status() method, preserving all lifecycle
        logic (partial fills, cancels, retries, invalid data, orphan prevention).
        """
        order_id = str(event.get("i", ""))
        exec_type = str(event.get("x", ""))   # executionType
        exec_id = str(event.get("I", ""))      # execution ID (NOT "t")

        # Idempotency check using correct Binance fields
        if self._dedup.is_duplicate(order_id, exec_type, exec_id):
            logger.debug("Duplicate execution event for order %s, skipping", order_id)
            return

        # Look up tracked order — skip if we don't know about it
        with self._lock:
            tracked = self._pending_orders.get(order_id)
        if tracked is None:
            logger.debug("Execution event for untracked order %s, ignoring", order_id)
            return

        # Validate payload fields
        cum_filled = float(event.get("z", 0))
        cum_quote = float(event.get("Z", 0))
        avg_price = cum_quote / cum_filled if cum_filled > 0 else 0.0

        # Map WS status — skip unknown statuses
        status = self._map_ws_status(str(event.get("X", "")))
        if status is None:
            logger.warning("Unknown WS order status: %s for order %s", event.get("X"), order_id)
            return

        # Construct full Order object matching what REST get_order() returns.
        # All 13 required fields must be provided (see exchange_interface.py:76).
        event_time = datetime.fromtimestamp(event.get("E", 0) / 1000, tz=UTC)
        order = Order(
            order_id=order_id,
            symbol=str(event.get("s", tracked.symbol)),
            side=OrderSide(str(event.get("S", "BUY"))),
            order_type=self._map_ws_order_type(str(event.get("o", "MARKET"))),
            quantity=float(event.get("q", 0)),
            price=float(event.get("p", 0)) or None,
            status=status,
            filled_quantity=cum_filled,
            average_price=avg_price,
            commission=float(event.get("n", 0)),       # Commission amount
            commission_asset=str(event.get("N", "")),   # Commission asset
            create_time=datetime.fromtimestamp(event.get("O", 0) / 1000, tz=UTC),
            update_time=event_time,
            stop_price=float(event.get("P", 0)) or None,
            time_in_force=str(event.get("f", "GTC")),
            client_order_id=str(event.get("c", "")),
        )

        # Delegate to existing method — preserves ALL lifecycle logic
        self._process_order_status(order_id, tracked, order)

    @staticmethod
    def _map_ws_status(ws_status: str) -> OrderStatus | None:
        """Map Binance WS order status string to internal OrderStatus.

        Returns None for unknown statuses (caller should log and skip).
        Note: Binance "NEW" maps to our PENDING; Binance "CANCELED" (one L)
        maps to our CANCELLED (two Ls).
        """
        mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(ws_status)

    @staticmethod
    def _map_ws_order_type(ws_type: str) -> OrderType:
        """Map Binance WS order type to internal OrderType.

        Binance sends types like STOP_LOSS_LIMIT that don't exist in our enum.
        Extends the existing _convert_order_type() mapping with WS-specific types.
        """
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
        }
        return mapping.get(ws_type, OrderType.MARKET)
```

**What's preserved:** All existing lifecycle logic including:
- Partial fill handling with invalid data detection
- Cancel callback with orphan prevention
- Callback retry with max retry escalation
- `stop_tracking` only after successful callback (FILLED) or unconditionally (CANCELED)

**What changes:** The `_poll_loop` thread becomes optional. When WebSocket is active, polling is disabled via `disable_polling()`. On WebSocket failure, `enable_polling()` resumes REST polling.

**Files:** `src/engines/live/order_tracker.py`

### Phase 4: Trading Engine Integration (trading_engine.py)

Minimal changes to the existing trading loop — replace REST data source with cache reads and add resync gating.

**Important runtime detail:** The engine has TWO BinanceProvider instances:
1. `self.data_provider` — typically a `CachedDataProvider` wrapping a `BinanceProvider` for market data. The underlying provider is at `self.data_provider.data_provider`.
2. `self.exchange_interface` — a separate `BinanceProvider` created only when `enable_live_trading=True`, used for orders/account operations.

For **kline streaming** (paper + live): Use the underlying market data provider (`self.data_provider.data_provider` if wrapped, else `self.data_provider`). This works in both paper and live mode.

For **user data streaming** (live only): Use `self.exchange_interface` (the authenticated provider with order/account access).

**Also:** The loop calls `update_live_data()` (a REST `get_klines(limit=1)` call) before `_get_latest_data()`. When WS kline cache is active, `update_live_data()` must also be skipped.

```python
class LiveTradingEngine:
    def start(self, symbol, timeframe, ...):
        # ... existing startup ...

        # Try WebSocket streams
        self._kline_buffer = None
        self._user_data_processor = None
        ws_started = False

        self._ws_kline_active = False  # Initialize flag

        # Resolve the underlying BinanceProvider for kline streaming.
        # Works in both paper and live mode (CachedDataProvider wraps it).
        kline_provider = getattr(self.data_provider, 'data_provider', self.data_provider)

        # Kline streaming: paper + live mode (reduces API weight for both)
        if hasattr(kline_provider, 'start_kline_stream'):
            self._kline_buffer = KlineBuffer(
                self._active_symbol, self.timeframe, self.data_provider)
            kline_started = kline_provider.start_kline_stream(
                symbol=self._active_symbol, timeframe=self.timeframe,
                on_kline=self._kline_buffer.on_kline,
            )
            if kline_started:
                self._ws_kline_active = True
                self._ws_kline_provider = kline_provider  # Store for health checks
                logger.info("Kline WebSocket stream active — REST data polling disabled")

        # User data streaming: live mode only (requires authenticated exchange_interface)
        if self.enable_live_trading and self.exchange_interface and \
           hasattr(self.exchange_interface, 'start_user_stream'):
            # Dedup lives inside OrderTracker (set during __init__).
            # UserDataProcessor routes raw events; OrderTracker.process_execution_event()
            # calls self._dedup.is_duplicate() before processing.
            self._user_data_processor = UserDataProcessor(
                order_tracker=self.order_tracker,
            )
            user_started = self.exchange_interface.start_user_stream(
                on_user_event=self._user_data_processor.enqueue,
            )
            if user_started:
                self._user_data_processor.start()
                if self.order_tracker:
                    self.order_tracker.disable_polling()
                logger.info("User data WebSocket stream active — order polling disabled")

        # ... continue with existing _trading_loop() ...

    def _get_latest_data(self, symbol, timeframe):
        """Fetch latest market data — from cache or REST."""
        ws_provider = getattr(self, '_ws_kline_provider', None)
        # During resync in live mode, return None to trigger skip-cycle
        # Paper mode falls back to REST immediately (handled in loop above)
        if self.enable_live_trading and ws_provider and \
           getattr(ws_provider, 'ws_state', None) == WebSocketState.RESYNCING:
            logger.info("WebSocket resyncing — skipping data fetch")
            return None

        if self._kline_buffer and self._kline_buffer.is_fresh and \
           getattr(ws_provider, 'ws_healthy', False):
            return self._kline_buffer.get_dataframe()

        # Fallback to REST (existing behavior)
        return self.data_provider.get_live_data(symbol, timeframe, limit=500)

    # ALSO: In the trading loop, gate both REST data calls on WS state:
    # Before the existing `update_live_data()` call:
    #   if not self._ws_kline_active:
    #       self.data_provider.update_live_data(symbol, timeframe)
    #
    # On REST_DEGRADED fallback, reset the flag:
    #   self._ws_kline_active = False
```

**Resync gating in the trading loop:** During RESYNCING, `_get_latest_data()` returns `None` and the loop skips the cycle entirely. This is a full trading freeze — no entries, no strategy evaluation, no exit monitoring from stale cache. Position safety is ensured by exchange-side stop-loss orders (already placed) and the reconciler's `reconcile_once()` during resync. The resync window is typically seconds (REST calls to refresh state), not minutes.

**What stays the same:**
- The `_trading_loop` heartbeat cadence and `_sleep_with_interrupt`
- All position checking, PnL updates, trailing stops, partial exits
- Account snapshots, strategy hot-swap, dynamic risk management
- `_is_context_ready()`, all safety checks

**`_is_data_fresh()` change:** The current implementation checks freshness from the last candle timestamp. For higher timeframes (e.g., 1h), the candle timestamp stays the same for up to an hour even while WS updates the open candle's OHLC values. This would cause WS-fed frames to be incorrectly marked stale. When WS kline cache is active, bypass the existing freshness check and use `KlineBuffer.is_fresh` instead (which checks `_last_update` time, not candle timestamp).

**What changes:**
- `_get_latest_data()` reads from local cache when available
- Returns `None` during RESYNCING to prevent trading on stale/partial state
- Order tracker switches between WS-fed and REST-polling modes
- On WS health failure, automatic fallback to REST with full resync

**Shutdown sequence** — extend existing `stop()` to clean up WS resources:
```python
def stop(self):
    # 1. Stop inbound WS streams FIRST (no new events arrive)
    # Kline stream lives on kline_provider (unwrapped data_provider)
    kline_provider = getattr(self, '_ws_kline_provider', None)
    if kline_provider and hasattr(kline_provider, 'stop_streams'):
        kline_provider.stop_streams()
    # User data stream lives on exchange_interface (separate instance)
    if self.exchange_interface and hasattr(self.exchange_interface, 'stop_streams'):
        self.exchange_interface.stop_streams()
    # 2. Stop/drain UserDataProcessor (process remaining queued events)
    if self._user_data_processor:
        self._user_data_processor.stop()
    # 3. Existing cleanup: reconciler → order tracker → positions → threads
    ...
```
Order matters: stop streams before tracker to prevent late WS callbacks racing against teardown.

**Files:** `src/engines/live/trading_engine.py`

### Phase 5: Reconciler Adjustments (reconciliation.py)

**Keep the reconciler running at current interval initially.** Only reduce after WebSocket path proves production parity (measured by: zero missed fills over 7 days, zero balance drift, zero orphaned positions).

Changes:
- Reconciler validates that WebSocket-reported state matches REST truth
- Logs discrepancies between WS cache and REST as metrics (for parity monitoring)
- Startup reconciliation unchanged — always runs via REST

**After parity proven (separate PR):**
- Increase interval to 300-600s
- Reconciler becomes validation-only, not primary detection

**Files:** `src/engines/live/reconciliation.py`, `src/config/constants.py`

### Phase 6: Connection Resilience

**Separation of concerns:** Each BinanceProvider instance owns its own `_ws_state`. Since kline and user streams run on different provider instances, their states are independent:
- **Kline provider** (`_ws_kline_provider`): `_ws_state` reflects kline stream health
- **Exchange interface** (`self.exchange_interface`): `_ws_state` reflects user-data stream health (live mode only)

Resync orchestration lives in `LiveTradingEngine`, which owns the buffer, tracker, and reconciler.

**BinanceProvider** (socket lifecycle):
```python
def _on_ws_disconnect(self):
    """Handle WebSocket disconnection. Sets state; engine handles resync."""
    self._ws_state = WebSocketState.RESYNCING
    logger.warning("WebSocket disconnected — entering RESYNCING state")
    # Engine's health monitor thread detects RESYNCING and calls
    # _handle_kline_disconnect() or _handle_user_stream_disconnect()

def reconnect_kline(self) -> bool:
    """Reconnect kline stream only. Called by engine on kline staleness."""
    try:
        # Stop and recreate TWM (manager-wide stop is the only clean option)
        self.stop_streams()
        return self.start_kline_stream(
            self._active_symbol, self._active_timeframe, self._on_kline_cb)
    except Exception as e:
        logger.error("Kline reconnect failed: %s", e)
        return False

def reconnect_user(self) -> bool:
    """Reconnect user data stream only. Called by engine on user-stream failure."""
    try:
        # Only restart user stream (kline may be on a different provider)
        if self._user_socket_key and self._twm:
            self._twm.stop_socket(self._user_socket_key)
        if self._on_user_event_cb:
            return self.start_user_stream(self._on_user_event_cb)
        return False
    except Exception as e:
        logger.error("User stream reconnect failed: %s", e)
        return False
```

**LiveTradingEngine** (two independent reconnect handlers):
```python
def _handle_kline_disconnect(self):
    """Handle kline stream failure. Called by health monitor."""
    kline_provider = getattr(self, '_ws_kline_provider', None)
    if not kline_provider:
        return
    # Resync kline history from REST
    if self._kline_buffer:
        self._kline_buffer.resync_from_rest(
            self.data_provider, self._active_symbol, self.timeframe)
    # Attempt kline reconnect
    if kline_provider.reconnect_kline():
        self._ws_kline_active = True
        logger.info("Kline WebSocket reconnected")
    else:
        kline_provider._ws_state = WebSocketState.REST_DEGRADED
        self._ws_kline_active = False
        logger.warning("Kline reconnect failed — REST polling resumed")

def _handle_user_stream_disconnect(self):
    """Handle user data stream failure. Called by health monitor (live only)."""
    if not self.enable_live_trading or not self.exchange_interface:
        return
    # Resync order and position state from REST
    if self.order_tracker:
        self.order_tracker.poll_once()
    if self._periodic_reconciler:
        self._periodic_reconciler.reconcile_once()
    # Attempt user stream reconnect (manages exchange_interface._ws_state)
    if hasattr(self.exchange_interface, 'reconnect_user') and \
       self.exchange_interface.reconnect_user():
        # reconnect_user() calls start_user_stream() which sets _ws_state = PRIMARY
        if self.order_tracker:
            self.order_tracker.disable_polling()
        logger.info("User data WebSocket reconnected")
    else:
        self.exchange_interface._ws_state = WebSocketState.REST_DEGRADED
        if self.order_tracker:
            self.order_tracker.enable_polling()
        logger.warning("User stream reconnect failed — order polling resumed")
```

**New public methods required:**
- `OrderTracker.poll_once()` — public wrapper calling `_check_orders()` once
- `PeriodicReconciler.reconcile_once()` — public wrapper calling `_reconcile_cycle()` once

**Health monitoring:** Daemon thread in `LiveTradingEngine` monitors both streams independently:
- **Kline stream** (on `kline_provider`): Staleness > 2 min triggers kline resync/reconnect.
- **User data stream** (on `exchange_interface`, live mode only): Idleness during quiet periods (no open orders) is normal. But if orders ARE being tracked (`order_tracker.get_tracked_count() > 0`) and no user-data events arrive for > 2 min, trigger user-stream reconnect and re-enable order polling as fallback. This prevents fills/cancels from being silently missed while order polling is disabled.

The two streams are monitored and reconnected **independently** since they live on different provider instances. If user-stream reconnect fails, `order_tracker.enable_polling()` is called immediately to resume REST polling for tracked orders.

**Per-stream timestamps** in BinanceProvider:
```python
self._last_kline_event_time: datetime   # Updated on every kline callback
self._last_user_event_time: datetime    # Updated on every user data callback

@property
def ws_healthy(self) -> bool:
    """Kline stream must be alive. User-data idleness is normal."""
    if self._ws_state != WebSocketState.PRIMARY:
        return False
    kline_age = (datetime.now(UTC) - self._last_kline_event_time).total_seconds()
    return kline_age < 120
```

**24-hour rotation:** Binance terminates WebSocket connections every 24 hours. The health monitor detects kline staleness and triggers the resync/reconnect flow.

**Files:** `src/data_providers/binance_provider.py`

### Phase 7: REST Fallback

The existing polling code stays intact and activates when WebSocket is unhealthy:

```python
# In _get_latest_data():
ws_provider = getattr(self, '_ws_kline_provider', None)
if self._kline_buffer and self._kline_buffer.is_fresh and \
   getattr(ws_provider, 'ws_healthy', False):
    return self._kline_buffer.get_dataframe()
else:
    # Current REST polling — unchanged
    return self.data_provider.get_live_data(symbol, timeframe, limit=500)

# In order tracker:
if ws_active:
    # Events arrive via process_execution_event()
    pass
else:
    # Resume REST polling (existing _poll_loop)
    self.enable_polling()
```

**Mutual exclusion (per-stream):** For each data stream (kline, user data), only one mode (WS or REST) owns state mutation at a time. Kline and user streams can independently be in different modes.

**WS→REST handoff for user data:** Before `enable_polling()`, the `UserDataProcessor` queue must be stopped and drained so no late WS events race with the first REST poll. Sequence: (1) stop UserDataProcessor, (2) drain remaining queue events through `process_execution_event()`, (3) then `enable_polling()`. The `EventDeduplicator` inside `OrderTracker` provides a secondary safety net — both WS and REST paths call `_process_order_status()` through the tracker, and the dedup check runs before any state mutation. However, the drain-then-switch handoff is the primary protection.

**Margin fail-fast preserved:** The existing fail-fast guard that prevents falling back to offline/mock mode when margin trading is enabled remains untouched. WebSocket fallback goes to REST polling, never to offline mode.

**API ban handling:** The existing `@with_rate_limit_retry` decorator is only used on order placement paths, not on `get_live_data()`/`get_order()`. Ban detection for the resync/fallback paths requires explicit handling: when `get_live_data()` or `get_order()` raises a `-1003` BinanceAPIException during resync, the engine catches it, parses the ban expiry timestamp (reusing the same parsing logic from `binance_provider.py:96`), sets `SUSPENDED` on both providers, and schedules retry after the ban window. The `_handle_kline_disconnect()` and `_handle_user_stream_disconnect()` methods wrap their REST calls in try/except for `-1003` and enter SUSPENDED on detection. An IP ban affects all connections from the same IP, so the engine coordinates setting SUSPENDED on both `_ws_kline_provider` and `exchange_interface`.

## Files to Modify

| File | Changes |
|------|---------|
| `src/data_providers/binance_provider.py` | Add split WS streams (kline + user), state machine, per-stream health, error event handling, reconnect |
| `src/data_providers/exchange_interface.py` | No changes needed — WS methods are Binance-specific, detected via `hasattr` on `exchange_interface` |
| `src/engines/live/trading_engine.py` | Replace `_get_latest_data()` with cache reads, resync orchestration, WS shutdown, health monitor |
| `src/engines/live/order_tracker.py` | Add `process_execution_event()`, `poll_once()` public method, toggle polling on/off |
| `src/engines/live/reconciliation.py` | Add `reconcile_once()` public method, WS parity monitoring (interval unchanged initially) |
| `src/config/constants.py` | Add WebSocket-related constants |
| `src/engines/live/kline_buffer.py` | **New:** Thread-safe rolling kline cache with correct close/roll logic |
| `src/engines/live/user_data_processor.py` | **New:** Dedicated thread for user-data event processing |
| `src/engines/live/event_deduplicator.py` | **New:** Thread-safe idempotent event tracking by (orderId, executionType, I) |

## Exchange Compatibility

**Binance:** Full WebSocket support via `ThreadedWebsocketManager` in python-binance. Kline streams, user data streams (spot + margin), automatic reconnection. This plan targets Binance.

**Coinbase:** Has its own WebSocket feed (`wss://ws-feed.exchange.coinbase.com`) with similar capabilities (ticker, trades, user orders). However, the API is completely different from Binance — separate implementation needed. The `CoinbaseProvider` currently has no WebSocket support and would continue using REST polling.

**Design:** WS stream methods (`start_kline_stream`, `start_user_stream`, `stop_streams`) are added directly to `BinanceProvider`, NOT to the `ExchangeInterface` ABC. The engine detects kline WS capability on the unwrapped market-data provider (`hasattr(kline_provider, 'start_kline_stream')`) and user-stream capability on the exchange interface (`hasattr(self.exchange_interface, 'start_user_stream')`). Providers without these methods (e.g., `CoinbaseProvider`) automatically fall back to REST polling. This means adding Coinbase WebSocket support later is additive, not blocking.

## Security

- **Margin fail-fast preserved:** The existing safety guard that prevents offline/mock fallback during margin trading remains untouched. WebSocket failure falls back to REST, never to offline mode.
- **No raw payload logging:** User-data events contain balances, symbols, client IDs, and execution details. Log only structured fields (order_id, status, filled_qty) at INFO level; full payloads only at DEBUG.
- **Payload validation:** All event handlers validate fields before mutation. `Z / z` division guarded by `z > 0`. Malformed or unknown status events are logged and skipped.
- **Event deduplication:** `(orderId, executionType, I)` keyed dedup prevents duplicate state mutations during WS/REST transitions, correctly handling both trade fills and non-trade lifecycle events (cancels, rejects, expires where `t = -1`).

## Railway Considerations

- **No extra services needed.** WebSocket connections are outbound from the bot — no inbound ports or services required beyond the existing health endpoint.
- **Railway supports long-lived connections.** No proxy timeouts or idle disconnects.
- **Healthcheck stays HTTP.** The `/health` endpoint continues working independently of WebSocket streams.
- **Deploy restart:** On redeploy, the bot reconnects WebSocket streams during startup. One-time REST fetch for kline history costs ~10 weight (vs current ~100+ per restart).

## API Weight Impact

| Component | Current (REST) | WebSocket | Savings |
|-----------|---------------|-----------|---------|
| Kline data | ~10 weight/min | 0 (stream) | 100% |
| Order status | ~12 weight/min per order | 0 (user stream) | 100% |
| Balance/account | ~3 weight/min | ~3 weight/min (unchanged initially) | 0% initially |
| Reconciler | ~5 weight/min | ~5 weight/min (unchanged initially) | 0% initially |
| Startup | ~30 weight | ~10 weight (one-time history) | 67% |
| **Total** | **~30 weight/min** | **~8 weight/min** | **73%** |

*After parity proven: reconciler interval increased to 300-600s (~1 weight/min), balance/account moved to WS BalanceCache (0 weight). Estimated ~1 weight/min total (~97% savings). BalanceCache and account sync replacement are separate follow-up PRs, not in initial scope.*

## Verification

### Unit Tests
1. Mock `ThreadedWebsocketManager` — verify stream start/stop lifecycle
2. KlineBuffer — verify rolling window with correct replace-then-roll on close, intra-candle updates, thread safety, no duplicate timestamps
3. OrderTracker — verify `process_execution_event()` constructs correct `Order` object and delegates to `_process_order_status()` for all status types:
   - FILLED (normal, zero-qty guard on avg_price)
   - PARTIALLY_FILLED (valid and invalid avg_price)
   - CANCELED, REJECTED, EXPIRED
   - Callback retry logic preserved
4. EventDeduplicator — verify dedup with `(orderId, executionType, I)` key, eviction, thread safety, correct handling of `t = -1` events
5. UserDataProcessor — verify event routing, queue processing
6. Connection state machine — verify all transitions, including RESYNCING blocks trading

### Failure Mode Tests
7. Duplicate `executionReport` events — verify idempotent processing
8. Out-of-order events — verify correct state after reordering
9. Partial fill then cancel — verify callback sequence and orphan prevention
10. Callback exception during WS-fed fill — verify retry behavior
11. Reconnect during open orders — verify no missed fills after resync
12. 24-hour connection rotation — verify automatic reconnect/resync
13. Stream termination mid-trade — verify REST fallback activates
14. RESYNCING state — verify live mode freezes, paper mode falls back to REST immediately

### Integration Tests
15. Connect to Binance testnet WebSocket — verify kline + execution events
16. Fallback test: Kill WebSocket — verify REST polling resumes within 2 min
17. Parity test: Compare WS-fed vs REST-polled state over extended run

### Deployment
18. Deploy to dev (paper mode): Verify no API bans over 24h, correct strategy execution
19. Deploy to production: Monitor for missed fills, balance drift, reconnection events

## Migration Strategy

Implement in phases, each independently deployable:
1. **Phase 1+2** (kline streams + cache) — eliminates main loop REST polling (works in both paper and live mode)
2. **Phase 3+4** (user data streams + order tracker integration) — eliminates order polling
3. **Phase 5** (reconciler monitoring) — adds WS/REST parity metrics
4. **Phase 6+7** (resilience + fallback) — production hardening

Each phase is a separate PR. Phase N+1 only starts after Phase N passes CI and paper-trading validation.

## Branch

`feat/websocket-streams`
