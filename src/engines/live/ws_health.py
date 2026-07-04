"""WebSocket health monitoring for the live trading engine.

Owns the resilience subsystem for the exchange WebSocket streams: starting the
kline + user-data streams, the background health-monitor thread, staleness
detection with REST fallback, reconnect/probe throttling (kline circuit limit;
user exponential backoff), the kline/user disconnect handlers, and draining the
deferred stop-loss-fill queue. Extracted from ``LiveTradingEngine`` (#486) so the
engine orchestrates while this monitor owns stream health.

Thread-safety / lock ownership (CRITICAL — preserved verbatim from the engine):

* The monitor runs ``ws_health_loop`` on a single background daemon thread
  (``state._ws_health_thread``), created by ``start_ws_health_monitor`` and
  signalled to stop via ``state.stop_event``; it is a daemon, so ``stop()`` does
  not join it (it exits at the next ``stop_event.wait`` boundary, ≤ the health
  interval).
* The reconnect counters and the kline-active flag
  (``_kline_reconnect_failures``, ``_user_reconnect_failures``,
  ``_ws_kline_active``) are written ONLY by that background thread and read
  atomically (GIL) by the trading loop — a lock-free single-writer model. This
  module keeps that exact model: the same background thread still writes those
  engine attributes (via the ``state`` backref), so no lock is added or needed.
* ``_pending_fill_exits`` is a thread-safe ``queue.SimpleQueue``: the
  OrderTracker poll thread enqueues, the trading loop drains via
  ``drain_pending_fill_exits``. Unchanged.
* Provider WebSocket state (``_user_ws_state``, ``user_ws_healthy``, etc.) is
  owned and locked by the data provider; the monitor only reads it.

The monitor holds no state of its own — every attribute and the thread handle
live on the engine and are accessed through ``state`` at call time, so the
threading model is byte-identical to the pre-extraction engine.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from src.data_providers.binance_provider import WebSocketState
from src.database.models import EventType

if TYPE_CHECKING:
    from src.engines.live.execution.position_tracker import LivePosition, LivePositionTracker
    from src.engines.live.kline_buffer import KlineBuffer
    from src.engines.live.order_tracker import OrderTracker
    from src.engines.live.reconciliation import PeriodicReconciler
    from src.engines.live.user_data_processor import UserDataProcessor

logger = logging.getLogger(__name__)


class WebSocketHealthEngineState(Protocol):
    """Engine state the WS health monitor reads and mutates at call time.

    The reconnect counters / kline-active flag are written only on the monitor's
    background thread and read atomically by the trading loop (lock-free
    single-writer); the thread handle and queue live here too. Accessed
    dynamically because streams/threads are created during ``start()``.
    """

    enable_live_trading: bool
    exchange_interface: Any
    data_provider: Any
    order_tracker: OrderTracker | None
    is_running: bool
    stop_event: threading.Event
    timeframe: str | None
    live_position_tracker: LivePositionTracker
    _active_symbol: str | None
    _periodic_reconciler: PeriodicReconciler | None
    _kline_buffer: KlineBuffer | None
    _ws_kline_provider: Any
    _user_data_processor: UserDataProcessor | None
    _ws_kline_active: bool
    _kline_reconnect_failures: int
    _user_reconnect_failures: int
    _ws_health_thread: threading.Thread | None
    _pending_fill_exits: queue.SimpleQueue

    def _execute_exit(
        self,
        position: LivePosition,
        reason: str,
        limit_price: float | None,
        current_price: float,
        candle_high: float | None,
        candle_low: float | None,
        candle: Any,
        skip_live_close: bool = ...,
    ) -> None: ...

    # Engine wrappers the monitor routes sibling calls through so test mocks on
    # the engine (patch.object(engine, "_handle_*"/"_start_ws_health_monitor"))
    # still intercept. Each delegates back to this monitor.
    def _start_ws_health_monitor(self) -> None: ...

    def _handle_kline_disconnect(self) -> None: ...

    def _handle_user_stream_disconnect(self, *, hard: bool = ...) -> None: ...

    def _record_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str = ...,
        component: str | None = ...,
        error_code: str | None = ...,
        exc: BaseException | None = ...,
        alert: bool = ...,
    ) -> None: ...


class WebSocketHealthMonitor:
    """Owns the live engine's WebSocket stream health + reconnect subsystem."""

    def __init__(self, engine_state: WebSocketHealthEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def start_websocket_streams(self, symbol: str, timeframe: str) -> None:
        """Initialize WebSocket streams for reduced API weight.

        Kline streaming works in both paper and live mode.
        User data streaming requires credentials (live mode only).
        Falls back gracefully if provider doesn't support WebSocket.
        """
        state = self._state
        from src.engines.live.kline_buffer import KlineBuffer
        from src.engines.live.user_data_processor import UserDataProcessor

        # Resolve the underlying BinanceProvider for kline streaming.
        # CachedDataProvider wraps it; unwrap to access WS methods.
        kline_provider = getattr(state.data_provider, "data_provider", state.data_provider)

        # Kline streaming: paper + live mode
        if hasattr(kline_provider, "start_kline_stream"):
            try:
                state._kline_buffer = KlineBuffer(symbol, timeframe, state.data_provider)
                kline_started = kline_provider.start_kline_stream(
                    symbol=symbol,
                    timeframe=timeframe,
                    on_kline=state._kline_buffer.on_kline,
                )
                if kline_started:
                    state._ws_kline_active = True
                    state._ws_kline_provider = kline_provider
                    logger.info("Kline WebSocket stream active — REST data polling disabled")
            except Exception as e:
                logger.warning("Failed to start kline WebSocket stream: %s", e)

        # User data streaming: live mode only
        if (
            state.enable_live_trading
            and state.exchange_interface
            and hasattr(state.exchange_interface, "start_user_stream")
        ):
            try:
                state._user_data_processor = UserDataProcessor(
                    order_tracker=state.order_tracker,
                )
                user_started = state.exchange_interface.start_user_stream(
                    on_user_event=state._user_data_processor.enqueue,
                )
                if user_started:
                    state._user_data_processor.start()
                    # Catch-up: reconcile to detect any events missed during handoff.
                    if state.order_tracker:
                        state.order_tracker.poll_once()
                    if state._periodic_reconciler:
                        state._periodic_reconciler.reconcile_once()
                    # Polling stays ON until the FIRST real user event confirms the
                    # socket is delivering. start_margin_socket is fire-and-forget and
                    # returns True even on a dead multiplexed ws_api socket, so disabling
                    # polling here would blackout order tracking on a never-delivering
                    # stream. _restore_user_ws_primary (the single disable site) flips it
                    # off once user_ws_healthy goes True on a health cycle (#717).
                    logger.info(
                        "User data WebSocket stream started — REST polling stays on "
                        "until the first event confirms delivery (#717)"
                    )
            except Exception as e:
                logger.warning("Failed to start user data WebSocket stream: %s", e)

        # Start health monitor if any stream is active
        if state._ws_kline_active or state._user_data_processor:
            state._start_ws_health_monitor()

    def start_ws_health_monitor(self) -> None:
        """Start daemon thread to monitor WebSocket stream health."""
        state = self._state
        state._ws_health_thread = threading.Thread(
            target=self.ws_health_loop, daemon=True, name="WSHealthMonitor"
        )
        state._ws_health_thread.start()
        logger.info("WebSocket health monitor started")

    def ensure_ws_health_monitor_alive(self) -> None:
        """Watchdog-on-watchdog: restart the WS health monitor if its thread died.

        The monitor is the lone WS-staleness watchdog; if its thread dies, stale
        streams go unnoticed and never reconnect. The main trading loop — itself
        liveness-tracked (#627) and crash-exiting (#630) — supervises it here (#631).
        The initial ``state._ws_health_thread`` write happens-before the loop thread
        is started, and thereafter only the loop thread writes it, so the two
        writers never overlap and no lock is needed. Respawns are naturally spaced
        (the monitor's grace-period wait) so a thread that re-dies immediately
        can't busy-respawn.
        """
        state = self._state
        # Only relevant once a stream the monitor watches has been started. Gate on
        # the provider's existence, not _ws_kline_active — the latter is toggled off
        # during a momentary kline REST fallback (#662), and the monitor must stay
        # supervised then too (it is what detects the WS recovering).
        if state._ws_kline_provider is None and not state._user_data_processor:
            return
        if not state.is_running or state.stop_event.is_set():
            return
        t = state._ws_health_thread
        if t is not None and t.is_alive():
            return
        logger.critical("WS health monitor thread is dead — restarting it (watchdog).")
        try:
            state._start_ws_health_monitor()
        except Exception as e:
            logger.error("Failed to restart WS health monitor: %s", e)

    def ws_health_loop(self) -> None:
        """Monitor WebSocket streams and trigger reconnection on failure."""
        state = self._state
        from src.config.constants import DEFAULT_WS_HEALTH_CHECK_INTERVAL

        # Grace period: skip the first check to let streams deliver initial events
        state.stop_event.wait(DEFAULT_WS_HEALTH_CHECK_INTERVAL)

        while state.is_running and not state.stop_event.is_set():
            try:
                self.check_kline_health()
                self.check_user_stream_health()
            except Exception as e:
                logger.error("WS health check error: %s", e, exc_info=True)
            state.stop_event.wait(DEFAULT_WS_HEALTH_CHECK_INTERVAL)

    def drain_pending_fill_exits(self) -> None:
        """Execute stop-loss-fill exits deferred from the OrderTracker poll thread.

        The poll thread enqueues only identifiers (it must stay fast and unblocked);
        the actual close runs here, on the trading loop, where exits already happen.
        Each item is isolated so one bad item can't abort the rest of the queue. A
        close that fails is logged and absorbed by ``_execute_exit`` itself, and the
        periodic/startup reconcilers (#628/#629) recover an unclosed position (they
        detect one whose stop-loss has filled on the exchange). The stop-loss has
        already executed on-exchange, so this is purely local bookkeeping and a
        small latency here is harmless (#631).
        """
        state = self._state
        while True:
            try:
                position_order_id, fill_price = state._pending_fill_exits.get_nowait()
            except queue.Empty:
                return
            try:
                position = state.live_position_tracker.get_position(position_order_id)
                if position is None:
                    logger.info(
                        "Deferred stop-loss exit skipped — position %s already closed",
                        position_order_id,
                    )
                    continue
                state._execute_exit(
                    position,
                    reason="stop_loss",
                    limit_price=fill_price,
                    current_price=float(fill_price),
                    candle_high=None,
                    candle_low=None,
                    candle=None,
                    skip_live_close=True,
                )
            except Exception as e:
                # _execute_exit logs and absorbs its own close failures (reconciliation
                # then recovers an unclosed position); reaching here means an unexpected
                # error in the drain itself (e.g. the position lookup). Isolate it so
                # one bad item can't abort the rest of the queue.
                logger.critical(
                    "CRITICAL: unexpected error draining deferred stop-loss exit for "
                    "position %s: %s. Left for reconciliation to recover.",
                    position_order_id,
                    e,
                    exc_info=True,
                )

    def check_kline_health(self) -> None:
        """Kline stream health with a *recovering* REST fallback (#662).

        REST is only ever a momentary fallback while the WS is down. The instant a
        real kline event resumes (``ws_healthy`` requires ``_kline_event_received``)
        we return to WS-primary and clear the breaker, so the bot can never get
        stuck on REST after a disconnect. Unlike the user-stream breaker (#616),
        this one keeps probing forever (throttled) rather than staying degraded
        until restart.

        Runs only on the WS health-monitor thread (the single writer of
        ``_ws_kline_active`` / ``_kline_reconnect_failures``), matching the existing
        lock-free convention; main-loop reads of the flag are GIL-atomic.
        """
        state = self._state
        provider = state._ws_kline_provider
        if not provider:
            return

        # Recovery: a real kline event arrived since the last reconnect, so the WS
        # is delivering again — return to WS-primary and reset the breaker.
        if getattr(provider, "ws_healthy", False):
            if not state._ws_kline_active:
                logger.info(
                    "Kline WebSocket recovered after %d health cycle(s) on REST — "
                    "REST polling disabled, WS primary again (#662)",
                    state._kline_reconnect_failures,
                )
            state._ws_kline_active = True
            state._kline_reconnect_failures = 0
            return

        # Unhealthy. Drop to REST immediately so the trading loop keeps trading
        # while we work to restore the WS. The data-fetch path already prefers REST
        # whenever the WS is not healthy, so this flips off the WS cache-warming
        # shortcut and records the degraded state (logged once per outage).
        if state._ws_kline_active:
            state._ws_kline_active = False
            provider.mark_kline_degraded()
            logger.warning(
                "Kline stream unhealthy — falling back to REST polling while reconnecting (#662)"
            )

        state._kline_reconnect_failures += 1
        if not self.should_probe_kline_reconnect(state._kline_reconnect_failures):
            return

        logger.warning(
            "Kline stream stale — attempting WS reconnect (failure #%d)",
            state._kline_reconnect_failures,
        )
        state._handle_kline_disconnect()

    def should_probe_kline_reconnect(self, failures: int) -> bool:
        """Whether to attempt a kline WS reconnect on this health cycle (#662).

        Fast phase (``failures`` <= circuit limit): probe every cycle for quick
        recovery. Throttled phase: probe only every Nth cycle so a persistently
        dead socket doesn't busy-loop on needless socket churn + REST resyncs.

        NEVER returns permanently False — past the limit it still returns True at
        every multiple of ``DEFAULT_WS_KLINE_DEGRADED_PROBE_EVERY``, so the WS
        always retains a path back to primary (the owner's "never REST-forever"
        requirement). Recovery itself is independent of this predicate: the
        ``ws_healthy`` branch in ``_check_kline_health`` restores WS-primary the
        instant a real event resumes, even on a throttled (non-probing) cycle.
        """
        from src.config.constants import (
            DEFAULT_WS_KLINE_DEGRADED_PROBE_EVERY,
            DEFAULT_WS_KLINE_RECONNECT_CIRCUIT_LIMIT,
        )

        if failures <= DEFAULT_WS_KLINE_RECONNECT_CIRCUIT_LIMIT:
            return True
        return failures % DEFAULT_WS_KLINE_DEGRADED_PROBE_EVERY == 0

    def check_user_stream_health(self) -> None:
        """User-stream health with a *recovering* REST fallback (#717).

        Mirrors the kline self-heal (#662): the engine falls back to REST order
        polling whenever the user/margin stream is down, but the instant a real
        user event resumes (``user_ws_healthy`` requires ``_user_event_received``)
        it returns to WS-primary and resets the breaker — so it can never get
        stuck on REST until restart, the absorbing-state bug of the original #616
        circuit breaker.

        Unlike kline, "WS-primary" here is not a per-cycle data-fetch choice: order
        events are pushed into the UserDataProcessor and the fallback is the
        OrderTracker REST poll loop, so WS-primary == ``order_tracker`` polling
        disabled. Recovery therefore rebuilds the processor + reconciles (in
        ``_handle_user_stream_disconnect``) and ``disable_polling`` is flipped only
        by ``_restore_user_ws_primary``, gated on a confirmed real event.

        Runs only on the WS health-monitor thread (the single writer of
        ``_user_reconnect_failures``), matching the lock-free convention; main-loop
        reads of the polling flag are GIL-atomic.
        """
        state = self._state
        if not state.enable_live_trading or not state.exchange_interface:
            return
        exchange = state.exchange_interface
        # Idleness is normal: with nothing tracked there is no staleness signal to
        # act on, no fills to miss, and no reason to churn the socket (which would
        # generate needless #716 teardown noise). REST polling, if on, keeps
        # backstopping order/balance state. The breaker only ever accumulates past
        # this gate, so it cannot be non-zero while idle. Gate first (#717).
        if not state.order_tracker or state.order_tracker.get_tracked_count() == 0:
            return

        # Recovery: a real user event arrived (user_ws_healthy requires
        # _user_event_received of the current generation), so the stream is
        # delivering — return to WS-primary and reset the breaker. Checked before
        # the state gate below so it also recovers from REST_DEGRADED, not just
        # PRIMARY. This is the self-heal #616 lacked (#717).
        if getattr(exchange, "user_ws_healthy", False):
            self.restore_user_ws_primary()
            state._user_reconnect_failures = 0
            return

        # RESYNCING (set by the error callback) needs an immediate recovery cycle.
        if getattr(exchange, "_user_ws_state", None) == WebSocketState.RESYNCING:
            logger.warning("User data stream in RESYNCING state — triggering recovery")
            state._handle_user_stream_disconnect()
            return

        from src.config.constants import (
            DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT,
            DEFAULT_WS_USER_STALENESS_THRESHOLD,
        )

        # REST_DEGRADED: the circuit is open and we are on REST polling. THE FIX —
        # instead of staying absorbed here until restart (#616), keep probing the
        # WS on a throttle so a recovered network path can return us to WS-primary.
        # The probe re-enters _handle_user_stream_disconnect (re-stop + reconnect);
        # the recovery branch above promotes us back the moment a real event lands.
        if getattr(exchange, "_user_ws_state", None) == WebSocketState.REST_DEGRADED:
            state._user_reconnect_failures += 1
            if not self.should_probe_user_reconnect(state._user_reconnect_failures):
                return
            # Decide the reconnect MODE here, BEFORE _handle_user_stream_disconnect's
            # stop_user_stream flips state to DISCONNECTED (the handler can no longer
            # read REST_DEGRADED). hard=True (full teardown + fresh AsyncClient/ws_api,
            # #723) only when the flag is on AND the kline/user coupling guard passes;
            # otherwise the cheap in-place re-subscribe. This is the ONLY caller that
            # may set hard=True.
            hard = self.should_hard_reconnect_user()
            logger.warning(
                "User data stream degraded — probing WS %s (failure #%d, "
                "next probe in ~%.0fm, #717/#723)",
                "HARD-reconnect" if hard else "reconnect",
                state._user_reconnect_failures,
                self.user_next_probe_eta_minutes(state._user_reconnect_failures),
            )
            state._handle_user_stream_disconnect(hard=hard)
            return

        # From here we require PRIMARY: anything else (DISCONNECTED/SUSPENDED) is
        # not a state this watchdog drives.
        if getattr(exchange, "_user_ws_state", None) != WebSocketState.PRIMARY:
            return

        last_event = getattr(exchange, "_last_user_event_time", None)
        if not last_event:
            return
        age = (datetime.now(UTC) - last_event).total_seconds()
        if age <= DEFAULT_WS_USER_STALENESS_THRESHOLD:
            return

        # Stale while PRIMARY with tracked orders means the previous reconnect
        # produced no real events. python-binance's start_margin_socket is
        # fire-and-forget and reports success even on a dead multiplexed ws_api
        # socket, so reconnect_user returns True and this watchdog would otherwise
        # reconnect every ~2 min forever (spewing asyncio re-entrancy errors).
        # After a few unproductive reconnects, open the circuit: tear down the dead
        # socket, mark REST_DEGRADED, and run REST-polling-only. Unlike #616, the
        # REST_DEGRADED branch above now keeps throttled-probing, so this is the
        # fast-phase boundary, not a terminal state (#717).
        if state._user_reconnect_failures >= DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT:
            logger.warning(
                "User data stream did not recover after %d reconnects — circuit open, "
                "falling back to REST polling and throttled-probing the WS (#717).",
                state._user_reconnect_failures,
            )
            # Tear down the dead user socket BEFORE marking degraded so the terminal
            # state is REST_DEGRADED, not DISCONNECTED (stop_user_stream sets
            # DISCONNECTED). Order matters: a DISCONNECTED terminal state would make
            # the REST_DEGRADED probe branch unreachable. stop_user_stream also halts
            # the dead socket's asyncio _read_ready spam (~2,100/hr) and bumps the
            # generation so any in-flight stale callback is dropped (#616/#717).
            if hasattr(exchange, "stop_user_stream"):
                exchange.stop_user_stream()
            if hasattr(exchange, "mark_user_degraded"):
                exchange.mark_user_degraded()
            state.order_tracker.enable_polling()
            # The real-time order/balance push feed is now dead; the bot runs on
            # slower REST polling until a real event returns it to primary. Page an
            # operator — degraded fill/balance visibility on a live account was
            # previously only a logger.warning (#717/#853). Fires once per
            # circuit-open: while REST_DEGRADED the method returns early above, so
            # this is edge-triggered by construction (no per-cycle spam).
            state._record_event(
                EventType.ALERT,
                f"User data stream circuit-open after {state._user_reconnect_failures} "
                "reconnects — REST-degraded; real-time fills/balance updates unavailable",
                severity="critical",
                component="connectivity",
                error_code="USER_WS_DEGRADED",
                alert=True,
            )
            return

        state._user_reconnect_failures += 1
        logger.warning(
            "User data stream stale (%ds) with tracked orders — reconnecting (attempt %d/%d)",
            int(age),
            state._user_reconnect_failures,
            DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT,
        )
        state._handle_user_stream_disconnect()

    def should_probe_user_reconnect(self, failures: int) -> bool:
        """Whether to probe a user-stream WS reconnect on this degraded cycle (#717/#723).

        Fast phase (``failures`` <= the circuit limit) probes every cycle for quick
        recovery. Throttled phase: probe on an *exponential-backoff* schedule of
        absolute failure-count boundaries (10, 20, 40, 80, 160, 280, 400, 520, …),
        not the old fixed every-Nth cadence, so an unrecoverable margin stream (the
        #616 dead multiplexed ws_api socket an in-place re-subscribe can't restore)
        stops churning ~200×/day while still being probed indefinitely.

        NEVER permanently False — past the geometric ramp the gap is fixed at the
        CEILING, so every CEILING-wide window of ``failures`` still contains a probe
        boundary; the WS therefore always retains a path back to primary (#717's
        guarantee). Recovery itself is independent of this predicate (the
        ``user_ws_healthy`` branch in ``_check_user_stream_health`` promotes the
        instant a real event resumes, even on a throttled non-probing cycle).

        Pure function of ``failures`` (no mutable backoff state), preserving the
        lock-free single-writer model: the boundary set is derived analytically from
        the three backoff constants on every call (see ``_user_probe_boundary_reached``).
        """
        from src.config.constants import DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT

        if failures <= DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT:
            return True
        return self.user_probe_boundary_reached(failures)

    @staticmethod
    def user_probe_boundary_reached(failures: int) -> bool:
        """True iff ``failures`` is an exponential-backoff probe boundary (#723).

        Boundaries start at ``FIRST`` and grow by gaps that double each step
        (``gap *= BASE``) until a gap would exceed ``CEILING``, after which the gap
        is fixed at ``CEILING`` forever — yielding the sequence
        ``FIRST, FIRST+FIRST, …`` = ``10, 20, 40, 80, 160, 280, 400, 520, 640, …``
        (gaps ``10, 20, 40, 80, 120, 120, …``) for the default constants. The set is
        generated ITERATIVELY from the constants (no hardcoded list, no closed form —
        a cumulative-sum closed form would give the wrong ``10, 30, 70, 150, …``): we
        walk boundaries up to ``failures`` and test exact equality. Membership is the
        whole predicate, so the walk is O(log(failures/FIRST)) up to the ramp then
        O((failures-ramp)/CEILING) — cheap at the once-per-30-s health-check cadence.

        The post-ramp gap equals the constant ``CEILING``, so every window of
        ``CEILING`` consecutive failure counts contains exactly one boundary — the
        never-permanently-False invariant the user stream relies on (#717).
        """
        from src.config.constants import (
            DEFAULT_WS_USER_DEGRADED_PROBE_BACKOFF_BASE as BASE,
        )
        from src.config.constants import (
            DEFAULT_WS_USER_DEGRADED_PROBE_CEILING as CEILING,
        )
        from src.config.constants import (
            DEFAULT_WS_USER_DEGRADED_PROBE_FIRST_BOUNDARY as FIRST,
        )

        if failures < FIRST:
            return False
        boundary = FIRST
        gap = FIRST
        # Advance to the first boundary >= failures, growing the gap geometrically
        # (capped at CEILING). The order — add the gap, THEN grow it — makes the first
        # increment FIRST (10→20), matching the pinned 10,20,40,80,160,280,… schedule.
        while boundary < failures:
            boundary += gap
            gap = min(gap * BASE, CEILING)
        return boundary == failures

    @staticmethod
    def user_next_probe_eta_minutes(failures: int) -> float:
        """Approx minutes until the next user degraded-probe after ``failures`` (#723).

        Diagnostic only: the gap (in health cycles) from the current failure count to
        the next exponential-backoff boundary, scaled by the health-check interval, so
        the degraded-probe log shows the backoff widening (e.g. "next probe in ~5m" →
        "~10m" → "~60m") for prod monitoring. Derived from the same constants as the
        boundary set, so it can never drift from the actual probe cadence.
        """
        from src.config.constants import (
            DEFAULT_WS_HEALTH_CHECK_INTERVAL as INTERVAL,
        )
        from src.config.constants import (
            DEFAULT_WS_USER_DEGRADED_PROBE_BACKOFF_BASE as BASE,
        )
        from src.config.constants import (
            DEFAULT_WS_USER_DEGRADED_PROBE_CEILING as CEILING,
        )
        from src.config.constants import (
            DEFAULT_WS_USER_DEGRADED_PROBE_FIRST_BOUNDARY as FIRST,
        )

        # Walk to the first boundary strictly past `failures`; the cycle gap to it,
        # times the seconds-per-cycle, is the ETA. (At a boundary, this returns the
        # gap to the NEXT one — the spacing that will apply after this probe fires.)
        boundary = FIRST
        gap = FIRST
        while boundary <= failures:
            boundary += gap
            gap = min(gap * BASE, CEILING)
        return (boundary - failures) * INTERVAL / 60.0

    def should_hard_reconnect_user(self) -> bool:
        """Whether the degraded probe should use a HARD reconnect this cycle (#723).

        True only when ALL hold:
        1. ``FEATURE_WS_USER_HARD_RECONNECT`` is enabled (default OFF — money/stability-
           adjacent, ships inert per LESSONS §3/§4). A missing env var resolves False.
        2. The kline and user streams are on SEPARATE providers (B-VERIFY-1). The hard
           path does a full ``exchange_interface.stop_streams()``; if a future refactor
           ever co-located kline on the same provider (``_ws_kline_provider is
           exchange_interface``), that teardown would also kill kline. We REFUSE the
           hard path in that case — log CRITICAL and fall back to the cheap in-place
           reconnect — rather than a bare ``assert`` that would crash the WS health
           thread (the codex guard, #723).
        3. The provider actually exposes ``hard_reconnect_user`` (defensive).

        Pure read; safe on the WS health thread (no state mutation, never raises).
        """
        state = self._state
        from src.config.feature_flags import is_enabled

        if not is_enabled("ws_user_hard_reconnect", False):
            return False
        if not hasattr(state.exchange_interface, "hard_reconnect_user"):
            return False
        # Coupling guard: a shared provider would let the user teardown kill kline.
        if (
            state._ws_kline_provider is not None
            and state._ws_kline_provider is state.exchange_interface
        ):
            logger.critical(
                "Kline and user streams share a provider — refusing hard reconnect, "
                "using in-place reconnect (#723)"
            )
            return False
        return True

    def restore_user_ws_primary(self) -> None:
        """Return the user stream to WS-primary: disable REST polling once a real
        event confirms delivery (#717).

        This is the SINGLE site that disables user-stream order polling, for both
        fresh boot and post-degradation recovery. It is gated on
        ``order_tracker.is_polling_enabled()`` — NOT on ``_user_ws_state``, which
        flips to PRIMARY in ``start_user_stream`` *before* the first real event, so
        gating on it would disable polling on a never-delivering socket (the
        blackout this fix prevents). When polling is already off (steady state) this
        is a no-op, so it does not spam a "recovered" log every health cycle.
        """
        state = self._state
        if not state.order_tracker or not state.order_tracker.is_polling_enabled():
            return
        # A real user event arrived while polling was still on — the WS is now the
        # primary order-event path. poll_once first closes any gap between the last
        # poll and the handoff so no fill is missed.
        state.order_tracker.poll_once()
        state.order_tracker.disable_polling()
        logger.info(
            "User data WebSocket recovered — real event confirmed, REST polling "
            "disabled, WS primary again (#717)"
        )
        # Paired recovery signal so an operator sees the degraded→recovered
        # transition in system_events (no page — recovery is good news). #853
        state._record_event(
            EventType.WARNING,
            "User data WebSocket recovered — WS primary again, REST polling disabled",
            severity="info",
            component="connectivity",
            error_code="USER_WS_RECOVERED",
        )

    def handle_kline_disconnect(self) -> None:
        """Resync kline history from REST and attempt one WS reconnect.

        Does NOT decide WS-primary vs REST — that belongs to ``_check_kline_health``,
        which returns to WS-primary only once a real event confirms ``ws_healthy``
        (#662). ``reconnect_kline`` returning True only means a socket was *opened*,
        not that it delivers data (the #650/#663 failure signature), so trusting it
        here is exactly what caused the 30s "reconnected" churn.
        """
        state = self._state
        provider = state._ws_kline_provider
        if not provider:
            return
        # Resync kline history from REST so the buffer has no gap, whether we end up
        # recovering the WS or keep serving the loop from REST in the meantime.
        if state._kline_buffer:
            try:
                # start() sets _active_symbol/timeframe before the buffer exists.
                state._kline_buffer.resync_from_rest(
                    state.data_provider,
                    cast(str, state._active_symbol),
                    cast(str, state.timeframe),
                )
            except Exception as e:
                logger.error("Kline REST resync failed: %s", e)
        # Attempt reconnect (bounded internally by the REST socket timeout, #631).
        # Whether it actually restored event flow is confirmed next health cycle via
        # ws_healthy; we deliberately do not treat the return value as "healthy".
        try:
            if hasattr(provider, "reconnect_kline"):
                provider.reconnect_kline()
        except Exception as e:
            logger.error("Kline reconnect attempt failed: %s", e)

    def handle_user_stream_disconnect(self, *, hard: bool = False) -> None:
        """Handle user data stream failure. Resync orders and attempt reconnect.

        ``hard`` selects the reconnect strategy and MUST be decided by the caller:
        this handler's first action (``stop_user_stream``) flips ``_user_ws_state`` to
        DISCONNECTED, so the handler can no longer read "were we REST_DEGRADED?" itself
        (the codex ordering fix, #723). When ``hard`` is True it does a full
        teardown + fresh-AsyncClient rebuild (``hard_reconnect_user``); otherwise the
        cheap in-place re-subscribe (``reconnect_user``). Only the REST_DEGRADED
        throttled-probe branch passes ``hard=True`` (and only when
        ``FEATURE_WS_USER_HARD_RECONNECT`` is on AND the kline/user coupling guard
        passes); the fast-phase, RESYNCING, and PRIMARY-stale callers pass ``hard=False``
        so a flag-on PRIMARY/RESYNCING cycle still uses the cheap path.
        """
        state = self._state
        from src.engines.live.user_data_processor import UserDataProcessor

        if not state.enable_live_trading or not state.exchange_interface:
            return
        # 1. Stop the old user socket FIRST to prevent new events arriving. (When
        #    hard=True the subsequent hard_reconnect_user does a full stop_streams
        #    teardown anyway; this early stop_user_stream keeps the state machine /
        #    generation bump identical across both paths and is harmless/idempotent.)
        if hasattr(state.exchange_interface, "stop_user_stream"):
            state.exchange_interface.stop_user_stream()
        # 2. Now drain the UserDataProcessor (no new events can arrive)
        processor_clean = True
        if state._user_data_processor:
            processor_clean = state._user_data_processor.stop()
            state._user_data_processor = None
        # 3. Enable REST polling as fallback
        if state.order_tracker:
            state.order_tracker.enable_polling()
        # 4. Resync order and position state from REST. Bounded by the REST
        #    socket timeout (#631); a failed resync must not abort the rest of
        #    recovery — REST polling (step 3) keeps orders tracked meanwhile.
        try:
            if state.order_tracker:
                state.order_tracker.poll_once()
            if state._periodic_reconciler:
                state._periodic_reconciler.reconcile_once()
        except Exception as e:
            logger.error("User-stream REST resync failed: %s", e)
        # 5. If processor didn't stop cleanly, stay degraded — don't reconnect
        #    while the old thread may still be mutating order state
        if not processor_clean:
            state.exchange_interface.mark_user_degraded()
            logger.critical("UserDataProcessor did not stop cleanly — staying in REST_DEGRADED")
            return
        # 6. Attempt user stream reconnect with fresh callback. `hard` (decided by the
        #    caller, see docstring) picks a full teardown + fresh-AsyncClient rebuild
        #    (#723) over the cheap in-place re-subscribe. Both are fire-and-forget: a
        #    True return means a socket opened, NOT that it delivers — recovery stays
        #    gated on a real event via _restore_user_ws_primary (the single disable
        #    site), so neither path can blackout order tracking on a dead socket (#717).
        reconnect_method = "hard_reconnect_user" if hard else "reconnect_user"
        reconnected = False
        if hasattr(state.exchange_interface, reconnect_method):
            try:
                new_processor = UserDataProcessor(
                    order_tracker=state.order_tracker,
                )
                reconnect_fn = getattr(state.exchange_interface, reconnect_method)
                if reconnect_fn(on_user_event=new_processor.enqueue):
                    state._user_data_processor = new_processor
                    state._user_data_processor.start()
                    reconnected = True
                    logger.info(
                        "User data WebSocket %s",
                        (
                            "hard-reconnected (fresh AsyncClient/ws_api, #723)"
                            if hard
                            else "reconnected"
                        ),
                    )
                    # Post-reconnect catch-up: reconcile events from the handoff gap
                    # so nothing is lost. Polling deliberately stays ON — a reconnect
                    # only re-OPENS the socket; reconnect_user/start_margin_socket is
                    # fire-and-forget and may return success on a dead socket. Only a
                    # real event (user_ws_healthy) on a later health cycle restores
                    # WS-primary via _restore_user_ws_primary, the single disable site.
                    # This closes the post-reconnect blackout window (#717).
                    try:
                        if state.order_tracker:
                            state.order_tracker.poll_once()
                        if state._periodic_reconciler:
                            state._periodic_reconciler.reconcile_once()
                    except Exception as e:
                        logger.error(
                            "Post-reconnect catch-up failed (staying on REST polling): %s", e
                        )
            except Exception as e:
                logger.error("User stream reconnect attempt failed: %s", e)
        if not reconnected:
            state.exchange_interface.mark_user_degraded()
            logger.warning("User stream reconnect failed — order polling resumed")
