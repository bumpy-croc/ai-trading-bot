"""Tests for WebSocket health monitoring in LiveTradingEngine."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_providers.binance_provider import WebSocketState


@pytest.fixture
def mock_engine():
    """Create a LiveTradingEngine with mocked dependencies for WS health testing."""
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        from src.engines.live.trading_engine import LiveTradingEngine

        mock_strategy = MagicMock()
        mock_strategy.get_risk_overrides.return_value = {}
        mock_strategy.__class__.__name__ = "MockStrategy"
        mock_strategy.config = {}

        mock_dp = MagicMock()
        mock_dp.get_live_data.return_value = _make_df()
        mock_dp.get_current_price.return_value = 50000.0

        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_dp,
            enable_live_trading=False,
            initial_balance=1000.0,
            enable_dynamic_risk=False,
            enable_hot_swapping=False,
        )
        engine._active_symbol = "BTCUSDT"
        engine.timeframe = "1h"
        return engine


def _make_df(rows: int = 5) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    now = datetime.now(UTC)
    idx = pd.date_range(end=now, periods=rows, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [50000.0] * rows,
            "high": [51000.0] * rows,
            "low": [49000.0] * rows,
            "close": [50500.0] * rows,
            "volume": [100.0] * rows,
        },
        index=idx,
    )


class TestCheckKlineHealth:
    """Tests for _check_kline_health()."""

    @pytest.mark.fast
    def test_triggers_disconnect_when_unhealthy(self, mock_engine):
        """Unhealthy first cycle: drop to REST (flag off + degrade once) and reconnect."""
        mock_provider = MagicMock()
        mock_provider.ws_healthy = False
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._ws_kline_active = True
        mock_engine._kline_reconnect_failures = 0

        with patch.object(mock_engine, "_handle_kline_disconnect") as mock_disconnect:
            mock_engine._check_kline_health()
            mock_disconnect.assert_called_once()
            # The degrade-on-first-unhealthy step must not regress silently.
            assert mock_engine._ws_kline_active is False
            mock_provider.mark_kline_degraded.assert_called_once()

    @pytest.mark.fast
    def test_does_nothing_when_healthy(self, mock_engine):
        """Should not trigger disconnect when ws_healthy is True."""
        mock_provider = MagicMock()
        mock_provider.ws_healthy = True
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._ws_kline_active = True

        with patch.object(mock_engine, "_handle_kline_disconnect") as mock_disconnect:
            mock_engine._check_kline_health()
            mock_disconnect.assert_not_called()

    @pytest.mark.fast
    def test_does_nothing_when_no_provider(self, mock_engine):
        """Should return early when no kline provider exists."""
        mock_engine._ws_kline_provider = None
        mock_engine._ws_kline_active = False

        with patch.object(mock_engine, "_handle_kline_disconnect") as mock_disconnect:
            mock_engine._check_kline_health()
            mock_disconnect.assert_not_called()


class TestCheckUserStreamHealth:
    """Tests for _check_user_stream_health()."""

    @pytest.mark.fast
    def test_triggers_disconnect_when_stale_with_tracked_orders(self, mock_engine):
        """Should reconnect when user stream is stale and orders are tracked."""
        mock_engine.enable_live_trading = True
        mock_exchange = MagicMock()
        mock_exchange._user_ws_state = WebSocketState.PRIMARY
        mock_exchange._last_user_event_time = datetime.now(UTC) - timedelta(seconds=300)
        mock_engine.exchange_interface = mock_exchange

        mock_tracker = MagicMock()
        mock_tracker.get_tracked_count.return_value = 2
        mock_engine.order_tracker = mock_tracker

        with patch.object(mock_engine, "_handle_user_stream_disconnect") as mock_disconnect:
            mock_engine._check_user_stream_health()
            mock_disconnect.assert_called_once()

    @pytest.mark.fast
    def test_does_nothing_when_no_tracked_orders(self, mock_engine):
        """Should not trigger disconnect when no orders are tracked (idleness is normal)."""
        mock_engine.enable_live_trading = True
        mock_exchange = MagicMock()
        mock_exchange._user_ws_state = WebSocketState.PRIMARY
        mock_exchange._last_user_event_time = datetime.now(UTC) - timedelta(seconds=300)
        mock_engine.exchange_interface = mock_exchange

        mock_tracker = MagicMock()
        mock_tracker.get_tracked_count.return_value = 0
        mock_engine.order_tracker = mock_tracker

        with patch.object(mock_engine, "_handle_user_stream_disconnect") as mock_disconnect:
            mock_engine._check_user_stream_health()
            mock_disconnect.assert_not_called()

    @staticmethod
    def _dead_socket_engine(mock_engine):
        """Engine whose user stream is PRIMARY+stale and never gets a real event
        (user_ws_healthy False) — the #616 dead multiplexed ws_api socket."""
        mock_engine.enable_live_trading = True
        ex = MagicMock()
        ex._user_ws_state = WebSocketState.PRIMARY
        ex._last_user_event_time = datetime.now(UTC) - timedelta(seconds=300)
        ex.user_ws_healthy = False
        mock_engine.exchange_interface = ex
        tracker = MagicMock()
        tracker.get_tracked_count.return_value = 2
        mock_engine.order_tracker = tracker
        return ex, tracker

    @pytest.mark.fast
    def test_breaker_trips_after_limit_unproductive_reconnects(self, mock_engine):
        """After LIMIT dead reconnects, stop reconnecting and degrade to REST (#616)."""
        from src.config.constants import DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT as LIMIT

        ex, tracker = self._dead_socket_engine(mock_engine)
        with patch.object(mock_engine, "_handle_user_stream_disconnect") as disc:
            for _ in range(LIMIT):
                mock_engine._check_user_stream_health()
            assert disc.call_count == LIMIT
            assert mock_engine._user_reconnect_failures == LIMIT
            ex.mark_user_degraded.assert_not_called()

            # Next cycle: circuit open — degrade to REST, no further reconnect.
            mock_engine._check_user_stream_health()
            assert disc.call_count == LIMIT  # unchanged
            ex.mark_user_degraded.assert_called_once()
            # The dead socket is torn down so its asyncio _read_ready spam stops.
            ex.stop_user_stream.assert_called_once()
            tracker.enable_polling.assert_called()
            # Teardown must happen BEFORE degrade, else the terminal state is
            # DISCONNECTED instead of REST_DEGRADED and the one-shot guard breaks.
            ordered = [
                c[0] for c in ex.mock_calls if c[0] in ("stop_user_stream", "mark_user_degraded")
            ]
            assert ordered == ["stop_user_stream", "mark_user_degraded"]

    @pytest.mark.fast
    def test_breaker_resets_on_real_event(self, mock_engine):
        """A genuinely healthy stream (real event) clears the failure counter."""
        mock_engine.enable_live_trading = True
        ex = MagicMock()
        ex._user_ws_state = WebSocketState.PRIMARY
        ex.user_ws_healthy = True  # a real user event arrived
        ex._last_user_event_time = datetime.now(UTC)  # fresh
        mock_engine.exchange_interface = ex
        tracker = MagicMock()
        tracker.get_tracked_count.return_value = 2
        mock_engine.order_tracker = tracker
        mock_engine._user_reconnect_failures = 2  # had prior failures

        with patch.object(mock_engine, "_handle_user_stream_disconnect") as disc:
            mock_engine._check_user_stream_health()
            assert mock_engine._user_reconnect_failures == 0  # reset by healthy stream
            disc.assert_not_called()  # not stale → no reconnect

    @pytest.mark.fast
    def test_degraded_state_short_circuits(self, mock_engine):
        """Once REST_DEGRADED, the check returns early (warning logged only once)."""
        mock_engine.enable_live_trading = True
        ex = MagicMock()
        ex._user_ws_state = WebSocketState.REST_DEGRADED
        mock_engine.exchange_interface = ex
        tracker = MagicMock()
        tracker.get_tracked_count.return_value = 2
        mock_engine.order_tracker = tracker

        with patch.object(mock_engine, "_handle_user_stream_disconnect") as disc:
            mock_engine._check_user_stream_health()
            disc.assert_not_called()
            ex.mark_user_degraded.assert_not_called()

    @pytest.mark.fast
    def test_reconnect_calls_bounded_over_many_cycles(self, mock_engine):
        """Regression for the #616 churn: a dead socket yields BOUNDED reconnects,
        not an unbounded ~2-min loop."""
        from src.config.constants import DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT as LIMIT

        ex, tracker = self._dead_socket_engine(mock_engine)
        # mark_user_degraded flips state to REST_DEGRADED, as the real method does.
        ex.mark_user_degraded.side_effect = lambda: setattr(
            ex, "_user_ws_state", WebSocketState.REST_DEGRADED
        )
        with patch.object(mock_engine, "_handle_user_stream_disconnect") as disc:
            for _ in range(20):
                mock_engine._check_user_stream_health()

        assert disc.call_count == LIMIT  # bounded, not 20
        ex.mark_user_degraded.assert_called_once()  # degraded exactly once
        ex.stop_user_stream.assert_called_once()  # dead socket torn down (asyncio spam stops)

    @pytest.mark.fast
    def test_breaker_survives_post_reconnect_fresh_timestamp(self, mock_engine):
        """Real-timing regression: each reconnect refreshes _last_user_event_time to
        now (as start_user_stream does), so the next health checks see age<threshold
        and skip. The breaker must still accumulate across stale cycles — only a real
        event (user_ws_healthy) may reset it, never the fresh post-reconnect timestamp.
        """
        from src.config.constants import DEFAULT_WS_USER_RECONNECT_CIRCUIT_LIMIT as LIMIT

        ex, _tracker = self._dead_socket_engine(mock_engine)

        # Faithful reconnect: refresh the timestamp but leave the socket dead — no
        # real event arrives, so user_ws_healthy stays False.
        def _reconnect():
            ex._last_user_event_time = datetime.now(UTC)

        with patch.object(
            mock_engine, "_handle_user_stream_disconnect", side_effect=_reconnect
        ) as disc:
            for cycle in range(1, LIMIT + 1):
                # Stale again (simulate ~150s elapsed since the last reconnect refresh).
                ex._last_user_event_time = datetime.now(UTC) - timedelta(seconds=300)
                mock_engine._check_user_stream_health()  # increment + reconnect (refreshes ts)
                assert mock_engine._user_reconnect_failures == cycle
                # Immediately after: fresh timestamp → NOT stale → no-op; the counter
                # must be PRESERVED (the fresh timestamp must not reset it).
                mock_engine._check_user_stream_health()
                assert mock_engine._user_reconnect_failures == cycle
                assert disc.call_count == cycle

            # Next stale cycle: circuit trips → degrade, no further reconnect.
            ex._last_user_event_time = datetime.now(UTC) - timedelta(seconds=300)
            mock_engine._check_user_stream_health()
            ex.mark_user_degraded.assert_called_once()
            assert disc.call_count == LIMIT


class TestHandleKlineDisconnect:
    """Tests for _handle_kline_disconnect().

    Post-#662 the handler is *state-free*: it resyncs from REST and fires one
    reconnect attempt, but it does NOT decide WS-primary vs REST or mark the
    stream degraded — that is owned by _check_kline_health, which returns to
    WS-primary only once a real event confirms ws_healthy. Trusting
    reconnect_kline's return value (a socket opened, not events flowing) here is
    exactly what caused the 30s reconnect churn.
    """

    @pytest.mark.fast
    def test_resyncs_buffer_and_attempts_reconnect(self, mock_engine):
        """Resyncs from REST and calls reconnect_kline, without owning WS state."""
        mock_provider = MagicMock()
        mock_provider.reconnect_kline.return_value = True
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._ws_kline_active = False

        mock_buffer = MagicMock()
        mock_engine._kline_buffer = mock_buffer

        mock_engine._handle_kline_disconnect()

        mock_buffer.resync_from_rest.assert_called_once_with(
            mock_engine.data_provider, "BTCUSDT", "1h"
        )
        mock_provider.reconnect_kline.assert_called_once()
        # Does NOT flip the flag — even on a "successful" reconnect, only a real
        # event (checked next health cycle) may restore WS-primary (#662).
        assert mock_engine._ws_kline_active is False

    @pytest.mark.fast
    def test_does_not_mark_degraded_or_flip_state_on_failure(self, mock_engine):
        """A failed reconnect does NOT degrade here — _check_kline_health owns that
        (it already dropped to REST before calling us). The handler just returns."""
        mock_provider = MagicMock()
        mock_provider.reconnect_kline.return_value = False
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._ws_kline_active = True  # caller's state; handler must not change it

        mock_buffer = MagicMock()
        mock_engine._kline_buffer = mock_buffer

        mock_engine._handle_kline_disconnect()

        mock_provider.mark_kline_degraded.assert_not_called()
        assert mock_engine._ws_kline_active is True  # unchanged by the handler

    @pytest.mark.fast
    def test_reconnect_exception_does_not_propagate(self, mock_engine):
        """A raising reconnect (e.g. REST timeout) is swallowed, not a crash (#631)."""
        mock_provider = MagicMock()
        mock_provider.reconnect_kline.side_effect = TimeoutError("rest socket hung")
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._kline_buffer = None  # skip resync

        mock_engine._handle_kline_disconnect()  # must not raise
        mock_provider.reconnect_kline.assert_called_once()


class TestKlineRecoveringRestFallback:
    """#662: the kline WS uses a *recovering* REST fallback — momentary REST while
    the WS is down, auto-return to WS-primary the instant real events resume, and
    it never stops trying (so it can never get stuck on REST until restart)."""

    @staticmethod
    def _dead_kline_engine(mock_engine):
        """Engine whose kline socket is dead-but-'started': start_kline_socket
        returned True but no event ever arrives → ws_healthy stays False."""
        provider = MagicMock()
        provider.ws_healthy = False
        mock_engine._ws_kline_provider = provider
        mock_engine._ws_kline_active = True
        mock_engine._kline_reconnect_failures = 0
        return provider

    @pytest.mark.fast
    def test_should_probe_predicate_never_permanently_false(self, mock_engine):
        """The throttle predicate probes every cycle up to the limit, then every
        Nth cycle — and is never permanently False, so the WS always has a path
        back to primary (#662)."""
        from src.config.constants import (
            DEFAULT_WS_KLINE_DEGRADED_PROBE_EVERY as PROBE,
        )
        from src.config.constants import (
            DEFAULT_WS_KLINE_RECONNECT_CIRCUIT_LIMIT as LIMIT,
        )

        # Fast phase: probe on every cycle up to the limit.
        assert all(mock_engine._should_probe_kline_reconnect(n) for n in range(1, LIMIT + 1))
        # Throttled phase: only multiples of PROBE probe.
        assert mock_engine._should_probe_kline_reconnect(LIMIT + 1) is False
        assert mock_engine._should_probe_kline_reconnect(PROBE) is True
        # Never permanently false: a probe always recurs within every PROBE-cycle window.
        for start in range(LIMIT + 1, 5 * PROBE, PROBE):
            assert any(
                mock_engine._should_probe_kline_reconnect(n) for n in range(start, start + PROBE)
            )

    @pytest.mark.fast
    def test_unhealthy_drops_to_momentary_rest_and_reconnects(self, mock_engine):
        """First unhealthy cycle: flip to REST (loop keeps trading), degrade once,
        and attempt a reconnect."""
        provider = self._dead_kline_engine(mock_engine)
        with patch.object(mock_engine, "_handle_kline_disconnect") as disc:
            mock_engine._check_kline_health()
        assert mock_engine._ws_kline_active is False  # on REST now
        provider.mark_kline_degraded.assert_called_once()
        disc.assert_called_once()
        assert mock_engine._kline_reconnect_failures == 1

    @pytest.mark.fast
    def test_returns_to_ws_primary_when_events_resume(self, mock_engine):
        """The instant a real kline event resumes (ws_healthy True), return to
        WS-primary and reset the breaker — the recovery the user-stream lacks."""
        provider = MagicMock()
        provider.ws_healthy = True  # real event arrived
        mock_engine._ws_kline_provider = provider
        mock_engine._ws_kline_active = False  # had dropped to REST
        mock_engine._kline_reconnect_failures = 5

        with patch.object(mock_engine, "_handle_kline_disconnect") as disc:
            mock_engine._check_kline_health()
        assert mock_engine._ws_kline_active is True  # WS primary again
        assert mock_engine._kline_reconnect_failures == 0  # breaker reset
        disc.assert_not_called()
        provider.mark_kline_degraded.assert_not_called()

    @pytest.mark.fast
    def test_degrades_exactly_once_across_many_dead_cycles(self, mock_engine):
        """mark_kline_degraded fires once per outage (no per-cycle log spam) and the
        engine stays on REST throughout."""
        provider = self._dead_kline_engine(mock_engine)
        with patch.object(mock_engine, "_handle_kline_disconnect"):
            for _ in range(25):
                mock_engine._check_kline_health()
        provider.mark_kline_degraded.assert_called_once()
        assert mock_engine._ws_kline_active is False

    @pytest.mark.fast
    def test_reconnect_attempts_throttle_but_never_stop(self, mock_engine):
        """Regression vs BOTH the 30s churn and REST-until-restart: a persistently
        dead socket gets fast reconnects up to the limit, then throttled probes
        every Nth cycle — bounded, but NEVER zero (always a path back to WS)."""
        from src.config.constants import (
            DEFAULT_WS_KLINE_DEGRADED_PROBE_EVERY as PROBE,
        )
        from src.config.constants import (
            DEFAULT_WS_KLINE_RECONNECT_CIRCUIT_LIMIT as LIMIT,
        )

        provider = self._dead_kline_engine(mock_engine)
        attempt_cycles: list[int] = []

        def _record():
            # _handle_kline_disconnect is called after the failure counter is bumped,
            # so the counter == the cycle index at each attempt.
            attempt_cycles.append(mock_engine._kline_reconnect_failures)

        with patch.object(mock_engine, "_handle_kline_disconnect", side_effect=_record):
            for _ in range(5 * PROBE):  # 50 cycles
                mock_engine._check_kline_health()

        expected = [n for n in range(1, 5 * PROBE + 1) if n <= LIMIT or n % PROBE == 0]
        assert attempt_cycles == expected
        # Never stops: probes continue into the final stretch of cycles.
        assert any(c > 5 * PROBE - PROBE for c in attempt_cycles)
        provider.mark_kline_degraded.assert_called_once()

    @pytest.mark.fast
    def test_recovers_after_long_degradation(self, mock_engine):
        """Even after a long dead-socket stretch, a real event returns to WS-primary
        — proving REST is never permanent (#662)."""
        provider = self._dead_kline_engine(mock_engine)
        with patch.object(mock_engine, "_handle_kline_disconnect"):
            for _ in range(30):
                mock_engine._check_kline_health()
        assert mock_engine._ws_kline_active is False
        assert mock_engine._kline_reconnect_failures >= 30

        provider.ws_healthy = True  # events resume
        with patch.object(mock_engine, "_handle_kline_disconnect") as disc:
            mock_engine._check_kline_health()
        assert mock_engine._ws_kline_active is True
        assert mock_engine._kline_reconnect_failures == 0
        disc.assert_not_called()

    @pytest.mark.fast
    def test_no_spurious_recovery_log_when_healthy_at_startup(self, mock_engine, caplog):
        """A normally-healthy stream logs no 'recovered' message and stays primary."""
        import logging

        provider = MagicMock()
        provider.ws_healthy = True
        mock_engine._ws_kline_provider = provider
        mock_engine._ws_kline_active = True  # already primary
        mock_engine._kline_reconnect_failures = 0

        with caplog.at_level(logging.INFO):
            with patch.object(mock_engine, "_handle_kline_disconnect") as disc:
                mock_engine._check_kline_health()
        assert mock_engine._ws_kline_active is True
        assert "recovered" not in caplog.text.lower()
        disc.assert_not_called()


class TestHandleUserStreamDisconnect:
    """Tests for _handle_user_stream_disconnect()."""

    @pytest.mark.fast
    def test_resyncs_orders_and_reconnects(self, mock_engine):
        """Should resync orders/positions via REST, then reconnect user stream."""
        mock_engine.enable_live_trading = True

        mock_exchange = MagicMock()
        mock_exchange.reconnect_user.return_value = True
        mock_engine.exchange_interface = mock_exchange

        mock_tracker = MagicMock()
        mock_engine.order_tracker = mock_tracker

        mock_reconciler = MagicMock()
        mock_engine._periodic_reconciler = mock_reconciler

        mock_engine._handle_user_stream_disconnect()

        # poll_once called twice: pre-reconnect catch-up + post-reconnect catch-up
        assert mock_tracker.poll_once.call_count == 2
        assert mock_reconciler.reconcile_once.call_count == 2
        mock_exchange.reconnect_user.assert_called_once()
        mock_tracker.disable_polling.assert_called_once()

    @pytest.mark.fast
    def test_enables_polling_on_reconnect_failure(self, mock_engine):
        """Should enable REST polling when user stream reconnect fails."""
        mock_engine.enable_live_trading = True

        mock_exchange = MagicMock()
        mock_exchange.reconnect_user.return_value = False
        mock_engine.exchange_interface = mock_exchange

        mock_tracker = MagicMock()
        mock_engine.order_tracker = mock_tracker

        mock_reconciler = MagicMock()
        mock_engine._periodic_reconciler = mock_reconciler

        mock_engine._handle_user_stream_disconnect()

        mock_exchange.mark_user_degraded.assert_called_once()
        mock_tracker.enable_polling.assert_called_once()
