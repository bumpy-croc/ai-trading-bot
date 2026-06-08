"""Tests for BinanceProvider WebSocket stream management."""

import asyncio
import gc
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data_providers.binance_provider import BinanceProvider, WebSocketState


@pytest.fixture
def provider():
    """Create a BinanceProvider with mocked client for testing."""
    with patch.object(BinanceProvider, "_initialize_client"):
        p = BinanceProvider(api_key="test_key", api_secret="test_secret", testnet=True)
        p._client = MagicMock()
        p._use_margin = False
        return p


class TestWebSocketState:
    """Tests for WebSocketState enum."""

    @pytest.mark.fast
    def test_enum_values(self):
        """Verify all expected states exist."""
        assert WebSocketState.DISCONNECTED.value == "disconnected"
        assert WebSocketState.PRIMARY.value == "primary"
        assert WebSocketState.RESYNCING.value == "resyncing"
        assert WebSocketState.REST_DEGRADED.value == "degraded"
        assert WebSocketState.SUSPENDED.value == "suspended"


class TestEnsureTwm:
    """Tests for lazy TWM creation."""

    @pytest.mark.fast
    def test_creates_twm_on_first_call(self, provider):
        """TWM is created and started on first call."""
        with patch("src.data_providers.binance_provider.ThreadedWebsocketManager") as mock_twm_cls:
            mock_twm = MagicMock()
            mock_twm_cls.return_value = mock_twm
            provider._ensure_twm()
            mock_twm_cls.assert_called_once()
            mock_twm.start.assert_called_once()
            assert provider._twm is mock_twm

    @pytest.mark.fast
    def test_noop_if_twm_exists(self, provider):
        """TWM is not recreated if already exists."""
        existing_twm = MagicMock()
        provider._twm = existing_twm
        provider._ensure_twm()
        assert provider._twm is existing_twm

    @pytest.mark.fast
    def test_passes_testnet_flag(self, provider):
        """Testnet flag is forwarded to TWM."""
        provider.testnet = True
        with patch("src.data_providers.binance_provider.ThreadedWebsocketManager") as mock_twm_cls:
            mock_twm_cls.return_value = MagicMock()
            provider._ensure_twm()
            call_kwargs = mock_twm_cls.call_args[1]
            assert call_kwargs.get("testnet") is True

    @pytest.mark.fast
    def test_passes_tld_for_us(self, provider):
        """TLD 'us' is set for Binance US endpoint."""
        with (
            patch("src.data_providers.binance_provider.ThreadedWebsocketManager") as mock_twm_cls,
            patch(
                "src.data_providers.binance_provider.get_binance_api_endpoint",
                return_value="binanceus",
            ),
        ):
            mock_twm_cls.return_value = MagicMock()
            provider._ensure_twm()
            call_kwargs = mock_twm_cls.call_args[1]
            assert call_kwargs.get("tld") == "us"


class TestStartKlineStream:
    """Tests for kline stream start."""

    @pytest.mark.fast
    def test_starts_kline_stream_successfully(self, provider):
        """Kline stream starts and sets PRIMARY state."""
        mock_twm = MagicMock()
        mock_twm.start_kline_socket.return_value = "kline_key_123"
        provider._twm = mock_twm

        callback = MagicMock()
        result = provider.start_kline_stream("BTCUSDT", "1h", callback)

        assert result is True
        assert provider._kline_ws_state == WebSocketState.PRIMARY
        assert provider._kline_socket_key == "kline_key_123"
        mock_twm.start_kline_socket.assert_called_once()

    @pytest.mark.fast
    def test_stores_symbol_and_timeframe(self, provider):
        """Active symbol and timeframe stored for reconnect."""
        mock_twm = MagicMock()
        mock_twm.start_kline_socket.return_value = "key"
        provider._twm = mock_twm

        provider.start_kline_stream("ETHUSDT", "4h", MagicMock())
        assert provider._active_symbol == "ETHUSDT"
        assert provider._active_timeframe == "4h"

    @pytest.mark.fast
    def test_returns_false_on_exception(self, provider):
        """Returns False if stream start fails."""
        mock_twm = MagicMock()
        mock_twm.start_kline_socket.side_effect = Exception("Connection failed")
        provider._twm = mock_twm

        result = provider.start_kline_stream("BTCUSDT", "1h", MagicMock())
        assert result is False

    @pytest.mark.fast
    def test_kline_callback_updates_timestamp(self, provider):
        """Kline callback updates last event time."""
        mock_twm = MagicMock()
        provider._twm = mock_twm

        captured_cb = None

        def capture_callback(**kwargs):
            nonlocal captured_cb
            captured_cb = kwargs["callback"]
            return "key"

        mock_twm.start_kline_socket.side_effect = capture_callback
        user_cb = MagicMock()
        provider.start_kline_stream("BTCUSDT", "1h", user_cb)

        # Simulate a kline event
        before = provider._last_kline_event_time
        captured_cb({"e": "kline", "k": {"t": 1234567890000}})
        assert provider._last_kline_event_time >= before
        user_cb.assert_called_once()

    @pytest.mark.fast
    def test_kline_error_event_triggers_disconnect(self, provider):
        """Error events trigger disconnect handler."""
        mock_twm = MagicMock()
        provider._twm = mock_twm

        captured_cb = None

        def capture_callback(**kwargs):
            nonlocal captured_cb
            captured_cb = kwargs["callback"]
            return "key"

        mock_twm.start_kline_socket.side_effect = capture_callback
        provider.start_kline_stream("BTCUSDT", "1h", MagicMock())

        # Simulate error event
        captured_cb({"e": "error", "m": "test error"})
        assert provider._kline_ws_state == WebSocketState.RESYNCING


class TestStartUserStream:
    """Tests for user data stream start."""

    @pytest.mark.fast
    def test_starts_user_stream_spot(self, provider):
        """User stream starts in spot mode."""
        mock_twm = MagicMock()
        mock_twm.start_user_socket.return_value = "user_key"
        provider._twm = mock_twm
        provider._use_margin = False

        callback = MagicMock()
        result = provider.start_user_stream(callback)

        assert result is True
        mock_twm.start_user_socket.assert_called_once()
        mock_twm.start_margin_socket.assert_not_called()

    @pytest.mark.fast
    def test_starts_margin_stream(self, provider):
        """User stream starts margin socket when margin mode active."""
        mock_twm = MagicMock()
        mock_twm.start_margin_socket.return_value = "margin_key"
        provider._twm = mock_twm
        provider._use_margin = True

        callback = MagicMock()
        result = provider.start_user_stream(callback)

        assert result is True
        mock_twm.start_margin_socket.assert_called_once()
        mock_twm.start_user_socket.assert_not_called()

    @pytest.mark.fast
    def test_returns_false_on_exception(self, provider):
        """Returns False if user stream start fails."""
        mock_twm = MagicMock()
        mock_twm.start_user_socket.side_effect = Exception("Auth failed")
        provider._twm = mock_twm

        result = provider.start_user_stream(MagicMock())
        assert result is False

    @pytest.mark.fast
    def test_resets_event_received_flag_on_start(self, provider):
        """start_user_stream resets _user_event_received to False so a stale True
        from a prior generation can't read as healthy before the first real event
        of the new socket (Defense A; mirrors kline) (#717)."""
        mock_twm = MagicMock()
        mock_twm.start_user_socket.return_value = "user_key"
        provider._twm = mock_twm
        provider._use_margin = False
        provider._user_event_received = True  # leftover from a previous socket

        provider.start_user_stream(MagicMock())
        assert provider._user_event_received is False


class TestUserStreamGeneration:
    """#717: a monotonic generation token invalidates callbacks from a torn-down
    user socket so a stale python-binance read_ready callback can't fake a recovery
    (set _user_event_received / refresh freshness) the watchdog would act on."""

    @staticmethod
    def _capture_user_callback(provider):
        """Start a margin user stream and return (callback, captured_events)."""
        mock_twm = MagicMock()
        captured: dict[str, object] = {}

        def capture(**kwargs):
            captured["cb"] = kwargs["callback"]
            return "margin_key"

        mock_twm.start_margin_socket.side_effect = capture
        provider._twm = mock_twm
        provider._use_margin = True
        sink_events: list[dict] = []
        provider.start_user_stream(sink_events.append)
        return captured["cb"], sink_events

    @pytest.mark.fast
    def test_start_bumps_generation_and_captures_it(self, provider):
        """Each (re)start advances the generation token."""
        mock_twm = MagicMock()
        mock_twm.start_user_socket.return_value = "k"
        provider._twm = mock_twm
        provider._use_margin = False

        before = provider._user_stream_generation
        provider.start_user_stream(MagicMock())
        assert provider._user_stream_generation == before + 1

    @pytest.mark.fast
    def test_current_generation_callback_is_processed(self, provider):
        """A callback whose generation is current updates freshness, sets the event
        flag, and forwards the event."""
        cb, sink = self._capture_user_callback(provider)
        provider._user_event_received = False

        cb({"e": "executionReport", "i": 42})

        assert provider._user_event_received is True
        assert sink == [{"e": "executionReport", "i": 42}]

    @pytest.mark.fast
    def test_stale_generation_callback_is_ignored(self, provider):
        """A callback from a superseded generation (socket torn down + a newer start)
        is dropped: no event forwarded, no freshness refresh, no event-flag flip —
        so it cannot fake a recovery the watchdog acts on (#717)."""
        cb, sink = self._capture_user_callback(provider)
        # A later teardown/restart advances the generation past the captured one.
        provider._user_stream_generation += 1
        provider._user_event_received = False
        provider._last_user_event_time = datetime.now(UTC) - timedelta(seconds=999)
        stale_before = provider._last_user_event_time

        cb({"e": "executionReport", "i": 7})

        assert sink == []  # event dropped
        assert provider._user_event_received is False  # no false recovery
        assert provider._last_user_event_time == stale_before  # freshness untouched

    @pytest.mark.fast
    def test_stop_user_stream_invalidates_generation(self, provider):
        """stop_user_stream bumps the generation FIRST so an in-flight callback
        draining during teardown is dropped."""
        cb, sink = self._capture_user_callback(provider)
        provider._user_socket_key = "margin_key"

        provider.stop_user_stream()
        provider._user_event_received = False
        cb({"e": "executionReport", "i": 1})  # late callback after stop

        assert sink == []
        assert provider._user_event_received is False

    @pytest.mark.fast
    def test_stop_streams_invalidates_generation(self, provider):
        """stop_streams bumps the user generation before teardown so a callback
        still draining on the loop during shutdown is dropped."""
        cb, sink = self._capture_user_callback(provider)
        before = provider._user_stream_generation

        provider.stop_streams()
        assert provider._user_stream_generation == before + 1

        provider._user_event_received = False
        cb({"e": "executionReport", "i": 2})
        assert sink == []
        assert provider._user_event_received is False

    @pytest.mark.fast
    def test_callback_generation_check_and_write_are_atomic(self, provider):
        """Regression for the check-then-act TOCTOU: the callback's generation
        check and its _user_event_received write share ONE locked critical section
        with the generation bump in stop/start, so a stop that lands after the check
        passed cannot leave a stale True under the new generation (#717).

        We interpose on _user_stream_lock so that, the instant the callback acquires
        it, a teardown (generation bump + flag reset) has already completed. The
        callback must then observe the bumped generation and drop the event."""
        cb, sink = self._capture_user_callback(provider)
        captured_gen = provider._user_stream_generation
        real_lock = provider._user_stream_lock

        class _StopInjectingLock:
            """Delegates to the real lock, but on the callback's acquire it first
            simulates a concurrent stop having fully completed under the lock."""

            def __init__(self, provider):
                self._provider = provider
                self._fired = False

            def __enter__(self):
                real_lock.acquire()
                if not self._fired:
                    self._fired = True
                    # Concurrent teardown+restart that completed before the callback
                    # got the lock: generation advanced, flag cleared.
                    self._provider._user_stream_generation += 2
                    self._provider._user_event_received = False
                return self

            def __exit__(self, *exc):
                real_lock.release()
                return False

        provider._user_stream_lock = _StopInjectingLock(provider)
        provider._user_event_received = False

        # The callback (captured at captured_gen) now runs; on acquiring the lock the
        # generation has already advanced past captured_gen → event dropped.
        cb({"e": "executionReport", "i": 99})

        assert provider._user_stream_generation == captured_gen + 2
        assert provider._user_event_received is False  # no stale True under new gen
        assert sink == []  # event not forwarded

    @pytest.mark.fast
    def test_concurrent_stop_and_callback_never_leave_stale_true(self, provider):
        """Hammer the callback against stop_user_stream on two real threads. The
        post-condition invariant: _user_event_received is True ONLY if the recorded
        generation is the current one — a stale callback can never leave it True
        after a stop bumped the generation (#717)."""
        import threading

        cb, _sink = self._capture_user_callback(provider)
        provider._user_socket_key = "margin_key"
        violations: list[str] = []

        def fire_callbacks():
            for _ in range(200):
                cb({"e": "executionReport", "i": 1})

        def stop_and_check():
            for _ in range(200):
                provider.stop_user_stream()
                # After a stop (gen bumped, flag cleared under the lock), the flag may
                # only be True if a CURRENT-generation callback set it post-stop. We
                # can't observe gen-at-write directly, but the lock guarantees the
                # callback's check+write is atomic with the bump; so immediately after
                # stop returns, if no new start ran, the flag must be False.
                with provider._user_stream_lock:
                    if provider._user_event_received:
                        violations.append("stale True after stop")
                    provider._user_event_received = False  # reset for next round

        t1 = threading.Thread(target=fire_callbacks)
        t2 = threading.Thread(target=stop_and_check)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert violations == []


class TestStopStreams:
    """Tests for stopping WebSocket streams."""

    @pytest.mark.fast
    def test_stops_twm_and_resets_state(self, provider):
        """stop_streams() stops TWM and resets all socket keys."""
        mock_twm = MagicMock()
        provider._twm = mock_twm
        provider._kline_socket_key = "key1"
        provider._user_socket_key = "key2"
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._user_ws_state = WebSocketState.PRIMARY

        provider.stop_streams()

        mock_twm.stop.assert_called_once()
        assert provider._twm is None
        assert provider._kline_socket_key is None
        assert provider._user_socket_key is None
        assert provider._kline_ws_state == WebSocketState.DISCONNECTED
        assert provider._user_ws_state == WebSocketState.DISCONNECTED

    @pytest.mark.fast
    def test_noop_when_no_twm(self, provider):
        """stop_streams() is safe when no TWM exists."""
        provider._twm = None
        provider.stop_streams()  # Should not raise


class TestWsProperties:
    """Tests for ws_state and ws_healthy properties."""

    @pytest.mark.fast
    def test_ws_state_returns_worst_state(self, provider):
        """ws_state property returns the worse of the two stream states."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._user_ws_state = WebSocketState.PRIMARY
        assert provider.ws_state == WebSocketState.PRIMARY

        # User stream error should not affect kline-only ws_state check
        provider._user_ws_state = WebSocketState.RESYNCING
        assert provider.ws_state == WebSocketState.RESYNCING

        provider._kline_ws_state = WebSocketState.REST_DEGRADED
        provider._user_ws_state = WebSocketState.PRIMARY
        assert provider.ws_state == WebSocketState.REST_DEGRADED

    @pytest.mark.fast
    def test_ws_healthy_true_when_primary_and_fresh(self, provider):
        """ws_healthy returns True when kline PRIMARY and recent kline event."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._last_kline_event_time = datetime.now(UTC)
        assert provider.ws_healthy is True

    @pytest.mark.fast
    def test_ws_healthy_false_when_stale(self, provider):
        """ws_healthy returns False when kline event is stale."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._last_kline_event_time = datetime.now(UTC) - timedelta(seconds=130)
        assert provider.ws_healthy is False

    @pytest.mark.fast
    def test_ws_healthy_false_when_kline_not_primary(self, provider):
        """ws_healthy returns False when kline not in PRIMARY state."""
        provider._kline_ws_state = WebSocketState.REST_DEGRADED
        provider._last_kline_event_time = datetime.now(UTC)
        assert provider.ws_healthy is False

    @pytest.mark.fast
    def test_ws_healthy_true_even_when_user_stream_resyncing(self, provider):
        """ws_healthy is True when kline is PRIMARY and no user stream configured.

        This is the paper-trading / data-only path where ``start_user_stream``
        has never been called. ``_on_user_event_cb`` remains None so the
        user stream's state must be ignored.
        """
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._user_ws_state = WebSocketState.RESYNCING
        provider._on_user_event_cb = None  # explicit: no user stream configured
        provider._last_kline_event_time = datetime.now(UTC)
        assert provider.ws_healthy is True

    # ------------------------------------------------------------------ #
    # GH #608: user-stream watchdog                                       #
    # ------------------------------------------------------------------ #

    @pytest.mark.fast
    def test_ws_healthy_false_when_user_stream_configured_but_resyncing(self, provider):
        """When user stream IS configured, RESYNCING flips ws_healthy to False.

        Live margin trading subscribes a user stream — its state must be
        reflected in the global health flag so the engine can fall back to
        REST account-sync rather than trusting potentially stale WS state.
        """
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._last_kline_event_time = datetime.now(UTC)
        provider._on_user_event_cb = MagicMock()  # user stream configured
        provider._user_ws_state = WebSocketState.RESYNCING
        provider._user_event_received = True
        provider._last_user_event_time = datetime.now(UTC)
        assert provider.ws_healthy is False

    @pytest.mark.fast
    def test_ws_healthy_false_when_user_stream_stale(self, provider):
        """User stream PRIMARY but no events for >120s flips ws_healthy off."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._last_kline_event_time = datetime.now(UTC)
        provider._on_user_event_cb = MagicMock()
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._user_event_received = True
        provider._last_user_event_time = datetime.now(UTC) - timedelta(seconds=130)
        assert provider.ws_healthy is False

    @pytest.mark.fast
    def test_ws_healthy_true_when_both_streams_primary_and_fresh(self, provider):
        """Both streams configured, both PRIMARY, both fresh → healthy."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._last_kline_event_time = datetime.now(UTC)
        provider._on_user_event_cb = MagicMock()
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._user_event_received = True
        provider._last_user_event_time = datetime.now(UTC)
        assert provider.ws_healthy is True

    @pytest.mark.fast
    def test_ws_healthy_false_when_user_event_never_received(self, provider):
        """User stream PRIMARY but first event not yet seen → not healthy."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._last_kline_event_time = datetime.now(UTC)
        provider._on_user_event_cb = MagicMock()
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._user_event_received = False  # not yet
        provider._last_user_event_time = datetime.now(UTC)
        assert provider.ws_healthy is False

    @pytest.mark.fast
    def test_user_ws_healthy_false_when_no_callback(self, provider):
        """user_ws_healthy reports False when no user stream is configured.

        Caller must combine with ``_on_user_event_cb is not None`` to tell
        'unhealthy' from 'not configured'. Documented on the property.
        """
        provider._on_user_event_cb = None
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._user_event_received = True
        provider._last_user_event_time = datetime.now(UTC)
        assert provider.user_ws_healthy is False

    @pytest.mark.fast
    def test_user_ws_healthy_true_when_primary_fresh_and_received(self, provider):
        provider._on_user_event_cb = MagicMock()
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._user_event_received = True
        provider._last_user_event_time = datetime.now(UTC)
        assert provider.user_ws_healthy is True


class TestReconnect:
    """Tests for reconnect methods."""

    @pytest.mark.fast
    def test_reconnect_kline_stops_and_restarts(self, provider):
        """reconnect_kline() stops only kline socket and restarts kline stream."""
        provider._active_symbol = "BTCUSDT"
        provider._active_timeframe = "1h"
        provider._on_kline_cb = MagicMock()
        provider._kline_socket_key = "old_kline_key"
        mock_twm = MagicMock()
        provider._twm = mock_twm

        with patch.object(provider, "start_kline_stream", return_value=True) as mock_start:
            result = provider.reconnect_kline()
            assert result is True
            mock_twm.stop_socket.assert_called_once_with("old_kline_key")
            assert provider._kline_socket_key is None
            mock_start.assert_called_once_with("BTCUSDT", "1h", provider._on_kline_cb)

    @pytest.mark.fast
    def test_reconnect_kline_returns_false_on_failure(self, provider):
        """reconnect_kline() returns False on exception."""
        provider._active_symbol = "BTCUSDT"
        provider._active_timeframe = "1h"
        provider._on_kline_cb = MagicMock()
        provider._kline_socket_key = "old_kline_key"
        mock_twm = MagicMock()
        mock_twm.stop_socket.side_effect = Exception("fail")
        provider._twm = mock_twm

        result = provider.reconnect_kline()
        assert result is False

    @pytest.mark.fast
    def test_reconnect_user_restarts_user_stream(self, provider):
        """reconnect_user() restarts user data stream."""
        provider._on_user_event_cb = MagicMock()
        provider._user_socket_key = "old_key"
        mock_twm = MagicMock()
        provider._twm = mock_twm

        with patch.object(provider, "start_user_stream", return_value=True) as mock_start:
            result = provider.reconnect_user()
            assert result is True
            mock_twm.stop_socket.assert_called_once_with("old_key")
            mock_start.assert_called_once()

    @pytest.mark.fast
    def test_reconnect_user_returns_false_without_callback(self, provider):
        """reconnect_user() returns False when no callback stored."""
        provider._on_user_event_cb = None
        result = provider.reconnect_user()
        assert result is False


class TestOnStreamDisconnect:
    """Tests for per-stream disconnect handlers."""

    @pytest.mark.fast
    def test_kline_disconnect_sets_kline_resyncing(self, provider):
        """_on_kline_disconnect() transitions kline state to RESYNCING."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._on_kline_disconnect()
        assert provider._kline_ws_state == WebSocketState.RESYNCING
        assert provider._user_ws_state == WebSocketState.PRIMARY

    @pytest.mark.fast
    def test_user_disconnect_sets_user_resyncing(self, provider):
        """_on_user_disconnect() transitions user state to RESYNCING."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._user_ws_state = WebSocketState.PRIMARY
        provider._on_user_disconnect()
        assert provider._user_ws_state == WebSocketState.RESYNCING
        assert provider._kline_ws_state == WebSocketState.PRIMARY


class _FakeSocket:
    """Minimal async-context socket whose ``recv`` returns a configurable value.

    A falsy return makes python-binance's ``start_listener`` hit ``continue``
    and re-evaluate ``while self._socket_running[path]`` (the teardown raise
    site); a truthy return makes it invoke ``callback(msg)``.
    """

    def __init__(self, recv_value=None):
        self._recv_value = recv_value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def recv(self):
        return self._recv_value


def _capture_real_teardown_keyerror() -> KeyError:
    """Return the *genuine* python-binance teardown ``KeyError``.

    Drives the real ``ThreadedApiManager.start_listener`` through the
    ``stop_socket`` race so the captured exception has the true
    ``threaded_stream.py`` ``start_listener`` raise-site frame — exercising the
    provider's module-path anchor rather than a faked frame.
    """
    from binance.ws.threaded_stream import ThreadedApiManager

    captured: dict[str, KeyError] = {}

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, ctx: captured.__setitem__("exc", ctx["exception"]))
        mgr = ThreadedApiManager.__new__(ThreadedApiManager)
        mgr._socket_running = {"margin_subscription:0": True}
        mgr._log = logging.getLogger("test.binance")
        task = asyncio.create_task(
            mgr.start_listener(
                _FakeSocket(recv_value=None), "margin_subscription:0", lambda m: None
            )
        )
        await asyncio.sleep(0)
        del mgr._socket_running["margin_subscription:0"]
        await asyncio.sleep(0.05)
        del task
        gc.collect()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    exc = captured.get("exc")
    assert isinstance(exc, KeyError), "expected a real teardown KeyError"
    return exc


def _capture_real_callback_keyerror() -> KeyError:
    """Return a real ``KeyError`` raised by the user callback inside ``start_listener``.

    python-binance invokes ``callback(msg)`` *inside* ``start_listener``, so a
    callback failure's traceback passes *through* ``start_listener`` but its
    deepest frame is the callback — the exact false-positive the discriminator
    must NOT suppress.
    """
    from binance.ws.threaded_stream import ThreadedApiManager

    captured: dict[str, KeyError] = {}

    def bad_callback(_msg) -> None:
        raise KeyError("orderId")  # a genuine downstream lookup failure

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, ctx: captured.__setitem__("exc", ctx["exception"]))
        mgr = ThreadedApiManager.__new__(ThreadedApiManager)
        mgr._socket_running = {"margin_subscription:0": True}
        mgr._log = logging.getLogger("test.binance")
        task = asyncio.create_task(
            mgr.start_listener(
                _FakeSocket(recv_value={"e": "executionReport"}),
                "margin_subscription:0",
                bad_callback,
            )
        )
        await asyncio.sleep(0.05)
        del task
        gc.collect()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    exc = captured.get("exc")
    assert isinstance(exc, KeyError), "expected a real callback KeyError"
    return exc


class TestTwmLoopExceptionHandler:
    """Tests for the TWM loop exception handler that suppresses the benign
    python-binance socket-teardown ``KeyError`` on circuit-open (#716)."""

    @pytest.mark.fast
    def test_detects_real_teardown_keyerror(self, provider):
        """The genuine library teardown KeyError is classified as teardown noise."""
        exc = _capture_real_teardown_keyerror()
        assert provider._is_socket_teardown_keyerror(exc) is True

    @pytest.mark.fast
    def test_ignores_callback_keyerror_through_start_listener(self, provider):
        """A callback KeyError (deepest frame is the callback) is NOT suppressed.

        This is the live-trading safety case: python-binance runs our callback
        inside ``start_listener``, so a real downstream KeyError's traceback
        passes through ``start_listener`` — but it must still reach the default
        handler (ERROR), not be hidden at DEBUG.
        """
        exc = _capture_real_callback_keyerror()
        assert provider._is_socket_teardown_keyerror(exc) is False

    @pytest.mark.fast
    def test_ignores_non_keyerror(self, provider):
        """A non-KeyError loop exception is never classified as teardown noise."""
        assert provider._is_socket_teardown_keyerror(ValueError("x")) is False
        assert provider._is_socket_teardown_keyerror(None) is False

    @pytest.mark.fast
    def test_ignores_keyerror_without_traceback(self, provider):
        """A bare KeyError (no traceback) cannot be the teardown signature."""
        assert provider._is_socket_teardown_keyerror(KeyError("margin_subscription:0")) is False

    @pytest.mark.fast
    def test_handler_suppresses_teardown_keyerror(self, provider, caplog):
        """The teardown KeyError is logged at DEBUG and not delegated to default."""
        mock_loop = MagicMock()
        exc = _capture_real_teardown_keyerror()
        context = {"message": "Task exception was never retrieved", "exception": exc}

        with caplog.at_level(logging.DEBUG, logger="src.data_providers.binance_provider"):
            provider._twm_loop_exception_handler(mock_loop, context)

        mock_loop.default_exception_handler.assert_not_called()
        assert any(
            "socket-teardown KeyError" in rec.message and rec.levelno == logging.DEBUG
            for rec in caplog.records
        )

    @pytest.mark.fast
    def test_handler_delegates_unrelated_exception(self, provider):
        """A non-teardown loop exception is delegated to the default handler."""
        mock_loop = MagicMock()
        context = {"message": "boom", "exception": ValueError("unrelated")}

        provider._twm_loop_exception_handler(mock_loop, context)

        mock_loop.default_exception_handler.assert_called_once_with(context)

    @pytest.mark.fast
    def test_handler_delegates_callback_keyerror(self, provider):
        """A callback KeyError still reaches the default handler (stays visible)."""
        mock_loop = MagicMock()
        exc = _capture_real_callback_keyerror()
        context = {"message": "Task exception was never retrieved", "exception": exc}

        provider._twm_loop_exception_handler(mock_loop, context)

        mock_loop.default_exception_handler.assert_called_once_with(context)

    @pytest.mark.fast
    def test_ensure_twm_installs_handler_on_loop(self, provider):
        """_ensure_twm wires the suppression handler onto the per-manager loop."""
        with (
            patch("src.data_providers.binance_provider.ThreadedWebsocketManager") as mock_twm_cls,
            patch(
                "src.data_providers.binance_provider.get_binance_api_endpoint", return_value="com"
            ),
        ):
            mock_twm_cls.return_value = MagicMock()
            provider._ensure_twm()

        assert provider._twm_loop is not None
        # Bound methods are recreated per access, so compare by underlying
        # function + instance rather than identity.
        installed = provider._twm_loop.get_exception_handler()
        assert installed.__func__ is BinanceProvider._twm_loop_exception_handler
        assert installed.__self__ is provider
        provider._twm_loop.close()


class TestCircuitOpenTeardownNoUncaughtKeyError:
    """End-to-end reproduction of the #716 circuit-open teardown race against the
    REAL python-binance ``start_listener`` coroutine — asserts the resulting
    KeyError surfaces no uncaught exception once the provider handler is wired."""

    @pytest.mark.fast
    def test_real_library_teardown_keyerror_is_suppressed(self, provider):
        """Drive python-binance start_listener through the stop_socket race.

        Reproduces the exact prod signature: ``stop_user_stream`` flips
        ``_socket_running[path]`` to False, the library deletes the key on exit,
        and the listener loop re-reads it → ``KeyError('margin_subscription:0')``
        on the TWM loop. With the provider's exception handler installed, the
        error is intercepted (DEBUG) and nothing reaches the loop's default
        handler, so no ``Task exception was never retrieved`` ERROR is emitted.

        The default-handler spy is bound to the *concrete* running loop
        instance (not the abstract base), since that is what asyncio actually
        invokes — patching ``AbstractEventLoop`` would not observe the call.
        """
        pytest.importorskip("binance.ws.threaded_stream")
        from binance.ws.threaded_stream import ThreadedApiManager

        delegated: list[dict] = []

        async def scenario() -> None:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(provider._twm_loop_exception_handler)
            # Spy on the CONCRETE loop's default handler — this is the real call
            # target when the provider handler delegates.
            real_default = loop.default_exception_handler

            def spy_default(context):
                delegated.append(context)
                return real_default(context)

            with patch.object(loop, "default_exception_handler", spy_default):
                mgr = ThreadedApiManager.__new__(ThreadedApiManager)
                mgr._socket_running = {"margin_subscription:0": True}
                mgr._log = logging.getLogger("test.binance")

                task = asyncio.create_task(
                    mgr.start_listener(
                        _FakeSocket(recv_value=None), "margin_subscription:0", lambda m: None
                    )
                )
                await asyncio.sleep(0)  # enter the read loop once
                # Simulate stop_socket(key) + the library's `del` on the racing exit.
                del mgr._socket_running["margin_subscription:0"]
                await asyncio.sleep(0.05)
                del task  # drop ref → exception reported as "never retrieved"
                gc.collect()
                await asyncio.sleep(0.05)

        asyncio.run(scenario())

        # The benign teardown KeyError must NOT have reached the default handler
        # (which is what logs the ERROR + Traceback that trips monitoring).
        assert delegated == []

    @pytest.mark.fast
    def test_real_library_callback_keyerror_is_not_suppressed(self, provider):
        """A real callback KeyError DOES reach the loop default handler.

        Guards the live-trading safety property: if our user-data callback
        raises a genuine ``KeyError`` while python-binance runs it inside
        ``start_listener``, the provider handler must delegate it to the default
        handler (ERROR + Traceback) instead of hiding it at DEBUG.
        """
        pytest.importorskip("binance.ws.threaded_stream")
        from binance.ws.threaded_stream import ThreadedApiManager

        delegated: list[dict] = []

        def bad_callback(_msg) -> None:
            raise KeyError("orderId")

        async def scenario() -> None:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(provider._twm_loop_exception_handler)

            def spy_default(context):
                # Record the delegation and swallow so the genuine traceback
                # doesn't spam test output; we only assert delegation happened.
                delegated.append(context)

            with patch.object(loop, "default_exception_handler", spy_default):
                mgr = ThreadedApiManager.__new__(ThreadedApiManager)
                mgr._socket_running = {"margin_subscription:0": True}
                mgr._log = logging.getLogger("test.binance")

                task = asyncio.create_task(
                    mgr.start_listener(
                        _FakeSocket(recv_value={"e": "executionReport"}),
                        "margin_subscription:0",
                        bad_callback,
                    )
                )
                await asyncio.sleep(0.05)
                del task
                gc.collect()
                await asyncio.sleep(0.05)

        asyncio.run(scenario())

        assert len(delegated) == 1
        assert isinstance(delegated[0].get("exception"), KeyError)
