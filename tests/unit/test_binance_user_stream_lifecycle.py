"""Lifecycle and teardown behaviour of BinanceProvider's WebSocket streams (#723, #729).

Mock-level counterparts to ``test_binance_user_stream_recovery.py``: socket
bookkeeping, shutdown vs rebuild races, dead-listener detection and the quiet
teardown helpers.
"""

import asyncio
import gc
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("binance.ws.threaded_stream")

from binance.exceptions import BinanceWebsocketUnableToConnect  # noqa: E402
from binance.ws.constants import WSListenerState  # noqa: E402
from binance.ws.keepalive_websocket import KeepAliveWebsocket  # noqa: E402
from binance.ws.threaded_stream import ThreadedApiManager  # noqa: E402

from src.data_providers.binance_provider import (  # noqa: E402
    BinanceProvider,
    WebSocketState,
    _ExpectedTransportCloseFilter,
    _muted_expected_transport_close,
)


@pytest.fixture
def provider():
    """A margin BinanceProvider with a mocked REST client."""
    with patch.object(BinanceProvider, "_initialize_client"):
        p = BinanceProvider(api_key="test_key", api_secret="test_secret", testnet=True)
        p._client = MagicMock()
        p._use_margin = True
        return p


def _capture_real_subscribe_failure() -> BaseException:
    """The exception python-binance leaves unretrieved when the margin subscribe fails.

    Drives the real ``start_listener`` over a real ``KeepAliveWebsocket`` whose
    subscribe raises, so the traceback has the genuine library frames.
    """
    captured: dict[str, BaseException] = {}

    class FailingClient:
        tld = "com"
        testnet = False
        demo = False
        https_proxy = None
        ws_api = MagicMock()

        async def margin_create_listen_token(self, is_isolated: bool = False) -> dict:
            raise BinanceWebsocketUnableToConnect("Request timed out")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, ctx: captured.__setitem__("exc", ctx["exception"]))
        socket = KeepAliveWebsocket(
            client=FailingClient(), url="wss://unused/", keepalive_type="margin"
        )
        manager = ThreadedApiManager.__new__(ThreadedApiManager)
        manager._socket_running = {None: True}
        manager._log = logging.getLogger("test.binance")
        task = asyncio.create_task(manager.start_listener(socket, None, lambda m: None))
        await asyncio.sleep(0.05)
        del task
        gc.collect()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert "exc" in captured, "expected the unretrieved subscribe failure"
    return captured["exc"]


def _capture_real_callback_failure() -> BaseException:
    """A genuine exception raised by the user callback inside ``start_listener``."""
    captured: dict[str, BaseException] = {}

    class ReadyWebsocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def recv(self):
            return {"e": "executionReport"}

    def bad_callback(_msg) -> None:
        raise KeyError("orderId")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, ctx: captured.__setitem__("exc", ctx["exception"]))
        manager = ThreadedApiManager.__new__(ThreadedApiManager)
        manager._socket_running = {None: True}
        manager._log = logging.getLogger("test.binance")
        task = asyncio.create_task(manager.start_listener(ReadyWebsocket(), None, bad_callback))
        await asyncio.sleep(0.05)
        del task
        gc.collect()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    return captured["exc"]


class TestUserSubscribeFailureDetection:
    """A failed subscribe kills the listener silently; the provider must notice (#616)."""

    @pytest.mark.fast
    def test_real_subscribe_failure_is_classified(self, provider):
        """The genuine library traceback of a failed margin subscribe matches."""
        assert provider._is_user_subscribe_failure(_capture_real_subscribe_failure()) is True

    @pytest.mark.fast
    def test_callback_failure_is_not_classified(self, provider):
        """A failure inside our own callback is a real bug and must reach the default handler."""
        assert provider._is_user_subscribe_failure(_capture_real_callback_failure()) is False

    @pytest.mark.fast
    def test_none_and_plain_exceptions_are_not_classified(self, provider):
        assert provider._is_user_subscribe_failure(None) is False
        assert provider._is_user_subscribe_failure(RuntimeError("x")) is False

    @pytest.mark.fast
    def test_handler_flags_resyncing_for_the_live_loop(self, provider):
        """On the live loop a failed subscribe turns PRIMARY into RESYNCING, with no ERROR log."""
        loop = MagicMock()
        provider._twm_loop = loop
        provider._user_ws_state = WebSocketState.PRIMARY
        exc = _capture_real_subscribe_failure()
        with patch.object(loop, "default_exception_handler") as default:
            provider._twm_loop_exception_handler(loop, {"exception": exc})
        assert provider._user_ws_state == WebSocketState.RESYNCING
        default.assert_not_called()

    @pytest.mark.fast
    def test_handler_ignores_a_retired_loop(self, provider):
        """A failure from a loop that is no longer the live one must not touch the stream."""
        provider._twm_loop = MagicMock()
        provider._user_ws_state = WebSocketState.PRIMARY
        exc = _capture_real_subscribe_failure()
        provider._twm_loop_exception_handler(MagicMock(), {"exception": exc})
        assert provider._user_ws_state == WebSocketState.PRIMARY

    @pytest.mark.fast
    def test_handler_does_not_clobber_other_states(self, provider):
        """A degraded/disconnected stream keeps its state; only PRIMARY flips."""
        loop = MagicMock()
        provider._twm_loop = loop
        exc = _capture_real_subscribe_failure()
        for state in (WebSocketState.REST_DEGRADED, WebSocketState.DISCONNECTED):
            provider._user_ws_state = state
            provider._twm_loop_exception_handler(loop, {"exception": exc})
            assert provider._user_ws_state == state


class TestUserStreamStartStop:
    """Socket bookkeeping that python-binance's ``None`` ws-api key would defeat."""

    @pytest.mark.fast
    def test_stop_reaches_a_socket_whose_key_is_none(self, provider):
        """stop_socket(None) is how the first margin socket's listener is stopped."""
        twm = MagicMock()
        twm.start_margin_socket.return_value = None  # python-binance's ws-api margin key
        provider._twm = twm

        assert provider.start_user_stream(MagicMock()) is True
        provider.stop_user_stream()

        twm.stop_socket.assert_called_once_with(None)
        assert provider._user_ws_state == WebSocketState.DISCONNECTED
        assert provider._user_socket_active is False

    @pytest.mark.fast
    def test_stop_without_a_started_socket_is_a_noop(self, provider):
        twm = MagicMock()
        provider._twm = twm
        provider.stop_user_stream()
        twm.stop_socket.assert_not_called()

    @pytest.mark.fast
    def test_state_is_primary_before_the_socket_starts(self, provider):
        """A fast subscribe failure flags RESYNCING from the loop; PRIMARY must already be set."""
        seen: list[WebSocketState] = []
        twm = MagicMock()
        twm.start_margin_socket.side_effect = lambda callback: (
            seen.append(provider._user_ws_state),
            "key",
        )[1]
        provider._twm = twm
        provider.start_user_stream(MagicMock())
        assert seen == [WebSocketState.PRIMARY]

    @pytest.mark.fast
    def test_failed_start_does_not_leave_a_primary_state(self, provider):
        twm = MagicMock()
        twm.start_margin_socket.side_effect = RuntimeError("no BSM")
        provider._twm = twm
        assert provider.start_user_stream(MagicMock()) is False
        assert provider._user_ws_state == WebSocketState.DISCONNECTED
        assert provider._user_stream_subscribed_at is None

    @pytest.mark.fast
    def test_forwarding_failure_does_not_escape_the_callback(self, provider, caplog):
        """An exception from the downstream consumer would kill the listener task for good."""
        twm = MagicMock()
        twm.start_margin_socket.return_value = None
        provider._twm = twm
        consumer = MagicMock(side_effect=RuntimeError("consumer bug"))
        provider.start_user_stream(consumer)
        user_callback = twm.start_margin_socket.call_args.kwargs["callback"]

        with caplog.at_level(logging.ERROR):
            user_callback({"e": "executionReport"})  # must not raise

        consumer.assert_called_once()
        assert any("forwarding failed" in r.getMessage() for r in caplog.records)
        assert provider._user_event_received is True


class TestShutdownVersusRebuild:
    """The health thread must never build a TWM after the engine shut the streams down."""

    @pytest.mark.fast
    def test_stop_streams_blocks_later_rebuilds(self, provider):
        provider._on_user_event_cb = MagicMock()
        provider.stop_streams()
        with patch.object(provider, "_ensure_twm") as ensure:
            assert provider.hard_reconnect_user() is False
            assert provider.reconnect_user() is False
            assert provider.reconnect_kline() is False
        ensure.assert_not_called()
        assert provider._twm is None

    @pytest.mark.fast
    def test_explicit_start_rearms_the_provider(self, provider):
        provider.stop_streams()
        provider._twm = MagicMock()
        provider._twm.start_margin_socket.return_value = "key"
        assert provider.start_user_stream(MagicMock()) is True
        assert provider._streams_shut_down.is_set() is False

    @pytest.mark.fast
    def test_rebuild_cannot_rearm_a_provider_shut_down_meanwhile(self, provider):
        """A shutdown landing mid-rebuild stays in force: the rebuild's own start call
        must not clear it, and the rebuild abandons before building a new TWM."""
        provider._on_user_event_cb = MagicMock()

        def shutdown_during_teardown() -> None:
            provider._streams_shut_down.set()  # stop_streams() sets it before taking the lock

        with (
            patch.object(provider, "_teardown_twm", side_effect=shutdown_during_teardown),
            patch.object(provider, "_ensure_twm") as ensure,
        ):
            assert provider.hard_reconnect_user() is False

        ensure.assert_not_called()
        assert provider._streams_shut_down.is_set()

    @pytest.mark.fast
    def test_start_inside_a_rebuild_does_not_clear_the_shutdown(self, provider):
        provider._streams_shut_down.set()
        provider._twm = MagicMock()
        provider._twm.start_margin_socket.return_value = "key"
        with provider._rebuild_scope():
            provider.start_user_stream(MagicMock())
        assert provider._streams_shut_down.is_set()

    @pytest.mark.fast
    def test_backoff_wait_wakes_on_shutdown(self, provider):
        """A reconnect sleeping between attempts is released by stop_streams, not by the timer."""
        provider._on_user_event_cb = MagicMock()
        attempts: list[int] = []

        def failing_start(_callback) -> bool:
            attempts.append(1)
            provider._streams_shut_down.set()  # shutdown arrives during the first attempt
            return False

        started = time.monotonic()
        with patch.object(provider, "start_user_stream", side_effect=failing_start):
            assert provider.reconnect_user() is False

        assert attempts == [1]
        assert time.monotonic() - started < 0.5  # the 1 s backoff was cut short

    @pytest.mark.fast
    def test_shutdown_waits_for_an_in_flight_rebuild(self, provider):
        """stop_streams takes the lifecycle lock, so it queues behind a rebuild."""
        provider._on_user_event_cb = MagicMock()
        release = threading.Event()
        in_teardown = threading.Event()

        def slow_teardown() -> None:
            in_teardown.set()
            release.wait(5)

        with (
            patch.object(provider, "_teardown_twm", side_effect=slow_teardown),
            patch.object(provider, "_ensure_twm"),
            patch.object(provider, "start_user_stream", return_value=True),
        ):
            rebuild = threading.Thread(target=provider.hard_reconnect_user)
            rebuild.start()
            assert in_teardown.wait(5)
            stopper = threading.Thread(target=provider.stop_streams)
            stopper.start()
            stopper.join(timeout=0.3)
            assert stopper.is_alive(), "stop_streams must wait for the rebuild"
            release.set()
            rebuild.join(5)
            stopper.join(5)
        assert provider._streams_shut_down.is_set() is True

    @pytest.mark.fast
    def test_hard_reconnect_refuses_when_this_provider_hosts_kline(self, provider):
        """A full teardown would kill the kline stream sharing the TWM."""
        provider._on_user_event_cb = MagicMock()
        provider._kline_socket_key = "btcusdt@kline_1h"
        with patch.object(provider, "_teardown_twm") as teardown:
            assert provider.hard_reconnect_user() is False
        teardown.assert_not_called()


class TestTeardown:
    """Quiet, bounded TWM teardown (#729)."""

    @pytest.mark.fast
    def test_dead_listener_keys_are_released_when_the_thread_will_not_exit(self, provider):
        """A listener that died never removes its key, pinning the manager thread."""
        twm = MagicMock()
        twm._socket_running = {None: True}
        twm.is_alive.side_effect = [True, False]
        provider._twm = twm
        provider._twm_loop = None

        provider._teardown_twm()

        assert twm._socket_running == {}
        assert twm.join.call_count == 2
        assert provider._twm is None

    @pytest.mark.fast
    def test_a_thread_that_will_not_exit_leaves_its_loop_open(self, provider, caplog):
        """Never close a loop another thread may still be running."""
        loop = MagicMock()
        loop.is_closed.return_value = False
        loop.is_running.return_value = False
        twm = MagicMock()
        twm._socket_running = {}
        twm.is_alive.return_value = True
        provider._twm = twm
        provider._twm_loop = loop

        with (
            caplog.at_level(logging.WARNING),
            patch.object(provider, "_close_stopped_loop") as close,
        ):
            provider._teardown_twm()

        close.assert_not_called()
        assert any("leaving its loop open" in r.getMessage() for r in caplog.records)
        assert provider._twm is None  # still retired, so the next start builds a fresh one

    @pytest.mark.fast
    def test_clean_exit_does_not_touch_the_socket_keys(self, provider):
        twm = MagicMock()
        twm._socket_running = {"live": True}
        twm.is_alive.return_value = False
        provider._twm = twm
        provider._twm_loop = None

        provider._teardown_twm()

        assert twm._socket_running == {"live": True}
        assert twm.join.call_count == 1

    @pytest.mark.fast
    def test_teardown_clears_every_socket_and_event_flag(self, provider):
        twm = MagicMock()
        twm.is_alive.return_value = False
        provider._twm = twm
        provider._twm_loop = None
        provider._kline_socket_key = "k"
        provider._user_socket_key = "u"
        provider._user_socket_active = True
        provider._user_event_received = True
        generation = provider._user_stream_generation

        provider._teardown_twm()

        assert provider._user_stream_generation == generation + 1
        assert provider._user_event_received is False
        assert provider._kline_socket_key is None
        assert provider._user_socket_key is None
        assert provider._user_socket_active is False
        assert provider._user_ws_state == WebSocketState.DISCONNECTED

    @pytest.mark.fast
    def test_stopped_loop_is_drained_before_it_is_closed(self):
        """Pending library tasks are cancelled and finished, not destroyed with the loop."""
        loop = asyncio.new_event_loop()
        finished = []

        async def linger() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                finished.append("cancelled")
                raise

        loop.create_task(linger())
        loop.run_until_complete(asyncio.sleep(0))  # let the task start

        BinanceProvider._close_stopped_loop(loop)

        assert finished == ["cancelled"]
        assert loop.is_closed()

    @pytest.mark.fast
    def test_stopped_loop_closes_a_ws_api_reopened_by_the_exit_unsubscribe(self):
        """The listener's exit-unsubscribe can re-open the ws-api after the first close."""
        loop = asyncio.new_event_loop()
        closed: list[str] = []
        ws_api = MagicMock()

        async def close() -> None:
            closed.append("closed")

        ws_api.ws.close = close

        BinanceProvider._close_stopped_loop(loop, ws_api)

        assert closed == ["closed"]
        assert loop.is_closed()

    @pytest.mark.fast
    def test_ws_api_quiesce_marks_exiting_and_closes_the_socket(self):
        """EXITING is set before the close so the read loop stops instead of reconnecting."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            order: list[object] = []
            ws_api = MagicMock()

            async def close() -> None:
                order.append(ws_api.ws_state)

            ws_api.ws.close = close
            ws_api._handle_read_loop = None  # read loop already gone after the first pass

            BinanceProvider._close_ws_api_transport(ws_api, loop)

            assert order == [WSListenerState.EXITING]
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()

    @pytest.mark.fast
    def test_ws_api_quiesce_is_a_noop_without_a_running_loop(self):
        ws_api = MagicMock()
        BinanceProvider._close_ws_api_transport(ws_api, None)
        stopped = asyncio.new_event_loop()
        try:
            BinanceProvider._close_ws_api_transport(ws_api, stopped)
        finally:
            stopped.close()
        ws_api.ws.close.assert_not_called()

    @pytest.mark.fast
    def test_close_filter_drops_only_our_own_close_reports(self):
        """Only the library's report of OUR close is dropped; real errors still log."""
        close_filter = _ExpectedTransportCloseFilter()

        def record(message: str) -> logging.LogRecord:
            return logging.LogRecord("binance", logging.ERROR, __file__, 1, message, None, None)

        assert (
            close_filter.filter(
                record("ConnectionClosedOK (sent 1000 (OK); then received 1000 (OK))")
            )
            is False
        )
        assert (
            close_filter.filter(
                record("BinanceWebsocketClosed (Connection closed. Reconnecting...)")
            )
            is False
        )
        assert close_filter.filter(record("Unknown exception: ValueError (boom)")) is True

    @pytest.mark.fast
    def test_close_filter_only_mutes_the_closing_thread(self):
        """Another provider's genuine close report, on another thread, still logs."""
        close_filter = _ExpectedTransportCloseFilter(thread_ident=123)
        ours = logging.LogRecord(
            "binance", logging.ERROR, __file__, 1, "ConnectionClosedOK", None, None
        )
        ours.thread = 123
        other = logging.LogRecord(
            "binance", logging.ERROR, __file__, 1, "ConnectionClosedOK", None, None
        )
        other.thread = 456

        assert close_filter.filter(ours) is False
        assert close_filter.filter(other) is True

    @pytest.mark.fast
    def test_close_filter_is_installed_only_inside_the_context(self):
        ws_logger = logging.getLogger("binance.ws.reconnecting_websocket")
        before = list(ws_logger.filters)

        with _muted_expected_transport_close():
            installed = [f for f in ws_logger.filters if f not in before]
            assert len(installed) == 1
            assert isinstance(installed[0], _ExpectedTransportCloseFilter)

        assert ws_logger.filters == before


def _start_kline(provider, twm):
    """Start a kline stream on ``twm`` and return (callbacks seen, consumer mock)."""
    callbacks: list = []

    def start_kline_socket(callback, symbol, interval):
        callbacks.append(callback)
        return "btcusdt@kline_1h"

    twm.start_kline_socket.side_effect = start_kline_socket
    provider._twm = twm
    consumer = MagicMock()
    assert provider.start_kline_stream("BTCUSDT", "1h", consumer) is True
    return callbacks, consumer


class TestKlineErrorTeardown:
    """The kline error callback stops the dead socket at once (#665)."""

    @pytest.mark.fast
    def test_error_storm_tears_the_socket_down_exactly_once(self, provider, caplog):
        twm = MagicMock()
        callbacks, _ = _start_kline(provider, twm)

        with caplog.at_level(logging.ERROR):
            for _ in range(50):
                callbacks[0](
                    {"e": "error", "type": "ReadLoopClosed", "m": "Read loop has been closed"}
                )

        twm.stop_socket.assert_called_once_with("btcusdt@kline_1h")
        assert provider._kline_ws_state == WebSocketState.RESYNCING
        assert provider._kline_socket_key is None
        assert len([r for r in caplog.records if "Kline WS error" in r.getMessage()]) == 1

    @pytest.mark.fast
    def test_superseded_socket_errors_are_dropped(self, provider):
        """A late error from the previous socket must not stop or flag the new one."""
        twm = MagicMock()
        old_callbacks, _ = _start_kline(provider, twm)
        new_callbacks, _ = _start_kline(provider, twm)

        old_callbacks[0]({"e": "error", "m": "late"})

        twm.stop_socket.assert_not_called()
        assert provider._kline_ws_state == WebSocketState.PRIMARY
        assert provider._kline_socket_key == "btcusdt@kline_1h"
        assert new_callbacks[0] is not old_callbacks[0]

    @pytest.mark.fast
    def test_error_before_the_key_is_assigned_still_flags_resyncing(self, provider):
        """An error that beats the start's return flags RESYNCING; start must not undo it."""
        twm = MagicMock()

        def start_kline_socket(callback, symbol, interval):
            callback({"e": "error", "m": "instant failure"})
            return "btcusdt@kline_1h"

        twm.start_kline_socket.side_effect = start_kline_socket
        provider._twm = twm
        provider.start_kline_stream("BTCUSDT", "1h", MagicMock())

        assert provider._kline_ws_state == WebSocketState.RESYNCING
        twm.stop_socket.assert_not_called()

    @pytest.mark.fast
    def test_stop_socket_failure_does_not_escape_the_callback(self, provider):
        twm = MagicMock()
        twm.stop_socket.side_effect = RuntimeError("manager gone")
        callbacks, _ = _start_kline(provider, twm)

        callbacks[0]({"e": "error", "m": "x"})  # must not raise

        assert provider._kline_ws_state == WebSocketState.RESYNCING

    @pytest.mark.fast
    def test_reconnect_after_teardown_restores_event_flow(self, provider):
        """Error -> immediate teardown -> health-monitor reconnect -> events flow again."""
        twm = MagicMock()
        callbacks, consumer = _start_kline(provider, twm)
        callbacks[0]({"e": "error", "m": "x"})
        assert provider._kline_event_received is False

        assert provider.reconnect_kline() is True
        callbacks[-1]({"e": "kline", "k": {"t": 1}})

        assert len(callbacks) == 2
        consumer.assert_called_once()
        assert provider._kline_ws_state == WebSocketState.PRIMARY
        assert provider._kline_event_received is True
        assert provider.ws_healthy is True

    @pytest.mark.fast
    def test_failed_start_does_not_leave_a_primary_state(self, provider):
        twm = MagicMock()
        twm.start_kline_socket.side_effect = RuntimeError("no BSM")
        provider._twm = twm
        assert provider.start_kline_stream("BTCUSDT", "1h", MagicMock()) is False
        assert provider._kline_ws_state == WebSocketState.DISCONNECTED

    @pytest.mark.fast
    def test_real_listener_stops_spinning_on_read_loop_closed(self, provider):
        """Real ``start_listener`` over a socket whose read loop is gone (the #662 zombie).

        Without the teardown the listener re-reads ``ReadLoopClosed`` as fast as it can
        spin and never exits; with it, the run flag is cleared on the first error and
        the listener leaves after a single further read.
        """
        from binance.exceptions import ReadLoopClosed

        reads = 0

        class ZombieSocket:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def recv(self):
                nonlocal reads
                reads += 1
                raise ReadLoopClosed("Read loop has been closed")

        async def scenario() -> ThreadedApiManager:
            manager = ThreadedApiManager.__new__(ThreadedApiManager)
            path = "btcusdt@kline_1h"
            manager._socket_running = {path: True}
            manager._log = logging.getLogger("test.binance")
            twm = MagicMock()
            twm.stop_socket.side_effect = manager.stop_socket  # the real flag flip
            callbacks, _ = _start_kline(provider, twm)
            await asyncio.wait_for(manager.start_listener(ZombieSocket(), path, callbacks[0]), 5)
            return manager

        manager = asyncio.run(scenario())

        assert manager._socket_running == {}  # the listener left and removed its key
        assert reads <= 2
        assert provider._kline_ws_state == WebSocketState.RESYNCING
