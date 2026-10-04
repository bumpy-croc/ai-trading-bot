"""User-data stream recovery against the REAL python-binance WS stack.

The user-data stream could never be restored in place (#616/#723): python-binance
caches one ``KeepAliveWebsocket`` per account socket, so an in-place re-subscribe
reuses an object whose old listener later unsubscribes the NEW subscription; the
first socket's key is ``None`` so its listener could not be stopped; and a subscribe
racing the ws-api's own reconnect kills the listener silently. These tests drive the
unmodified ``ThreadedWebsocketManager`` / ``WebsocketAPI`` against a local fake
ws-api server (no exchange connection) so the behaviour is proven end to end:
stream dies, is rebuilt, and events flow again with a clean teardown.
"""

import asyncio
import itertools
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("binance.ws.threaded_stream")
websockets = pytest.importorskip("websockets")

from binance.ws import threaded_stream  # noqa: E402
from binance.ws.websocket_api import WebsocketAPI  # noqa: E402

from src.data_providers.binance_provider import BinanceProvider, WebSocketState  # noqa: E402

# Library teardown artifacts that issue #729 asked to be gone from a normal rebuild.
TEARDOWN_ARTIFACTS = (
    "Error stopping client",
    "Task was destroyed but it is pending",
    "Task exception was never retrieved",
    "ConnectionClosedOK",
)


class FakeWsApiServer:
    """Fake Binance ws-api: answers subscribe/ping/unsubscribe, pushes events, can die."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.connections: list = []
        self.active_subscriptions: dict[int, object] = {}  # subscription id -> connection
        self.requests: list[str] = []
        self.silent = False
        self._subscription_ids = itertools.count(0)
        self.port = 0
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start(ready))
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, name="fake-ws-api-server", daemon=True)
        self.thread.start()
        assert ready.wait(5), "fake ws-api server did not start"

    async def _start(self, ready: threading.Event) -> None:
        self.server = await websockets.serve(self._handle, "127.0.0.1", 0, close_timeout=0.1)
        self.port = self.server.sockets[0].getsockname()[1]
        ready.set()

    async def _handle(self, connection) -> None:
        self.connections.append(connection)
        try:
            async for raw in connection:
                request = json.loads(raw)
                method = request["method"]
                self.requests.append(method)
                if self.silent:
                    continue
                result: dict = {}
                if method == "userDataStream.subscribe.listenToken":
                    subscription_id = next(self._subscription_ids)
                    self.active_subscriptions[subscription_id] = connection
                    result = {"subscriptionId": subscription_id}
                elif method == "userDataStream.unsubscribe":
                    self.active_subscriptions.pop(request["params"]["subscriptionId"], None)
                await connection.send(
                    json.dumps({"id": request["id"], "status": 200, "result": result})
                )
        except websockets.ConnectionClosed:
            pass
        finally:
            if connection in self.connections:
                self.connections.remove(connection)
            for subscription_id, owner in list(self.active_subscriptions.items()):
                if owner is connection:
                    del self.active_subscriptions[subscription_id]

    def _run(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop).result(5)

    def push_event(self, event: dict) -> None:
        """Push ``event`` on every subscription the server still considers active."""

        async def push() -> None:
            for subscription_id, connection in list(self.active_subscriptions.items()):
                try:
                    await connection.send(
                        json.dumps({"subscriptionId": subscription_id, "event": event})
                    )
                except websockets.ConnectionClosed:
                    pass

        self._run(push())

    def subscription_count(self) -> int:
        """Number of subscribe requests answered so far."""
        return self.requests.count("userDataStream.subscribe.listenToken")

    def kill_connections(self) -> None:
        """Drop every client connection, as Binance does on a ws-api reset."""

        async def kill() -> None:
            for connection in list(self.connections):
                await connection.close(code=1011)
            self.connections.clear()

        self._run(kill())

    def close(self) -> None:
        """Stop serving, close every connection and end the server thread."""

        async def shutdown() -> None:
            self.server.close()
            await self.server.wait_closed()

        self._run(shutdown())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


class LocalAsyncClient:
    """The slice of ``AsyncClient`` the margin user stream uses, over the REAL ``WebsocketAPI``."""

    tld = "com"
    testnet = False
    demo = False
    https_proxy = None

    def __init__(self, port: int) -> None:
        self.ws_api = WebsocketAPI(url=f"ws://127.0.0.1:{port}", tld="com")
        self.session = None

    async def margin_create_listen_token(self, is_isolated: bool = False) -> dict:
        return {"token": "listen-token", "expirationTime": 1}

    async def _ws_api_request(self, method: str, signed: bool, params: dict) -> dict:
        request_id = params.pop("id")
        return await self.ws_api.request(
            request_id, {"id": request_id, "method": method, "params": params}
        )

    async def ws_ping(self) -> dict:
        return await self._ws_api_request("ping", False, {"id": f"ping-{time.monotonic()}"})

    async def close_connection(self) -> None:
        await self.ws_api.close()


@pytest.fixture
def server() -> Iterator[FakeWsApiServer]:
    fake = FakeWsApiServer()
    yield fake
    fake.close()


@pytest.fixture
def provider(server: FakeWsApiServer) -> Iterator[BinanceProvider]:
    """A margin BinanceProvider whose TWM clients talk to the fake ws-api server."""

    async def create_client(*_args, **_kwargs) -> LocalAsyncClient:
        return LocalAsyncClient(server.port)

    with (
        patch.object(BinanceProvider, "_initialize_client"),
        patch.object(threaded_stream.AsyncClient, "create", create_client),
        patch(
            "src.data_providers.binance_provider.get_binance_api_endpoint", return_value="binance"
        ),
        # The library's 10 s request timeout would make a failed subscribe take 10 s.
        patch.object(WebsocketAPI, "TIMEOUT", 1),
    ):
        instance = BinanceProvider(api_key="test_key", api_secret="test_secret", testnet=True)
        instance._client = MagicMock()
        instance._use_margin = True
        try:
            yield instance
        finally:
            instance.stop_streams()


def wait_until(
    predicate: Callable[[], bool], timeout: float = 8.0, what: str = "condition"
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _all_twm_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if isinstance(t, threaded_stream.ThreadedApiManager)]


_TWM_THREADS_BEFORE_TEST: list[threading.Thread] = []


@pytest.fixture(autouse=True)
def _remember_twm_threads() -> None:
    """Scope twm_threads() to this test, whatever other tests in the process left behind."""
    _TWM_THREADS_BEFORE_TEST[:] = _all_twm_threads()


def twm_threads() -> list[threading.Thread]:
    """Manager threads started since the test began."""
    return [t for t in _all_twm_threads() if t not in _TWM_THREADS_BEFORE_TEST]


@pytest.mark.medium
def test_stream_dies_is_rebuilt_and_events_flow_again(provider, server, caplog):
    """The core regression: a dead ws-api is rebuilt without a restart and delivers."""
    delivered: list[tuple[str, dict]] = []
    assert provider.start_user_stream(lambda m: delivered.append(("first", m)))
    wait_until(lambda: server.subscription_count() == 1, what="first subscription")
    server.push_event({"e": "executionReport", "n": 1})
    wait_until(lambda: provider.user_ws_healthy, what="first event")

    server.kill_connections()
    wait_until(
        lambda: provider._user_ws_state == WebSocketState.RESYNCING, what="RESYNCING after death"
    )
    assert provider.user_ws_healthy is False

    with caplog.at_level(logging.DEBUG):
        caplog.clear()
        assert provider.hard_reconnect_user(lambda m: delivered.append(("rebuilt", m)))
    wait_until(lambda: server.subscription_count() == 2, what="re-subscription")

    server.push_event({"e": "executionReport", "n": 2})
    wait_until(lambda: provider.user_ws_healthy, what="event after rebuild")

    assert [tag for tag, _ in delivered] == ["first", "rebuilt"]
    assert provider._user_ws_state == WebSocketState.PRIMARY
    noisy = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and any(a in r.getMessage() for a in TEARDOWN_ARTIFACTS)
    ]
    assert noisy == []


@pytest.mark.medium
def test_repeated_rebuilds_keep_one_listener_and_one_thread(provider, server):
    """Every rebuild replaces the previous TWM: no stacked listeners, threads or loops."""
    delivered: list[int] = []
    loops = []
    for generation in range(3):
        if generation == 0:
            assert provider.start_user_stream(lambda m, g=generation: delivered.append(g))
        else:
            assert provider.hard_reconnect_user(lambda m, g=generation: delivered.append(g))
        loops.append(provider._twm_loop)
        wait_until(lambda g=generation: server.subscription_count() == g + 1, what="subscription")

    server.push_event({"e": "executionReport"})
    wait_until(lambda: provider.user_ws_healthy, what="event on the latest stream")
    time.sleep(0.3)

    assert delivered == [2]  # only the newest generation's callback saw the event
    assert len(twm_threads()) == 1
    wait_until(
        lambda: len(server.connections) == 1,
        what="retired ws-api sockets to close (not wait for GC)",
    )
    assert all(loop.is_closed() for loop in loops[:-1])
    assert server.requests.count("userDataStream.unsubscribe") >= 2


@pytest.mark.medium
def test_failed_subscribe_flags_resyncing_and_a_rebuild_recovers(provider, server):
    """A subscribe that times out kills python-binance's listener silently (#616).

    The provider must notice (the unretrieved task exception is the only signal) and
    flag RESYNCING, since a ws-api ping would still succeed and the stream would
    otherwise be held as idle-healthy forever.
    """
    delivered: list[dict] = []
    server.silent = True
    assert provider.start_user_stream(delivered.append)
    wait_until(
        lambda: provider._user_ws_state == WebSocketState.RESYNCING,
        timeout=8,
        what="RESYNCING after the subscribe timed out",
    )

    server.silent = False
    assert provider.hard_reconnect_user(delivered.append)
    wait_until(lambda: bool(server.active_subscriptions), what="successful subscription")
    server.push_event({"e": "executionReport", "n": 1})
    wait_until(lambda: provider.user_ws_healthy, what="event after recovery")

    assert provider._user_ws_state == WebSocketState.PRIMARY
    assert len(delivered) == 1
    # The failed subscribe's listener died holding its key; the rebuild must still
    # have let that manager thread exit, leaving exactly one.
    assert len(twm_threads()) == 1


@pytest.mark.medium
def test_stop_streams_leaves_no_thread_or_open_loop(provider, server):
    """Shutdown joins the TWM thread, closes its loop and refuses a later rebuild."""
    assert provider.start_user_stream(lambda m: None)
    wait_until(lambda: server.subscription_count() == 1, what="subscription")
    loop = provider._twm_loop

    provider.stop_streams()

    assert twm_threads() == []
    assert loop.is_closed()
    wait_until(lambda: server.connections == [], what="the ws-api socket to close")
    assert provider.hard_reconnect_user(lambda m: None) is False
    assert provider._twm is None
