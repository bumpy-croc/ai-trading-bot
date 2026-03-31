"""Tests for BinanceProvider WebSocket stream management."""

import threading
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
            patch("src.data_providers.binance_provider.get_binance_api_endpoint", return_value="binanceus"),
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
        """ws_healthy returns True when kline is PRIMARY even if user stream is RESYNCING."""
        provider._kline_ws_state = WebSocketState.PRIMARY
        provider._kline_event_received = True
        provider._user_ws_state = WebSocketState.RESYNCING
        provider._last_kline_event_time = datetime.now(UTC)
        assert provider.ws_healthy is True


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
