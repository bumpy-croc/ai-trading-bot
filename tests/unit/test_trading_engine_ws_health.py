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
        """Should call _handle_kline_disconnect when ws_healthy is False."""
        mock_provider = MagicMock()
        mock_provider.ws_healthy = False
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._ws_kline_active = True

        with patch.object(mock_engine, "_handle_kline_disconnect") as mock_disconnect:
            mock_engine._check_kline_health()
            mock_disconnect.assert_called_once()

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


class TestHandleKlineDisconnect:
    """Tests for _handle_kline_disconnect()."""

    @pytest.mark.fast
    def test_resyncs_buffer_and_reconnects(self, mock_engine):
        """Should resync from REST and reconnect kline stream."""
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
        assert mock_engine._ws_kline_active is True

    @pytest.mark.fast
    def test_sets_rest_degraded_on_reconnect_failure(self, mock_engine):
        """Should fall back to REST_DEGRADED when reconnect fails."""
        mock_provider = MagicMock()
        mock_provider.reconnect_kline.return_value = False
        mock_engine._ws_kline_provider = mock_provider
        mock_engine._ws_kline_active = True

        mock_buffer = MagicMock()
        mock_engine._kline_buffer = mock_buffer

        mock_engine._handle_kline_disconnect()

        assert mock_provider._kline_ws_state == WebSocketState.REST_DEGRADED
        assert mock_engine._ws_kline_active is False


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

        mock_tracker.poll_once.assert_called_once()
        mock_reconciler.reconcile_once.assert_called_once()
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

        assert mock_exchange._user_ws_state == WebSocketState.REST_DEGRADED
        mock_tracker.enable_polling.assert_called_once()
