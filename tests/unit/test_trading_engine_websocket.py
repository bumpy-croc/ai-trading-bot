"""Tests for LiveTradingEngine WebSocket integration."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from src.data_providers.binance_provider import WebSocketState


@pytest.fixture
def mock_engine():
    """Create a LiveTradingEngine with mocked dependencies for testing WS integration."""
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


class TestGetLatestDataWithWebSocket:
    """Tests for _get_latest_data() WebSocket cache integration."""

    @pytest.mark.fast
    def test_returns_cache_when_ws_healthy(self, mock_engine):
        """Returns cached DataFrame when WS is healthy."""
        mock_buffer = MagicMock()
        cached_df = _make_df(10)
        mock_buffer.get_dataframe.return_value = cached_df
        mock_buffer.is_fresh = True

        mock_ws_provider = MagicMock()
        mock_ws_provider.ws_healthy = True
        mock_ws_provider.ws_state = WebSocketState.PRIMARY

        mock_engine._kline_buffer = mock_buffer
        mock_engine._ws_kline_provider = mock_ws_provider

        result = mock_engine._get_latest_data("BTCUSDT", "1h")
        assert result is cached_df
        mock_engine.data_provider.get_live_data.assert_not_called()

    @pytest.mark.fast
    def test_falls_back_to_rest_when_ws_unhealthy(self, mock_engine):
        """Falls back to REST when WS is not healthy."""
        mock_buffer = MagicMock()
        mock_buffer.is_fresh = True

        mock_ws_provider = MagicMock()
        mock_ws_provider.ws_healthy = False

        mock_engine._kline_buffer = mock_buffer
        mock_engine._ws_kline_provider = mock_ws_provider

        result = mock_engine._get_latest_data("BTCUSDT", "1h")
        mock_engine.data_provider.get_live_data.assert_called_once()

    @pytest.mark.fast
    def test_returns_none_during_resyncing_live_mode(self, mock_engine):
        """Returns None during RESYNCING in live mode to freeze trading."""
        mock_engine.enable_live_trading = True

        mock_ws_provider = MagicMock()
        mock_ws_provider._kline_ws_state = WebSocketState.RESYNCING

        mock_engine._ws_kline_provider = mock_ws_provider
        mock_engine._kline_buffer = MagicMock()

        result = mock_engine._get_latest_data("BTCUSDT", "1h")
        assert result is None

    @pytest.mark.fast
    def test_falls_back_to_rest_during_resyncing_paper_mode(self, mock_engine):
        """Paper mode falls back to REST during RESYNCING instead of freezing."""
        mock_engine.enable_live_trading = False

        mock_ws_provider = MagicMock()
        mock_ws_provider.ws_state = WebSocketState.RESYNCING
        mock_ws_provider.ws_healthy = False

        mock_engine._ws_kline_provider = mock_ws_provider
        mock_engine._kline_buffer = MagicMock()
        mock_engine._kline_buffer.is_fresh = False

        result = mock_engine._get_latest_data("BTCUSDT", "1h")
        mock_engine.data_provider.get_live_data.assert_called_once()

    @pytest.mark.fast
    def test_no_ws_falls_back_to_rest(self, mock_engine):
        """Without WS setup, falls back to REST (existing behavior)."""
        result = mock_engine._get_latest_data("BTCUSDT", "1h")
        mock_engine.data_provider.get_live_data.assert_called_once()


class TestIsDataFreshWithWebSocket:
    """Tests for _is_data_fresh() WebSocket bypass."""

    @pytest.mark.fast
    def test_uses_buffer_freshness_when_ws_active(self, mock_engine):
        """Bypasses candle-timestamp check when WS kline cache is active and healthy."""
        mock_buffer = MagicMock()
        mock_buffer.is_fresh = True
        mock_engine._kline_buffer = mock_buffer
        mock_engine._ws_kline_active = True
        mock_ws_provider = MagicMock()
        mock_ws_provider.ws_healthy = True
        mock_engine._ws_kline_provider = mock_ws_provider

        # Create df with old timestamp that would fail normal freshness check
        old_df = _make_df(5)
        old_df.index = pd.date_range(
            end=datetime.now(UTC) - timedelta(hours=2), periods=5, freq="1h", tz="UTC"
        )

        assert mock_engine._is_data_fresh(old_df) is True

    @pytest.mark.fast
    def test_uses_candle_timestamp_when_no_ws(self, mock_engine):
        """Uses normal freshness check when WS is not active."""
        mock_engine._ws_kline_active = False

        fresh_df = _make_df(5)
        assert mock_engine._is_data_fresh(fresh_df) is True


class TestStopWithWebSocket:
    """Tests for stop() WebSocket cleanup."""

    @pytest.mark.fast
    def test_stop_cleans_up_ws_resources(self, mock_engine):
        """stop() stops WS streams and user data processor."""
        mock_kline_provider = MagicMock()
        mock_engine._ws_kline_provider = mock_kline_provider

        mock_exchange = MagicMock()
        mock_engine.exchange_interface = mock_exchange

        mock_udp = MagicMock()
        mock_engine._user_data_processor = mock_udp

        mock_engine.is_running = True
        mock_engine.main_thread = MagicMock()
        mock_engine.main_thread.is_alive.return_value = False

        mock_engine.stop()

        mock_kline_provider.stop_streams.assert_called_once()
        mock_exchange.stop_streams.assert_called_once()
        mock_udp.stop.assert_called_once()

    @pytest.mark.fast
    def test_stop_handles_no_ws_gracefully(self, mock_engine):
        """stop() works when no WS was set up."""
        mock_engine.is_running = True
        mock_engine.main_thread = MagicMock()
        mock_engine.main_thread.is_alive.return_value = False
        mock_engine.stop()  # Should not raise
