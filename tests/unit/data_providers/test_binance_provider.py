from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

try:
    from src.data_providers.binance_provider import (
        STOP_LOSS_LIMIT_SLIPPAGE_FACTOR,
        BinanceProvider,
        with_rate_limit_retry,
    )
    from src.data_providers.exchange_interface import OrderSide

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceProvider = Mock
    with_rate_limit_retry = None
    STOP_LOSS_LIMIT_SLIPPAGE_FACTOR = 0.005
    OrderSide = Mock


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestBinanceDataProvider:
    @pytest.mark.data_provider
    def test_binance_provider_initialization(self):
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            assert provider is not None

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_historical_data_success(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.return_value = [
            [
                1640995200000,
                "50000",
                "50100",
                "49900",
                "50050",
                "100",
                1640995259999,
                "5000000",
                1000,
                "50",
                "2500000",
                "0",
            ],
            [
                1640998800000,
                "50050",
                "50150",
                "49950",
                "50100",
                "110",
                1640998859999,
                "5500000",
                1100,
                "55",
                "2750000",
                "0",
            ],
        ]
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            df = provider.get_historical_data(
                "BTCUSDT", "1h", datetime(2022, 1, 1), datetime(2022, 1, 2)
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_api_error_handling(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.side_effect = Exception("API Error")
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                assert "api" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_rate_limit_handling(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        try:
            from binance.exceptions import BinanceAPIException

            mock_response = Mock()
            mock_response.text = '{"code": -1003, "msg": "Rate limit exceeded"}'
            exception_to_raise = BinanceAPIException(
                mock_response, status_code=429, text=mock_response.text
            )
        except (ImportError, TypeError, AttributeError):
            exception_to_raise = Exception("Rate limit exceeded")
        mock_client.get_historical_klines.side_effect = exception_to_raise
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                assert any(s in str(e).lower() for s in ["rate limit", "exceeded", "error"])

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_data_validation(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "invalid", datetime.now(UTC))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except (ValueError, Exception) as e:
                assert any(s in str(e).lower() for s in ["invalid", "timeframe", "error"])


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestRateLimitRetryDecorator:
    """Tests for the with_rate_limit_retry decorator."""

    def test_successful_call_no_retry(self):
        """Verify decorated function returns immediately on success."""
        # Arrange
        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        # Act
        result = success_func()

        # Assert
        assert result == "success"
        assert call_count == 1

    @patch("src.data_providers.binance_provider.time.sleep")
    def test_retry_on_rate_limit_error(self, mock_sleep):
        """Verify retry with exponential backoff on rate limit errors."""
        # Arrange
        from binance.exceptions import BinanceAPIException

        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=1.0)
        def rate_limited_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = BinanceAPIException(Mock(), 429, "Rate limited")
                exc.code = -1003  # Rate limit error code
                raise exc
            return "success"

        # Act
        result = rate_limited_then_success()

        # Assert
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2
        # Verify exponential backoff: 1s, 2s
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("src.data_providers.binance_provider.time.sleep")
    def test_max_retries_exhausted(self, mock_sleep):
        """Verify exception raised after max retries exhausted."""
        # Arrange
        from binance.exceptions import BinanceAPIException

        @with_rate_limit_retry(max_retries=2, base_delay=0.01)
        def always_rate_limited():
            exc = BinanceAPIException(Mock(), 429, "Rate limited")
            exc.code = -1003
            raise exc

        # Act & Assert
        with pytest.raises(BinanceAPIException):
            always_rate_limited()
        assert mock_sleep.call_count == 2

    def test_non_rate_limit_error_not_retried(self):
        """Verify non-rate-limit errors are raised immediately."""
        # Arrange
        from binance.exceptions import BinanceAPIException

        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=0.01)
        def other_error():
            nonlocal call_count
            call_count += 1
            exc = BinanceAPIException(Mock(), 400, "Bad request")
            exc.code = -1000  # Not a rate limit code
            raise exc

        # Act & Assert
        with pytest.raises(BinanceAPIException):
            other_error()
        assert call_count == 1

    def test_non_binance_exception_not_retried(self):
        """Verify non-Binance exceptions are raised immediately."""
        # Arrange
        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=0.01)
        def generic_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Something went wrong")

        # Act & Assert
        with pytest.raises(ValueError):
            generic_error()
        assert call_count == 1


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestPlaceStopLossOrder:
    """Tests for the place_stop_loss_order method."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_sell_order_success(self, mock_config, mock_client_class):
        """Verify successful stop-loss sell order placement."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "12345"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result == "12345"
        mock_client.create_order.assert_called_once()
        call_args = mock_client.create_order.call_args
        assert call_args.kwargs["type"] == "STOP_LOSS_LIMIT"
        assert call_args.kwargs["side"] == "SELL"
        assert call_args.kwargs["timeInForce"] == "GTC"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_buy_order_success(self, mock_config, mock_client_class):
        """Verify successful stop-loss buy order placement."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "67890"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result == "67890"
        call_args = mock_client.create_order.call_args
        assert call_args.kwargs["side"] == "BUY"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_auto_limit_price_calculation_sell(self, mock_config, mock_client_class):
        """Verify limit price is calculated below stop for sell orders."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "123"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        stop_price = 50000.0

        # Act
        provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=stop_price,
        )

        # Assert
        call_args = mock_client.create_order.call_args
        limit_price = float(call_args.kwargs["price"])
        expected_limit = stop_price * (1 - STOP_LOSS_LIMIT_SLIPPAGE_FACTOR)
        assert abs(limit_price - expected_limit) < 0.01

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_auto_limit_price_calculation_buy(self, mock_config, mock_client_class):
        """Verify limit price is calculated above stop for buy orders."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "123"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        stop_price = 50000.0

        # Act
        provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            stop_price=stop_price,
        )

        # Assert
        call_args = mock_client.create_order.call_args
        limit_price = float(call_args.kwargs["price"])
        expected_limit = stop_price * (1 + STOP_LOSS_LIMIT_SLIPPAGE_FACTOR)
        assert abs(limit_price - expected_limit) < 0.01

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_custom_limit_price_used(self, mock_config, mock_client_class):
        """Verify custom limit price overrides auto-calculation."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "123"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()

        # Act
        provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
            limit_price=49500.0,
        )

        # Assert
        call_args = mock_client.create_order.call_args
        limit_price = float(call_args.kwargs["price"])
        assert abs(limit_price - 49500.0) < 0.01

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_order_exception_returns_none(self, mock_config, mock_client_class):
        """Verify BinanceOrderException returns None."""
        # Arrange
        from binance.exceptions import BinanceOrderException

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.side_effect = BinanceOrderException(Mock(), "Order rejected")
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_generic_exception_returns_none(self, mock_config, mock_client_class):
        """Verify generic exception returns None."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.side_effect = Exception("Network error")
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_missing_order_id_returns_none(self, mock_config, mock_client_class):
        """Verify missing orderId in response returns None."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {}  # No orderId
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result is None


# ========================================
# Margin Trading Tests
# ========================================


def _make_config_mock(overrides: dict | None = None):
    """Create a config mock with sensible defaults and dict-backed .get().

    Args:
        overrides: Key-value pairs to override defaults (e.g. BINANCE_ACCOUNT_TYPE).

    Returns:
        Configured Mock object for get_config().
    """
    defaults = {
        "BINANCE_ACCOUNT_TYPE": "spot",
        "TRADING_MODE": "paper",
        "ENV": "test",
    }
    if overrides:
        defaults.update(overrides)

    mock_config_obj = Mock()
    mock_config_obj.get.side_effect = lambda key, default=None: defaults.get(key, default)
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config_obj.get_float.return_value = 60.0
    return mock_config_obj


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginFlag:
    """Tests for margin flag initialization from config."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_flag_from_env_margin(self, mock_config, mock_client_class):
        """Verify _use_margin=True when BINANCE_ACCOUNT_TYPE=margin."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }

        provider = BinanceProvider()
        assert provider._use_margin is True
        assert provider._is_live is False

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_flag_from_env_spot(self, mock_config, mock_client_class):
        """Verify _use_margin=False when BINANCE_ACCOUNT_TYPE=spot."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "spot"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = BinanceProvider()
        assert provider._use_margin is False

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_is_live_flag(self, mock_config, mock_client_class):
        """Verify _is_live=True when TRADING_MODE=live."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "spot",
            "TRADING_MODE": "live",
        })
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = BinanceProvider()
        assert provider._is_live is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_symbol_verified_cache_initialized(self, mock_config, mock_client_class):
        """Verify _margin_symbol_verified set is initialized empty."""
        mock_config.return_value = _make_config_mock()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = BinanceProvider()
        assert provider._margin_symbol_verified == set()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginFailFast:
    """Tests for live+margin fail-fast on client init failure."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_live_mode_no_offline_fallback(self, mock_config, mock_client_class):
        """Live+margin raises RuntimeError instead of falling back to offline stub."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        })
        mock_client_class.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="FATAL.*live margin mode"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_paper_mode_allows_offline_fallback(self, mock_config, mock_client_class):
        """Paper+margin falls back to offline stub (no RuntimeError)."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "paper",
        })
        mock_client_class.side_effect = Exception("Connection refused")

        provider = BinanceProvider()
        assert provider._client is not None  # offline stub

    @patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", False)
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_live_no_sdk_fails_fast(self, mock_config):
        """Live+margin raises RuntimeError when SDK is not installed."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        })

        with pytest.raises(RuntimeError, match="Binance library not available"):
            BinanceProvider()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginStartupChecks:
    """Tests for margin account verification at startup."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_checks_pass(self, mock_config, mock_client_class):
        """Startup passes when tradeEnabled and borrowEnabled are True."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }

        provider = BinanceProvider()
        assert provider._use_margin is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_trade_disabled_raises(self, mock_config, mock_client_class):
        """RuntimeError raised when tradeEnabled=False."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": False,
            "borrowEnabled": True,
            "marginLevel": "2.5",
        }

        with pytest.raises(RuntimeError, match="tradeEnabled=False"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_borrow_disabled_raises(self, mock_config, mock_client_class):
        """RuntimeError raised when borrowEnabled=False."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": False,
            "marginLevel": "2.5",
        }

        with pytest.raises(RuntimeError, match="borrowEnabled=False"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_api_error_live_raises(self, mock_config, mock_client_class):
        """API error during margin verification in live mode raises RuntimeError."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        })
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.side_effect = Exception("API timeout")

        with pytest.raises(RuntimeError, match="Failed to verify margin account"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_api_error_paper_warns(self, mock_config, mock_client_class):
        """API error during margin verification in paper mode logs warning, no raise."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "paper",
        })
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.side_effect = Exception("API timeout")

        # Should not raise — just warns
        provider = BinanceProvider()
        assert provider._use_margin is True


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginBaseAssetGuard:
    """Tests for the non-USDT base asset startup guard."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_rejects_non_usdt_holdings_in_live_mode(self, mock_config, mock_client_class):
        """Live margin mode raises if wallet holds significant non-USDT assets."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        })
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"},
                {"asset": "ETH", "free": "0.05", "locked": "0", "netAsset": "0.05"},
            ],
        }
        # Price lookup for ETH value estimation
        mock_client.get_symbol_ticker.return_value = {"price": "2000"}

        with pytest.raises(RuntimeError, match="holds.*ETH.*Transfer.*out"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_allows_dust_holdings(self, mock_config, mock_client_class):
        """Dust amounts (< $1) are ignored."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        })
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"},
                {"asset": "ETH", "free": "0.0001", "locked": "0", "netAsset": "0.0001"},
            ],
        }
        mock_client.get_symbol_ticker.return_value = {"price": "2000"}

        # $0.20 worth of ETH — below $1 threshold, should not raise
        provider = BinanceProvider()
        assert provider._use_margin is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_warns_in_paper_mode(self, mock_config, mock_client_class):
        """Paper mode warns but doesn't raise for non-USDT holdings."""
        mock_config.return_value = _make_config_mock({
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "paper",
        })
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"},
                {"asset": "ETH", "free": "0.05", "locked": "0", "netAsset": "0.05"},
            ],
        }
        mock_client.get_symbol_ticker.return_value = {"price": "2000"}

        # Paper mode — should warn, not raise
        provider = BinanceProvider()
        assert provider._use_margin is True


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginDispatch:
    """Tests for margin/spot dispatch routing."""

    def _make_provider(self, mock_config, mock_client_class, use_margin=True):
        """Helper to create a provider with margin config."""
        account_type = "margin" if use_margin else "spot"
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": account_type})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        if use_margin:
            mock_client.get_margin_account.return_value = {
                "tradeEnabled": True,
                "borrowEnabled": True,
                "marginLevel": "2.5",
                "userAssets": [],
            }
            mock_client.get_margin_symbol.return_value = {
                "isMarginTrade": True,
                "isBuyAllowed": True,
                "isSellAllowed": True,
            }
        provider = BinanceProvider()
        return provider, mock_client

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_dispatch_routes_to_margin_methods(self, mock_config, mock_client_class):
        """Dispatch methods call margin client methods when _use_margin=True."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, use_margin=True)

        # _call_get_account
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "userAssets": [{"asset": "BTC", "free": "1.0", "locked": "0.0"}],
        }
        provider._call_get_account()
        mock_client.get_margin_account.assert_called()

        # _call_create_order
        mock_client.create_margin_order.return_value = {"orderId": "123"}
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        mock_client.create_margin_order.assert_called()

        # _call_get_order
        provider._call_get_order(symbol="BTCUSDT", orderId="123")
        mock_client.get_margin_order.assert_called()

        # _call_get_open_orders
        mock_client.get_open_margin_orders.return_value = []
        provider._call_get_open_orders(symbol="BTCUSDT")
        mock_client.get_open_margin_orders.assert_called()

        # _call_get_my_trades
        mock_client.get_margin_trades.return_value = []
        provider._call_get_my_trades(symbol="BTCUSDT")
        mock_client.get_margin_trades.assert_called()

        # _call_cancel_order
        provider._call_cancel_order(symbol="BTCUSDT", orderId="123")
        mock_client.cancel_margin_order.assert_called()

        # _call_get_all_orders
        mock_client.get_all_margin_orders.return_value = []
        provider._call_get_all_orders(symbol="BTCUSDT")
        mock_client.get_all_margin_orders.assert_called()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_spot_dispatch_routes_to_spot_methods(self, mock_config, mock_client_class):
        """Dispatch methods call spot client methods when _use_margin=False."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, use_margin=False)

        # _call_get_account
        mock_client.get_account.return_value = {"balances": [], "canTrade": True}
        provider._call_get_account()
        mock_client.get_account.assert_called()

        # _call_create_order
        mock_client.create_order.return_value = {"orderId": "123"}
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        mock_client.create_order.assert_called()

        # _call_get_order
        provider._call_get_order(symbol="BTCUSDT", orderId="123")
        mock_client.get_order.assert_called()

        # _call_get_open_orders
        mock_client.get_open_orders.return_value = []
        provider._call_get_open_orders(symbol="BTCUSDT")
        mock_client.get_open_orders.assert_called()

        # _call_get_my_trades
        mock_client.get_my_trades.return_value = []
        provider._call_get_my_trades(symbol="BTCUSDT")
        mock_client.get_my_trades.assert_called()

        # _call_cancel_order
        provider._call_cancel_order(symbol="BTCUSDT", orderId="123")
        mock_client.cancel_order.assert_called()

        # _call_get_all_orders
        mock_client.get_all_orders.return_value = []
        provider._call_get_all_orders(symbol="BTCUSDT")
        mock_client.get_all_orders.assert_called()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_dispatch_adds_isIsolated_false(self, mock_config, mock_client_class):
        """Margin dispatch injects isIsolated=FALSE for cross-margin mode."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, use_margin=True)

        mock_client.get_open_margin_orders.return_value = []
        provider._call_get_open_orders(symbol="BTCUSDT")
        call_kwargs = mock_client.get_open_margin_orders.call_args.kwargs
        assert call_kwargs["isIsolated"] == "FALSE"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_spot_dispatch_strips_sideEffectType(self, mock_config, mock_client_class):
        """Spot dispatch removes sideEffectType param to avoid Binance error."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, use_margin=False)

        mock_client.create_order.return_value = {"orderId": "123"}
        provider._call_create_order(
            symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1,
            sideEffectType="MARGIN_BUY",
        )
        call_kwargs = mock_client.create_order.call_args.kwargs
        assert "sideEffectType" not in call_kwargs


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginBalanceNormalization:
    """Tests for margin account balance normalization."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_balance_normalization(self, mock_config, mock_client_class):
        """_call_get_account normalizes userAssets to balances format."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {
                    "asset": "BTC",
                    "free": "1.0",
                    "locked": "0.5",
                    "borrowed": "0.2",
                    "interest": "0.001",
                    "netAsset": "1.299",
                },
                {
                    "asset": "USDT",
                    "free": "10000",
                    "locked": "0",
                    "borrowed": "0",
                    "interest": "0",
                    "netAsset": "10000",
                },
            ],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }

        provider = BinanceProvider()
        result = provider._call_get_account()

        assert "balances" in result
        assert len(result["balances"]) == 2
        btc_bal = result["balances"][0]
        assert btc_bal["asset"] == "BTC"
        assert btc_bal["free"] == "1.0"
        assert btc_bal["locked"] == "0.5"
        assert btc_bal["borrowed"] == "0.2"
        assert btc_bal["interest"] == "0.001"
        assert btc_bal["netAsset"] == "1.299"
        assert result["canTrade"] is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_spot_balance_passthrough(self, mock_config, mock_client_class):
        """_call_get_account returns spot account data unchanged."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "spot"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        spot_data = {"balances": [{"asset": "BTC", "free": "1.0", "locked": "0"}], "canTrade": True}
        mock_client.get_account.return_value = spot_data

        provider = BinanceProvider()
        result = provider._call_get_account()
        assert result == spot_data


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginSymbolValidation:
    """Tests for lazy margin symbol validation."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_symbol_validation_on_first_order(self, mock_config, mock_client_class):
        """Symbol is validated on first margin order, cached for subsequent ones."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }
        mock_client.create_margin_order.return_value = {"orderId": "123"}

        provider = BinanceProvider()
        # Reset call count after __init__ verification
        mock_client.get_margin_symbol.reset_mock()

        # First BUY order triggers validation
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 1

        # Second BUY order with same symbol+side skips validation (cached)
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 1  # Still 1

        # SELL order for same symbol triggers new validation (different side)
        provider._call_create_order(symbol="BTCUSDT", side="SELL", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 2

        # Second SELL skips (cached)
        provider._call_create_order(symbol="BTCUSDT", side="SELL", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 2  # Still 2

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_symbol_not_supported_raises(self, mock_config, mock_client_class):
        """RuntimeError raised if symbol doesn't support margin trading."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        # Return unsupported on the order-time check
        mock_client.get_margin_symbol.side_effect = [
            # Called during _call_create_order for symbol validation
            {"isMarginTrade": False, "isBuyAllowed": True, "isSellAllowed": True},
        ]
        mock_client.create_margin_order.return_value = {"orderId": "123"}

        provider = BinanceProvider()

        with pytest.raises(RuntimeError, match="does not support margin trading"):
            provider._call_create_order(symbol="BADUSDT", side="BUY", type="MARKET", quantity=0.1)


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginSideEffectIntegration:
    """Verify side_effect_type flows through place_order/place_stop_loss_order."""

    # Minimal exchange_info symbol data that satisfies get_symbol_info
    _ETHUSDT_SYMBOL = {
        "symbol": "ETHUSDT",
        "baseAsset": "ETH",
        "quoteAsset": "USDT",
        "status": "TRADING",
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
        ],
    }

    def _make_margin_provider(self, mock_config, mock_client_class):
        """Create a margin-mode provider with all mocks configured."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }
        mock_client.get_exchange_info.return_value = {
            "symbols": [self._ETHUSDT_SYMBOL],
        }
        mock_client.create_margin_order.return_value = {
            "orderId": "99",
            "status": "FILLED",
            "origQty": "0.1",
            "executedQty": "0.1",
            "cummulativeQuoteQty": "200.0",
            "fills": [],
        }
        return BinanceProvider(), mock_client

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_order_passes_side_effect_type(self, mock_config, mock_client_class):
        """place_order() injects sideEffectType into margin create_margin_order call."""
        from src.data_providers.exchange_interface import OrderType

        provider, mock_client = self._make_margin_provider(mock_config, mock_client_class)

        provider.place_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.05,
            side_effect_type="MARGIN_BUY",
        )

        call_kwargs = mock_client.create_margin_order.call_args.kwargs
        assert call_kwargs["sideEffectType"] == "MARGIN_BUY"
        assert call_kwargs["isIsolated"] == "FALSE"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_passes_side_effect_type(self, mock_config, mock_client_class):
        """place_stop_loss_order() injects sideEffectType into margin order."""
        provider, mock_client = self._make_margin_provider(mock_config, mock_client_class)

        provider.place_stop_loss_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=0.05,
            stop_price=1800.0,
            side_effect_type="AUTO_REPAY",
        )

        call_kwargs = mock_client.create_margin_order.call_args.kwargs
        assert call_kwargs["sideEffectType"] == "AUTO_REPAY"
        assert call_kwargs["isIsolated"] == "FALSE"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_order_no_side_effect_in_spot_mode(self, mock_config, mock_client_class):
        """place_order() does not inject sideEffectType in spot mode."""
        from src.data_providers.exchange_interface import OrderType

        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "spot"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_exchange_info.return_value = {
            "symbols": [self._ETHUSDT_SYMBOL],
        }
        mock_client.create_order.return_value = {
            "orderId": "99",
            "status": "FILLED",
            "origQty": "0.1",
            "executedQty": "0.1",
            "cummulativeQuoteQty": "200.0",
            "fills": [],
        }

        provider = BinanceProvider()
        provider.place_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.05,
            side_effect_type="MARGIN_BUY",
        )

        call_kwargs = mock_client.create_order.call_args.kwargs
        assert "sideEffectType" not in call_kwargs


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginErrorCodes:
    """Verify margin-specific Binance error codes are treated as definitive rejects."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_insufficient_balance_raises(self, mock_config, mock_client_class):
        """Error -3041 (insufficient margin balance) raises ValueError, not None."""
        from binance.exceptions import BinanceAPIException
        from src.data_providers.exchange_interface import OrderType

        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True, "borrowEnabled": True,
            "marginLevel": "2.5", "userAssets": [],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True, "isBuyAllowed": True, "isSellAllowed": True,
        }
        mock_client.get_exchange_info.return_value = {
            "symbols": [{
                "symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT",
                "status": "TRADING", "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                    {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                ],
            }],
        }

        # Simulate Binance margin error -3041
        error = BinanceAPIException(
            Mock(status_code=400, headers={}),
            400,
            '{"code":-3041,"msg":"Balance is not enough"}',
        )
        error.code = -3041
        error.message = "Balance is not enough"
        mock_client.create_margin_order.side_effect = error

        provider = BinanceProvider()
        with pytest.raises(ValueError, match="Order rejected by exchange.*-3041"):
            provider.place_order(
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.05,
                side_effect_type="MARGIN_BUY",
            )
