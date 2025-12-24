from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

try:
    from src.data_providers.binance_provider import (
        BinanceProvider,
        STOP_LOSS_LIMIT_SLIPPAGE_FACTOR,
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
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "invalid", datetime.now())
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
        mock_client.create_order.side_effect = BinanceOrderException(
            Mock(), "Order rejected"
        )
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
