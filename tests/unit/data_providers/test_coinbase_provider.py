import os
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_providers.coinbase_provider import CoinbaseProvider

pytestmark = pytest.mark.unit


class TestCoinbaseProvider:
    @pytest.mark.data_provider
    def test_initialization_without_keys(self):
        # Test with no credentials - should work in test environment
        with patch.dict(os.environ, {"ENV": "test"}, clear=False):
            # Ensure no Coinbase credentials are set
            for key in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"]:
                os.environ.pop(key, None)
            provider = CoinbaseProvider()
            assert provider is not None

    @pytest.mark.data_provider
    @patch("data_providers.coinbase_provider.requests.Session.get")
    def test_historical_data_fetch(self, mock_get):
        sample_candles = [
            [1640998800, 49000, 51000, 49500, 50000, 12.3],
            [1640995200, 48000, 50000, 48500, 49500, 10.0],
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_candles
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"ENV": "test"}, clear=False):
            # Ensure no Coinbase credentials are set
            for key in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"]:
                os.environ.pop(key, None)
            provider = CoinbaseProvider()
            start_date = datetime.utcfromtimestamp(1640995200)
            end_date = datetime.utcfromtimestamp(1640998800)
            df = provider.get_historical_data("BTC-USD", "1h", start_date, end_date)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in df.columns
                assert pd.api.types.is_numeric_dtype(df[col])
            assert df.index[0] < df.index[1]

    @pytest.mark.data_provider
    @patch("data_providers.coinbase_provider.requests.Session.get")
    def test_current_price(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"price": "12345.67"}
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"ENV": "test"}, clear=False):
            # Ensure no Coinbase credentials are set
            for key in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"]:
                os.environ.pop(key, None)
            provider = CoinbaseProvider()
            price = provider.get_current_price("BTC-USD")
            assert price == 12345.67


class TestCoinbaseGetOrderChecked:
    """Fail-closed order lookup (#713): None only on a confirmed 404."""

    @staticmethod
    def _provider() -> CoinbaseProvider:
        with patch.dict(os.environ, {"ENV": "test"}, clear=False):
            for key in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"]:
                os.environ.pop(key, None)
            return CoinbaseProvider()

    @staticmethod
    def _http_error(status_code: int):
        import requests

        response = Mock()
        response.status_code = status_code
        return requests.exceptions.HTTPError(response=response)

    def test_returns_none_on_confirmed_404(self):
        provider = self._provider()
        with patch.object(provider, "_request", side_effect=self._http_error(404)):
            assert provider.get_order_checked("abc-123", "BTC-USD") is None

    def test_raises_on_server_error(self):
        from src.data_providers.exchange_interface import OrderLookupError

        provider = self._provider()
        with patch.object(provider, "_request", side_effect=self._http_error(503)):
            with pytest.raises(OrderLookupError):
                provider.get_order_checked("abc-123", "BTC-USD")

    def test_raises_on_network_error(self):
        from src.data_providers.exchange_interface import OrderLookupError

        provider = self._provider()
        with patch.object(provider, "_request", side_effect=ConnectionError("reset")):
            with pytest.raises(OrderLookupError):
                provider.get_order_checked("abc-123", "BTC-USD")

    def test_returns_order_when_found(self):
        provider = self._provider()
        payload = {
            "id": "abc-123",
            "product_id": "BTC-USD",
            "side": "sell",
            "type": "limit",
            "size": "0.5",
            "filled_size": "0.0",
            "price": "50000.0",
            "status": "open",
            "created_at": "2026-01-01T00:00:00Z",
        }
        with patch.object(provider, "_request", return_value=payload):
            order = provider.get_order_checked("abc-123", "BTC-USD")
        assert order is not None
        assert order.order_id == "abc-123"


def test_exchange_interface_default_get_order_checked_fails_closed():
    """The ABC default must raise (unconfirmed), never silently delegate to a
    get_order that swallows errors into None (#713)."""
    from src.data_providers.exchange_interface import (
        ExchangeInterface,
        OrderLookupError,
    )

    class _MinimalExchange(ExchangeInterface):
        def _initialize_client(self):
            pass

        def test_connection(self):
            return True

        def get_account_info(self):
            return {}

        def get_balances(self):
            return []

        def get_balance(self, asset):
            return None

        def get_positions(self, symbol=None):
            return []

        def get_open_orders(self, symbol=None):
            return []

        def get_order(self, order_id, symbol):
            return None  # swallows errors — exactly why the default must raise

        def get_recent_trades(self, symbol, limit=100):
            return []

        def place_order(self, *a, **k):
            return None

        def cancel_order(self, order_id, symbol):
            return True

        def cancel_all_orders(self, symbol=None):
            return True

        def get_symbol_info(self, symbol):
            return None

        def place_stop_loss_order(self, *a, **k):
            return None

    exchange = _MinimalExchange("key", "secret")
    with pytest.raises(OrderLookupError):
        exchange.get_order_checked("123", "BTCUSDT")


class TestCoinbaseOrderTypeMapping:
    """Regression tests for #762: every order type was submitted as MARKET.

    ``OrderType.LIMIT.value`` is ``"LIMIT"`` (uppercase) while the old mapping
    was keyed lowercase, so ``mapping.get(...)`` always fell back to
    ``"market"`` — silently stripping price protection from limit orders and
    firing stop orders immediately.
    """

    @staticmethod
    def _provider() -> CoinbaseProvider:
        with patch.dict(os.environ, {"ENV": "test"}, clear=False):
            for key in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"]:
                os.environ.pop(key, None)
            return CoinbaseProvider()

    @pytest.mark.fast
    def test_enum_order_types_map_correctly(self):
        """Each supported OrderType maps to the matching Coinbase string."""
        from src.data_providers.exchange_interface import OrderType

        provider = self._provider()

        assert provider._convert_to_cb_type(OrderType.MARKET) == "market"
        assert provider._convert_to_cb_type(OrderType.LIMIT) == "limit"
        assert provider._convert_to_cb_type(OrderType.STOP_LOSS) == "stop"

    @pytest.mark.fast
    def test_string_order_types_map_case_insensitively(self):
        """Legacy string inputs (either case) resolve to the right type."""
        provider = self._provider()

        assert provider._convert_to_cb_type("limit") == "limit"
        assert provider._convert_to_cb_type("LIMIT") == "limit"
        assert provider._convert_to_cb_type("stop_loss") == "stop"
        assert provider._convert_to_cb_type("market") == "market"

    @pytest.mark.fast
    def test_unsupported_order_type_raises(self):
        """Unknown types raise instead of silently degrading to market."""
        from src.data_providers.exchange_interface import OrderType

        provider = self._provider()

        with pytest.raises(ValueError, match="Unsupported Coinbase order type"):
            provider._convert_to_cb_type(OrderType.TAKE_PROFIT)
        with pytest.raises(ValueError, match="Unsupported Coinbase order type"):
            provider._convert_to_cb_type("trailing_stop")

    @pytest.mark.fast
    def test_place_order_limit_builds_limit_body(self):
        """A limit order reaches the API as type=limit with its price."""
        from src.data_providers.exchange_interface import OrderSide, OrderType

        provider = self._provider()
        with patch.object(
            provider, "_request", autospec=True, return_value={"id": "ord-1"}
        ) as mock_request:
            order = provider.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.5,
                price=50000.0,
            )

        assert order is not None
        body = mock_request.call_args.kwargs.get("body") or mock_request.call_args.args[2]
        assert body["type"] == "limit"
        assert body["price"] == "50000.0"

    @pytest.mark.fast
    def test_place_order_gtd_rejected(self):
        """GTD requires an end_time this client cannot send — fail before the API."""
        from src.data_providers.exchange_interface import OrderSide, OrderType

        provider = self._provider()
        with patch.object(provider, "_request", autospec=True) as mock_request:
            order = provider.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.5,
                price=50000.0,
                time_in_force="GTD",
            )

        assert order is None
        mock_request.assert_not_called()
