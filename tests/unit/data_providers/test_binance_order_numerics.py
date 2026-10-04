"""Exchange-bound numerics sent by BinanceProvider are exactly on-step (#789).

Binance rejects `1648.8200000000001` as a stopPrice (-1111) and
`0.004000000000000001` as a quantity (51077), and -1100 on scientific notation
(`str(0.00009) == "9e-05"`). These tests pin the serialized request parameters,
not just the pre-serialization floats.
"""

import logging
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.exchange_interface import OrderSide, OrderType

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _symbol(name: str, step: str, tick: str) -> dict:
    return {
        "symbol": name,
        "baseAsset": name.removesuffix("USDT"),
        "quoteAsset": "USDT",
        "status": "TRADING",
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": step, "maxQty": "9000000", "stepSize": step},
            {
                "filterType": "PRICE_FILTER",
                "minPrice": tick,
                "maxPrice": "1000000",
                "tickSize": tick,
            },
            {"filterType": "NOTIONAL", "minNotional": "1"},
        ],
    }


@pytest.fixture
def provider():
    client = Mock()
    client.create_order.return_value = {"orderId": 42}
    client.get_exchange_info.return_value = {
        "symbols": [
            _symbol("ETHUSDT", "0.0001", "0.01"),
            _symbol("SHIBUSDT", "1", "0.00000001"),
        ]
    }
    with (
        patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True),
        patch("src.data_providers.binance_provider.Client", return_value=client),
        patch("src.data_providers.binance_provider.get_config") as get_config,
    ):
        get_config.return_value.get_required.return_value = "fake"
        provider = BinanceProvider()
        provider.get_balance = Mock(return_value=Mock(free=1_000_000_000.0))
        yield provider


def _sent(provider) -> dict:
    return provider._client.create_order.call_args.kwargs


class TestStopLossNumerics:
    def test_known_bad_price_is_sent_clean(self, provider):
        """1648.82 at tick 0.01: the float snap yields 1648.8200000000001."""
        provider.place_stop_loss_order(
            "ETHUSDT", OrderSide.SELL, quantity=0.004, stop_price=1648.82, limit_price=1648.82
        )
        sent = _sent(provider)
        assert sent["stopPrice"] == "1648.82"
        assert sent["price"] == "1648.82"

    def test_known_bad_quantity_is_sent_clean(self, provider):
        """40 * 0.0001 == 0.004000000000000001 in float math."""
        provider.place_stop_loss_order(
            "ETHUSDT", OrderSide.SELL, quantity=0.0040000000000001, stop_price=1700.0
        )
        assert _sent(provider)["quantity"] == "0.0040"

    def test_buy_cover_quantity_snaps_to_nearest_clean_lot(self, provider):
        provider.place_stop_loss_order(
            "ETHUSDT", OrderSide.BUY, quantity=0.30000000000000004, stop_price=1700.0
        )
        assert _sent(provider)["quantity"] == "0.3000"

    def test_tiny_price_is_fixed_point_not_scientific(self, provider):
        provider.place_stop_loss_order(
            "SHIBUSDT", OrderSide.SELL, quantity=1_000_000.0, stop_price=0.00001234
        )
        sent = _sent(provider)
        assert sent["stopPrice"] == "0.00001234"
        assert "e" not in sent["price"].lower()

    @pytest.mark.parametrize("stop", [1648.82, 1648.8249, 1648.8251, 1700.0049999, 0.01 * 3])
    def test_every_sent_price_is_on_tick(self, provider, stop):
        provider.place_stop_loss_order("ETHUSDT", OrderSide.SELL, quantity=0.5, stop_price=stop)
        sent = _sent(provider)
        for field in ("stopPrice", "price"):
            units = Decimal(sent[field]) / Decimal("0.01")
            assert units == units.to_integral_value(), f"{field}={sent[field]}"
            assert len(sent[field].partition(".")[2]) == 2


class TestPlaceOrderBoundaryGuard:
    def test_unquantized_quantity_is_corrected_and_logged(self, provider, caplog):
        provider.get_symbol_info("ETHUSDT")  # populates the symbol-info cache
        with caplog.at_level(logging.WARNING, logger="src.data_providers.binance_provider"):
            provider.place_order(
                "ETHUSDT", OrderSide.BUY, OrderType.MARKET, quantity=0.004000000000000001
            )
        assert _sent(provider)["quantity"] == "0.0040"
        assert any("Unquantized order quantity" in r.getMessage() for r in caplog.records)

    def test_clean_quantity_is_silent(self, provider, caplog):
        provider.get_symbol_info("ETHUSDT")
        with caplog.at_level(logging.WARNING, logger="src.data_providers.binance_provider"):
            provider.place_order("ETHUSDT", OrderSide.BUY, OrderType.MARKET, quantity=0.0123)
        assert _sent(provider)["quantity"] == "0.0123"
        assert not [r for r in caplog.records if "quantity" in r.getMessage()]

    def test_limit_price_and_stop_price_are_corrected_and_serialized(self, provider, caplog):
        provider.get_symbol_info("ETHUSDT")
        with caplog.at_level(logging.WARNING, logger="src.data_providers.binance_provider"):
            provider.place_order(
                "ETHUSDT",
                OrderSide.SELL,
                OrderType.STOP_LOSS,
                quantity=0.5,
                price=1648.8200000000001,
                stop_price=1648.8200000000001,
            )
        sent = _sent(provider)
        assert sent["price"] == "1648.82"
        assert sent["stopPrice"] == "1648.82"
        messages = [r.getMessage() for r in caplog.records]
        assert any("order price" in m for m in messages)
        assert any("order stopPrice" in m for m in messages)

    def test_off_step_quantity_is_sent_unchanged_and_logged_as_error(self, provider, caplog):
        """A silently moved size is worse than a rejected one (LESSONS 1.13)."""
        provider.get_symbol_info("ETHUSDT")
        with caplog.at_level(logging.ERROR, logger="src.data_providers.binance_provider"):
            provider.place_order("ETHUSDT", OrderSide.BUY, OrderType.MARKET, quantity=0.00123)
        assert _sent(provider)["quantity"] == "0.00123"
        assert any(
            r.levelno == logging.ERROR and "not on the exchange step" in r.getMessage()
            for r in caplog.records
        )

    def test_integer_step_quantity_is_fixed_point(self, provider):
        provider.get_symbol_info("SHIBUSDT")
        provider.place_order("SHIBUSDT", OrderSide.BUY, OrderType.MARKET, quantity=1.0)
        assert _sent(provider)["quantity"] == "1.0"

    def test_unknown_step_falls_back_to_plain_fixed_point(self, provider):
        """No cached symbol info: still never scientific notation."""
        with patch.object(provider, "validate_order_parameters", return_value=(True, None)):
            provider.place_order("ETHUSDT", OrderSide.BUY, OrderType.MARKET, quantity=0.00009)
        assert _sent(provider)["quantity"] == "0.00009"


class TestMarginRepayAmount:
    def test_small_decimal_amount_is_fixed_point(self, provider):
        """str(Decimal("0.0000001")) == "1E-7", which Binance rejects as malformed."""
        provider._use_margin = True
        provider._client.margin_borrow_repay.return_value = {}
        assert provider.repay_margin_loan("ETH", Decimal("0.0000001")) is True
        assert provider._client.margin_borrow_repay.call_args.kwargs["amount"] == "0.0000001"


class TestCreateOrderChokePoint:
    def test_raw_float_params_are_serialized_fixed_point_and_logged(self, provider, caplog):
        with caplog.at_level(logging.ERROR, logger="src.data_providers.binance_provider"):
            provider._call_create_order(
                symbol="ETHUSDT", side="SELL", quantity=0.00009, price=1648.82, stopPrice=1e-05
            )
        sent = _sent(provider)
        assert sent["quantity"] == "0.00009"
        assert sent["price"] == "1648.82"
        assert sent["stopPrice"] == "0.00001"
        assert any("unserialized" in r.getMessage() for r in caplog.records)

    def test_serialized_params_pass_untouched_and_silent(self, provider, caplog):
        with caplog.at_level(logging.ERROR, logger="src.data_providers.binance_provider"):
            provider._call_create_order(symbol="ETHUSDT", side="BUY", quantity="0.0040")
        assert _sent(provider)["quantity"] == "0.0040"
        assert not caplog.records


class TestMarginRepayFloatAmount:
    def test_float_amount_is_not_rounded_to_six_places(self, provider):
        provider._use_margin = True
        provider._client.margin_borrow_repay.return_value = {}
        provider.repay_margin_loan("ETH", 0.00282625)  # type: ignore[arg-type]
        assert provider._client.margin_borrow_repay.call_args.kwargs["amount"] == "0.00282625"


class TestNonFiniteRefused:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_place_order_refuses_non_finite_quantity(self, provider, bad):
        provider.get_symbol_info("ETHUSDT")
        with patch.object(provider, "validate_order_parameters", return_value=(True, None)):
            result = provider.place_order("ETHUSDT", OrderSide.BUY, OrderType.MARKET, quantity=bad)
        assert result is None
        provider._client.create_order.assert_not_called()
