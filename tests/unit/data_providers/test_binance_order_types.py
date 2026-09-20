"""Tests for the shared Binance order-type mapping (#1158) and NOTIONAL filter (#1062)."""

import logging
from unittest.mock import Mock, patch

import pytest

from src.data_providers.binance_order_types import BINANCE_ORDER_TYPE_MAP, map_binance_order_type
from src.data_providers.exchange_interface import OrderType
from src.engines.live.order_tracker import OrderTracker


def _make_provider(client=None):
    from src.data_providers.binance_provider import BinanceProvider

    with (
        patch("src.data_providers.binance_provider.Client") as client_cls,
        patch("src.data_providers.binance_provider.get_config") as cfg,
    ):
        cfg.return_value = Mock(get_required=Mock(return_value="k"))
        client_cls.return_value = client or Mock()
        return BinanceProvider()


@pytest.mark.fast
class TestBinanceOrderTypeMap:
    @pytest.mark.parametrize(
        ("binance_type", "expected"),
        [
            ("MARKET", OrderType.MARKET),
            ("LIMIT", OrderType.LIMIT),
            ("STOP_LOSS", OrderType.STOP_LOSS),
            ("STOP_LOSS_LIMIT", OrderType.STOP_LOSS),
            ("TAKE_PROFIT", OrderType.TAKE_PROFIT),
            ("TAKE_PROFIT_LIMIT", OrderType.TAKE_PROFIT),
        ],
    )
    def test_known_types(self, binance_type, expected):
        assert map_binance_order_type(binance_type) == expected

    def test_unknown_type_defaults_to_market_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert map_binance_order_type("LIMIT_MAKER") == OrderType.MARKET
        assert "LIMIT_MAKER" in caplog.text

    @pytest.mark.parametrize("binance_type", sorted(BINANCE_ORDER_TYPE_MAP))
    def test_rest_and_websocket_paths_agree(self, binance_type):
        """Both call sites resolve through the one table, so they cannot drift (#1152)."""
        rest = _make_provider()._convert_order_type(binance_type)
        assert rest == OrderTracker._map_ws_order_type(binance_type)


@pytest.mark.fast
class TestSymbolInfoMinNotional:
    """Binance returns a NOTIONAL filter, not MIN_NOTIONAL (#1062)."""

    @staticmethod
    def _min_notional(extra_filters):
        client = Mock()
        client.get_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
                        *extra_filters,
                    ],
                }
            ]
        }
        return _make_provider(client).get_symbol_info("ETHUSDT")["min_notional"]

    def test_reads_notional_filter(self):
        assert self._min_notional([{"filterType": "NOTIONAL", "minNotional": "5.0"}]) == 5.0

    def test_falls_back_to_legacy_min_notional_filter(self):
        assert self._min_notional([{"filterType": "MIN_NOTIONAL", "minNotional": "10"}]) == 10.0

    def test_missing_filter_defaults_to_zero(self):
        assert self._min_notional([]) == 0.0
