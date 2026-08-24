"""Durable capture of exchange order failures (#1094).

The 2026-08-20 prod close-only latch followed five stop-loss placement
failures whose Binance error code existed only in application logs, which then
aged out. These tests pin the durable record: the exchange code, its message
and the rejected order parameters must reach ``system_events`` — while the
control flow (``None``/``False`` returns, caller latching) stays byte-identical.
"""

from unittest.mock import Mock, patch

import pytest

from src.data_providers.binance_provider import BINANCE_AVAILABLE, BinanceProvider
from src.data_providers.exchange_interface import (
    ExchangeOrderError,
    OrderSide,
    OrderType,
)
from src.database.models import EventType
from src.engines.live.execution.execution_engine import LiveExecutionEngine


def _binance_provider(mock_client_class, mock_config, client: Mock) -> BinanceProvider:
    mock_config_obj = Mock()
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config.return_value = mock_config_obj
    mock_client_class.return_value = client
    client.get_exchange_info.return_value = {"symbols": []}
    return BinanceProvider()


class _RecordingSink:
    """Captures every ExchangeOrderError handed to the sink."""

    def __init__(self) -> None:
        self.errors: list[ExchangeOrderError] = []

    def __call__(self, error: ExchangeOrderError) -> None:
        self.errors.append(error)


class TestExchangeOrderErrorRecord:
    """The record itself: known-cause annotation and the details payload."""

    @pytest.mark.parametrize(
        "code,fragment",
        [
            (51077, "LOT_SIZE"),
            (-1111, "PRICE_FILTER"),
            (-2010, "insufficient free balance"),
        ],
    )
    def test_known_reject_codes_are_named(self, code, fragment):
        error = ExchangeOrderError(
            operation="place_stop_loss_order",
            symbol="BTCUSDT",
            error_message="rejected",
            error_code=code,
        )
        assert fragment in (error.known_cause or "")
        assert fragment in error.summary()

    def test_unknown_code_has_no_known_cause(self):
        error = ExchangeOrderError(
            operation="place_order", symbol="BTCUSDT", error_message="boom", error_code=-9999
        )
        assert error.known_cause is None
        assert "code=-9999" in error.summary()

    def test_details_payload_is_serialisable_and_complete(self):
        error = ExchangeOrderError(
            operation="place_stop_loss_order",
            symbol="ETHUSDT",
            error_message="Filter failure: LOT_SIZE",
            error_code=51077,
            error_type="BinanceOrderException",
            params={"quantity": 0.004000000000000001, "stopPrice": "1648.82"},
        )
        details = error.to_details()
        assert details["error_code"] == 51077
        assert details["rejected_params"]["quantity"] == 0.004000000000000001
        assert details["known_cause"] is not None
        assert details["occurred_at"]


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestStopLossErrorCapture:
    """place_stop_loss_order must record WHY, without changing what it returns."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_rejection_captures_code_message_and_params(self, mock_config, mock_client_class):
        from binance.exceptions import BinanceOrderException

        client = Mock()
        client.create_order.side_effect = BinanceOrderException(
            -1111, "Precision is over the maximum defined for this asset."
        )
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Control flow unchanged.
        assert result is None

        assert len(sink.errors) == 1
        error = sink.errors[0]
        assert error.operation == "place_stop_loss_order"
        assert error.symbol == "BTCUSDT"
        assert error.error_code == -1111
        assert "Precision" in error.error_message
        assert "PRICE_FILTER" in (error.known_cause or "")
        # The rejected request parameters travel with the error.
        assert error.params["side"] == "SELL"
        assert error.params["type"] == "STOP_LOSS_LIMIT"
        assert float(error.params["quantity"]) == pytest.approx(0.1)
        assert float(error.params["stopPrice"]) == pytest.approx(50000.0)
        assert error.params["timeInForce"] == "GTC"
        assert "price" in error.params
        # And it is retrievable later without re-plumbing return values.
        assert provider.last_order_error is error

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_generic_exception_captured(self, mock_config, mock_client_class):
        client = Mock()
        client.create_order.side_effect = Exception("Read timed out")
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        assert (
            provider.place_stop_loss_order(
                symbol="BTCUSDT", side=OrderSide.SELL, quantity=0.1, stop_price=50000.0
            )
            is None
        )
        assert sink.errors[0].error_code is None
        assert "Read timed out" in sink.errors[0].error_message

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_missing_order_id_captured(self, mock_config, mock_client_class):
        client = Mock()
        client.create_order.return_value = {"status": "NEW"}
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        assert (
            provider.place_stop_loss_order(
                symbol="BTCUSDT", side=OrderSide.SELL, quantity=0.1, stop_price=50000.0
            )
            is None
        )
        assert sink.errors[0].error_type == "MissingOrderId"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_invalid_stop_price_captured(self, mock_config, mock_client_class):
        client = Mock()
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        assert (
            provider.place_stop_loss_order(
                symbol="BTCUSDT", side=OrderSide.SELL, quantity=0.1, stop_price=0.0
            )
            is None
        )
        client.create_order.assert_not_called()
        assert sink.errors[0].error_type == "InvalidStopPrice"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sink_failure_does_not_propagate(self, mock_config, mock_client_class):
        from binance.exceptions import BinanceOrderException

        client = Mock()
        client.create_order.side_effect = BinanceOrderException(-2010, "insufficient balance")
        provider = _binance_provider(mock_client_class, mock_config, client)

        def exploding_sink(error):
            raise RuntimeError("observability backend down")

        provider.order_error_sink = exploding_sink

        # The observability failure is swallowed; the order path still returns None.
        assert (
            provider.place_stop_loss_order(
                symbol="BTCUSDT", side=OrderSide.SELL, quantity=0.1, stop_price=50000.0
            )
            is None
        )

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_success_records_nothing(self, mock_config, mock_client_class):
        client = Mock()
        client.create_order.return_value = {"orderId": "12345"}
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        assert (
            provider.place_stop_loss_order(
                symbol="BTCUSDT", side=OrderSide.SELL, quantity=0.1, stop_price=50000.0
            )
            == "12345"
        )
        assert sink.errors == []
        assert provider.last_order_error is None


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestOrderAndCancelErrorCapture:
    """The sibling safety-critical paths: entry/exit placement and cancellation."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_definitive_reject_still_raises_value_error_and_records(
        self, mock_config, mock_client_class
    ):
        from binance.exceptions import BinanceOrderException

        client = Mock()
        client.create_order.side_effect = BinanceOrderException(
            -2010, "Account has insufficient balance for requested action."
        )
        provider = _binance_provider(mock_client_class, mock_config, client)
        # Bypass the local filter check so the exchange rejection is reached.
        provider.validate_order_parameters = lambda *args, **kwargs: (True, None)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        # Control flow unchanged: definitive rejects still raise ValueError so
        # the caller can distinguish them from ambiguous failures.
        with pytest.raises(ValueError, match="code=-2010"):
            provider.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
            )

        assert sink.errors[0].operation == "place_order"
        assert sink.errors[0].error_code == -2010
        assert sink.errors[0].params["symbol"] == "BTCUSDT"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_unavailable_client_is_recorded(self, mock_config, mock_client_class):
        """A client that failed to init at boot must not be a silent None."""
        client = Mock()
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink
        provider._client = None

        assert (
            provider.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
            )
            is None
        )
        assert sink.errors[0].operation == "place_order"
        assert sink.errors[0].error_type == "ClientUnavailable"
        assert sink.errors[0].params["symbol"] == "BTCUSDT"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_cancel_failure_records_and_returns_false(self, mock_config, mock_client_class):
        client = Mock()
        client.cancel_order.side_effect = Exception("gateway error")
        provider = _binance_provider(mock_client_class, mock_config, client)
        sink = _RecordingSink()
        provider.order_error_sink = sink

        assert provider.cancel_order("999", "BTCUSDT") is False
        assert sink.errors[0].operation == "cancel_order"
        assert sink.errors[0].params["orderId"] == "999"


class TestExecutionEngineSink:
    """The sink lands the record in system_events, fault-isolated."""

    def _engine(self) -> tuple[LiveExecutionEngine, Mock]:
        engine = LiveExecutionEngine()
        db = Mock()
        engine.db_manager = db
        engine.session_id = 7
        return engine, db

    def test_stop_loss_error_written_as_critical_system_event(self):
        engine, db = self._engine()
        engine._record_exchange_order_error(
            ExchangeOrderError(
                operation="place_stop_loss_order",
                symbol="BTCUSDT",
                error_message="Precision is over the maximum",
                error_code=-1111,
                params={"quantity": 0.1},
            )
        )

        db.log_event.assert_called_once()
        kwargs = db.log_event.call_args.kwargs
        assert kwargs["event_type"] is EventType.ERROR
        assert kwargs["error_code"] == "STOP_LOSS_PLACEMENT_FAILED"
        assert kwargs["severity"] == "critical"
        assert kwargs["session_id"] == 7
        assert kwargs["details"]["error_code"] == -1111
        assert kwargs["details"]["rejected_params"] == {"quantity": 0.1}
        assert "PRICE_FILTER" in kwargs["message"]

    def test_order_error_written_at_error_severity(self):
        engine, db = self._engine()
        engine._record_exchange_order_error(
            ExchangeOrderError(
                operation="place_order", symbol="BTCUSDT", error_message="x", error_code=-2010
            )
        )
        kwargs = db.log_event.call_args.kwargs
        assert kwargs["error_code"] == "ORDER_PLACEMENT_FAILED"
        assert kwargs["severity"] == "error"

    def test_db_failure_does_not_propagate(self):
        engine, db = self._engine()
        db.log_event.side_effect = RuntimeError("db down")
        # Must not raise — observability can never break execution.
        engine._record_exchange_order_error(
            ExchangeOrderError(operation="place_order", symbol="BTCUSDT", error_message="x")
        )

    def test_attach_wires_sink_end_to_end(self):
        engine, db = self._engine()
        exchange = Mock()
        exchange.order_error_sink = None
        engine.attach_exchange_error_sink(exchange)
        exchange.order_error_sink(
            ExchangeOrderError(
                operation="place_stop_loss_order", symbol="BTCUSDT", error_message="rejected"
            )
        )
        assert db.log_event.call_args.kwargs["error_code"] == "STOP_LOSS_PLACEMENT_FAILED"

    def test_attach_ignores_exchange_without_hook(self):
        engine, _ = self._engine()
        engine.attach_exchange_error_sink(None)  # no exchange configured (paper mode)

        class _Legacy:
            pass

        legacy = _Legacy()
        engine.attach_exchange_error_sink(legacy)
        assert not hasattr(legacy, "order_error_sink")
