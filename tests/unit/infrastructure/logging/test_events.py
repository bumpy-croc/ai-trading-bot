"""Tests for infrastructure.logging.events module."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.logging.events import (
    log_data_event,
    log_db_event,
    log_decision_event,
    log_engine_error,
    log_engine_event,
    log_engine_warning,
    log_order_event,
    log_risk_event,
)


class TestLogEngineEvent:
    """Tests for log_engine_event function."""

    def test_logs_at_info_level(self):
        """Test that engine events are logged at INFO level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_event("Engine started")
            mock_logger.log.assert_called_once()
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_includes_event_type(self):
        """Test that event_type is included in extra."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_event("Test message")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "engine_event"

    def test_includes_custom_fields(self):
        """Test that custom fields are included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_event("Test", strategy="ml_basic", symbol="BTCUSDT")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["strategy"] == "ml_basic"
            assert extra["symbol"] == "BTCUSDT"


class TestLogEngineWarning:
    """Tests for log_engine_warning function."""

    def test_logs_at_warning_level(self):
        """Test that warnings are logged at WARNING level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_warning("Connection unstable")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.WARNING

    def test_includes_event_type(self):
        """Test that event_type is engine_event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_warning("Warning message")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "engine_event"


class TestLogEngineError:
    """Tests for log_engine_error function."""

    def test_logs_at_error_level(self):
        """Test that errors are logged at ERROR level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_error("Critical failure")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.ERROR

    def test_includes_error_details(self):
        """Test that error details can be included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_engine_error("Failed", error_code="E001", stack_trace="...")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["error_code"] == "E001"


class TestLogDecisionEvent:
    """Tests for log_decision_event function."""

    def test_logs_at_info_level(self):
        """Test that decision events are logged at INFO level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_decision_event("Trade decision made")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_includes_decision_event_type(self):
        """Test that event_type is decision_event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_decision_event("Decision")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "decision_event"

    def test_includes_decision_details(self):
        """Test that decision details can be included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_decision_event(
                "Signal generated",
                signal="BUY",
                confidence=0.85,
                price=50000.0,
            )
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["signal"] == "BUY"
            assert extra["confidence"] == 0.85
            assert extra["price"] == 50000.0


class TestLogOrderEvent:
    """Tests for log_order_event function."""

    def test_logs_at_info_level(self):
        """Test that order events are logged at INFO level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_order_event("Order placed")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_includes_order_event_type(self):
        """Test that event_type is order_event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_order_event("Order")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "order_event"

    def test_includes_order_details(self):
        """Test that order details can be included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_order_event(
                "Order filled",
                order_id="12345",
                side="BUY",
                quantity=0.5,
                fill_price=50100.0,
            )
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["order_id"] == "12345"
            assert extra["side"] == "BUY"


class TestLogRiskEvent:
    """Tests for log_risk_event function."""

    def test_logs_at_info_level(self):
        """Test that risk events are logged at INFO level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_risk_event("Risk limit checked")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_includes_risk_event_type(self):
        """Test that event_type is risk_event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_risk_event("Risk check")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "risk_event"

    def test_includes_risk_metrics(self):
        """Test that risk metrics can be included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_risk_event(
                "Position sized",
                risk_per_trade=0.02,
                max_drawdown=0.15,
                current_exposure=0.1,
            )
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["risk_per_trade"] == 0.02
            assert extra["max_drawdown"] == 0.15


class TestLogDataEvent:
    """Tests for log_data_event function."""

    def test_logs_at_info_level(self):
        """Test that data events are logged at INFO level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_data_event("Data fetched")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_includes_data_event_type(self):
        """Test that event_type is data_event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_data_event("Data")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "data_event"

    def test_includes_data_details(self):
        """Test that data details can be included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_data_event(
                "OHLCV data received",
                symbol="BTCUSDT",
                timeframe="1h",
                candle_count=100,
            )
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["symbol"] == "BTCUSDT"
            assert extra["candle_count"] == 100


class TestLogDbEvent:
    """Tests for log_db_event function."""

    def test_logs_at_info_level(self):
        """Test that DB events are logged at INFO level."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_db_event("Query executed")
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_includes_db_event_type(self):
        """Test that event_type is db_event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_db_event("DB operation")
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["event_type"] == "db_event"

    def test_includes_db_details(self):
        """Test that DB details can be included."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            log_db_event(
                "Trade saved",
                table="trades",
                operation="INSERT",
                row_count=1,
            )
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["table"] == "trades"
            assert extra["operation"] == "INSERT"


@pytest.mark.fast
class TestContextIntegration:
    """Tests for context integration with events."""

    def test_context_merged_into_event(self):
        """Test that context values are merged into event."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            with patch(
                "src.infrastructure.logging.events.get_context",
                return_value={"request_id": "abc123", "session_id": "sess1"},
            ):
                log_engine_event("Test", custom_field="value")
                call_args = mock_logger.log.call_args
                extra = call_args[1]["extra"]
                assert extra["request_id"] == "abc123"
                assert extra["session_id"] == "sess1"
                assert extra["custom_field"] == "value"

    def test_explicit_fields_override_context(self):
        """Test that explicit fields override context values."""
        with patch("src.infrastructure.logging.events._logger") as mock_logger:
            with patch(
                "src.infrastructure.logging.events.get_context",
                return_value={"key": "context_value"},
            ):
                log_engine_event("Test", key="explicit_value")
                call_args = mock_logger.log.call_args
                extra = call_args[1]["extra"]
                assert extra["key"] == "explicit_value"
