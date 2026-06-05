"""Event and monitoring logging tests for DatabaseManager."""

from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.unit


class TestEventLogging:
    """Test event logging methods"""

    def test_log_event(self, mock_postgresql_db):
        """Test logging a system event"""
        mock_event_obj = Mock()
        mock_event_obj.id = 101

        with patch("database.manager.SystemEvent") as mock_event_class:
            mock_event_class.return_value = mock_event_obj

            event_id = mock_postgresql_db.log_event(
                event_type="TEST", message="Test event message", severity="info"
            )

            assert event_id == 101
            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()

    def test_log_event_persists_observability_fields(self, mock_postgresql_db):
        """log_event forwards error_code/stack_trace/alert_sent/alert_method."""
        mock_event_obj = Mock()
        mock_event_obj.id = 202

        with patch("database.manager.SystemEvent") as mock_event_class:
            mock_event_class.return_value = mock_event_obj

            event_id = mock_postgresql_db.log_event(
                event_type="ERROR",
                message="Reconciler fell back to legacy path",
                severity="error",
                component="reconciler",
                error_code="RECONCILER_FALLBACK",
                stack_trace="Traceback (most recent call last): boom",
                alert_sent=True,
                alert_method="webhook",
            )

            assert event_id == 202
            _, kwargs = mock_event_class.call_args
            assert kwargs["error_code"] == "RECONCILER_FALLBACK"
            assert kwargs["stack_trace"] == "Traceback (most recent call last): boom"
            assert kwargs["alert_sent"] is True
            assert kwargs["alert_method"] == "webhook"
            assert kwargs["component"] == "reconciler"
            assert kwargs["severity"] == "error"

    def test_log_event_observability_fields_default_safely(self, mock_postgresql_db):
        """Existing callers get backward-compatible defaults for the new fields."""
        mock_event_obj = Mock()
        mock_event_obj.id = 303

        with patch("database.manager.SystemEvent") as mock_event_class:
            mock_event_class.return_value = mock_event_obj

            mock_postgresql_db.log_event(
                event_type="TEST", message="Legacy caller", severity="info"
            )

            _, kwargs = mock_event_class.call_args
            assert kwargs["error_code"] is None
            assert kwargs["stack_trace"] is None
            assert kwargs["alert_sent"] is False
            assert kwargs["alert_method"] is None

    def test_log_account_snapshot(self, mock_postgresql_db):
        """Test logging account snapshot"""
        with patch("database.manager.AccountHistory"):
            mock_postgresql_db.log_account_snapshot(
                balance=10100.0,
                equity=10150.0,
                total_pnl=100.0,
                open_positions=1,
                total_exposure=1000.0,
                drawdown=0.5,
            )

            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()

    def test_log_strategy_execution(self, mock_postgresql_db):
        """Test logging strategy execution"""
        with patch("database.manager.StrategyExecution"):
            mock_postgresql_db.log_strategy_execution(
                strategy_name="TestStrategy",
                symbol="BTCUSDT",
                signal_type="BUY",
                action_taken="OPENED_POSITION",
                price=45000.0,
            )

            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()
