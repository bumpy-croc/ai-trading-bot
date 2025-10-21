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
