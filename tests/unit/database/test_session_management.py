"""Session management tests for DatabaseManager."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSessionManagement:
    """Test session management methods"""

    def test_create_trading_session(self, mock_postgresql_db):
        """Test creating a trading session"""
        mock_session_obj = Mock()
        mock_session_obj.id = 123
        mock_postgresql_db._mock_session.add.return_value = None

        with patch("database.manager.TradingSession") as mock_trading_session_class:
            mock_trading_session_class.return_value = mock_session_obj

            session_id = mock_postgresql_db.create_trading_session(
                strategy_name="TestStrategy",
                symbol="BTCUSDT",
                timeframe="1h",
                mode="PAPER",
                initial_balance=10000.0,
            )

            assert session_id == 123
            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()

    def test_end_trading_session(self, mock_postgresql_db):
        """Test ending a trading session"""
        mock_session_obj = Mock()
        mock_session_obj.id = 123
        mock_session_obj.session_name = "test_session"
        mock_session_obj.start_time = datetime.now(UTC)

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_session_obj
        mock_query.filter_by.return_value.all.return_value = []
        mock_postgresql_db._mock_session.query.return_value = mock_query

        mock_postgresql_db._current_session_id = 123
        mock_postgresql_db.end_trading_session()

        mock_postgresql_db._mock_session.commit.assert_called()
