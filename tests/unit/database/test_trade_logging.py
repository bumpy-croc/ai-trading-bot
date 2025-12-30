"""Trade and position logging tests for DatabaseManager."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.database.models import PositionSide, PositionStatus

pytestmark = pytest.mark.unit


class TestTradeLogging:
    """Test trade logging methods"""

    def test_log_trade(self, mock_postgresql_db):
        """Test logging a trade"""
        mock_trade_obj = Mock()
        mock_trade_obj.id = 456

        with patch("database.manager.Trade") as mock_trade_class:
            mock_trade_class.return_value = mock_trade_obj

            trade_id = mock_postgresql_db.log_trade(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=45000.0,
                exit_price=46000.0,
                size=0.1,
                entry_time=datetime.now(UTC),
                exit_time=datetime.now(UTC),
                pnl=100.0,
                exit_reason="Take profit",
                strategy_name="TestStrategy",
            )

            assert trade_id == 456
            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()

    def test_log_position(self, mock_postgresql_db):
        """Test logging a position"""
        mock_position_obj = Mock()
        mock_position_obj.id = 789

        with patch("database.manager.Position") as mock_position_class:
            mock_position_class.return_value = mock_position_obj

            position_id = mock_postgresql_db.log_position(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=45000.0,
                size=0.1,
                strategy_name="TestStrategy",
                entry_order_id="test_order_123",
            )

            assert position_id == 789
            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()

    def test_update_position(self, mock_postgresql_db):
        """Test updating a position"""
        mock_position = Mock()
        mock_position.side = PositionSide.LONG
        mock_position.entry_price = 45000.0

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_position
        mock_postgresql_db._mock_session.query.return_value = mock_query

        mock_postgresql_db.update_position(
            position_id=789,
            current_price=46000.0,
            unrealized_pnl=100.0,
        )

        mock_postgresql_db._mock_session.commit.assert_called()

    def test_close_position(self, mock_postgresql_db):
        """Test closing a position"""
        mock_position = Mock()

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_position
        mock_postgresql_db._mock_session.query.return_value = mock_query

        result = mock_postgresql_db.close_position(789)

        assert result is True
        assert mock_position.status == PositionStatus.CLOSED
        mock_postgresql_db._mock_session.commit.assert_called()
