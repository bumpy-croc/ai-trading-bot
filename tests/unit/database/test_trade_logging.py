"""Trade and position logging tests for DatabaseManager."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.database.models import OrderStatus, PositionSide, PositionStatus

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

    def test_log_position_reuses_existing_journal_entry(self, mock_postgresql_db):
        """Regression: log_position must update an existing journal Order
        (created by create_order_journal_entry) instead of inserting a duplicate
        row, which would violate the exchange_order_id UNIQUE constraint."""
        mock_position_obj = Mock()
        mock_position_obj.id = 101
        mock_position_obj.session_id = 1

        # Simulate an existing journal Order row with matching client_order_id
        existing_journal = Mock()
        existing_journal.client_order_id = "atb_abc123"
        existing_journal.status = OrderStatus.PENDING_SUBMIT

        mock_query = Mock()
        # First query call: Order lookup by client_order_id returns the journal
        # Second query call would be for other queries
        mock_query.filter.return_value.first.return_value = existing_journal
        mock_postgresql_db._mock_session.query.return_value = mock_query

        with patch("src.database.manager.Position") as mock_position_class:
            mock_position_class.return_value = mock_position_obj

            mock_postgresql_db.log_position(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=45000.0,
                size=0.1,
                strategy_name="TestStrategy",
                entry_order_id="exch_order_999",
                client_order_id="atb_abc123",
            )

        # The existing journal row should be updated, not a new one added
        assert existing_journal.position_id == 101
        assert existing_journal.status == OrderStatus.FILLED
        assert existing_journal.exchange_order_id == "exch_order_999"
        assert existing_journal.filled_quantity == Decimal("0.1")
        assert existing_journal.filled_price == Decimal("45000.0")
        assert existing_journal.filled_at is not None

        # Verify no duplicate Order row was inserted via session.add for an Order
        # (session.add is called for Position, but NOT for a second Order)
        add_calls = mock_postgresql_db._mock_session.add.call_args_list
        added_types = [type(c[0][0]).__name__ for c in add_calls]
        assert added_types.count("Order") == 0, (
            f"Expected no new Order added when journal exists, got: {added_types}"
        )

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
