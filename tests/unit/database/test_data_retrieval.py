"""Data retrieval tests for DatabaseManager."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


class TestDataRetrieval:
    """Test data retrieval methods"""

    def test_get_active_positions(self, mock_postgresql_db):
        """Test getting active positions"""
        mock_position = Mock()
        mock_position.id = 1
        mock_position.symbol = "BTCUSDT"
        mock_position.side.value = "LONG"
        mock_position.entry_price = 45000.0
        mock_position.current_price = 46000.0
        mock_position.size = 0.1
        mock_position.unrealized_pnl = 100.0
        mock_position.unrealized_pnl_percent = 2.22
        mock_position.stop_loss = 43000.0
        mock_position.take_profit = 47000.0
        mock_position.entry_time = datetime.utcnow()
        mock_position.strategy_name = "TestStrategy"

        mock_order = Mock()
        mock_order.id = 1
        mock_order.order_type.value = "MARKET"
        mock_order.status.value = "FILLED"
        mock_order.exchange_order_id = "12345"
        mock_order.internal_order_id = "int_123"
        mock_order.side.value = "BUY"
        mock_order.quantity = 0.1
        mock_order.price = 45000.0
        mock_order.filled_quantity = 0.1
        mock_order.filled_price = 45000.0
        mock_order.commission = 0.0
        mock_order.created_at = datetime.utcnow()
        mock_order.filled_at = datetime.utcnow()
        mock_order.cancelled_at = None
        mock_order.target_level = None
        mock_order.size_fraction = None

        def mock_query_side_effect(model):
            if model.__name__ == "Position":
                mock_positions_query = Mock()
                mock_positions_query.filter.return_value.all.return_value = [mock_position]
                return mock_positions_query
            if model.__name__ == "Order":
                mock_orders_query = Mock()
                mock_orders_query.filter.return_value.order_by.return_value.all.return_value = [mock_order]
                return mock_orders_query
            return Mock()

        mock_postgresql_db._mock_session.query.side_effect = mock_query_side_effect

        positions = mock_postgresql_db.get_active_positions()

        assert isinstance(positions, list)
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"

    def test_get_recent_trades(self, mock_postgresql_db):
        """Test getting recent trades"""
        mock_trade = Mock()
        mock_trade.id = 1
        mock_trade.symbol = "BTCUSDT"
        mock_trade.side.value = "LONG"
        mock_trade.entry_price = 45000.0
        mock_trade.exit_price = 46000.0
        mock_trade.size = 0.1
        mock_trade.pnl = 100.0
        mock_trade.pnl_percent = 2.22
        mock_trade.entry_time = datetime.utcnow()
        mock_trade.exit_time = datetime.utcnow()
        mock_trade.exit_reason = "Take profit"
        mock_trade.strategy_name = "TestStrategy"

        mock_query = Mock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_trade]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        trades = mock_postgresql_db.get_recent_trades(limit=10)

        assert isinstance(trades, list)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTCUSDT"

    def test_get_performance_metrics(self, mock_postgresql_db):
        """Test getting performance metrics"""
        mock_trade = Mock()
        mock_trade.pnl = 100.0
        mock_trade.exit_time = datetime.utcnow()
        mock_trade.entry_time = datetime.utcnow() - timedelta(hours=1)

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [mock_trade]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        metrics = mock_postgresql_db.get_performance_metrics(session_id=123)

        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
