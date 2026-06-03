"""Data retrieval tests for DatabaseManager."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
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
        mock_position.entry_time = datetime.now(UTC)
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
        mock_order.created_at = datetime.now(UTC)
        mock_order.filled_at = datetime.now(UTC)
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
                mock_orders_query.filter.return_value.order_by.return_value.all.return_value = [
                    mock_order
                ]
                return mock_orders_query
            return Mock()

        mock_postgresql_db._mock_session.query.side_effect = mock_query_side_effect

        positions = mock_postgresql_db.get_active_positions()

        assert isinstance(positions, list)
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"

    def test_get_active_positions_coerces_numeric_fields_to_float(self, mock_postgresql_db):
        """Numeric columns come back from PostgreSQL as Decimal; get_active_positions
        must coerce them to float so downstream float arithmetic (e.g. the default
        stop-loss branches in reconciliation: entry_price * (1.0 ± DEFAULT_STOP_LOSS_PCT))
        never raises ``unsupported operand type(s) for *: 'Decimal' and 'float'``.
        """
        # Mirror real SQLAlchemy Numeric(18, 8) reads: every numeric field is a Decimal.
        mock_position = Mock()
        mock_position.id = 1
        mock_position.symbol = "BTCUSDT"
        mock_position.side.value = "LONG"
        mock_position.entry_price = Decimal("45000.12345678")
        mock_position.current_price = Decimal("46000.0")
        mock_position.size = Decimal("0.10000000")
        mock_position.quantity = Decimal("0.00500000")
        mock_position.entry_balance = Decimal("1000.0")
        mock_position.unrealized_pnl = Decimal("100.0")
        mock_position.unrealized_pnl_percent = Decimal("2.22")
        mock_position.stop_loss = Decimal("43000.0")
        mock_position.take_profit = Decimal("47000.0")
        mock_position.trailing_stop_price = Decimal("44000.0")
        mock_position.mfe = Decimal("0.03")
        mock_position.mae = Decimal("0.01")
        mock_position.mfe_price = Decimal("46500.0")
        mock_position.mae_price = Decimal("44500.0")
        mock_position.original_size = Decimal("1.0")
        mock_position.current_size = Decimal("0.5")
        mock_position.last_partial_exit_price = Decimal("46200.0")
        mock_position.last_scale_in_price = Decimal("44800.0")
        mock_position.entry_time = datetime.now(UTC)
        mock_position.strategy_name = "TestStrategy"

        def mock_query_side_effect(model):
            if model.__name__ == "Position":
                mock_positions_query = Mock()
                mock_positions_query.filter.return_value.all.return_value = [mock_position]
                return mock_positions_query
            if model.__name__ == "Order":
                mock_orders_query = Mock()
                mock_orders_query.filter.return_value.order_by.return_value.all.return_value = []
                return mock_orders_query
            return Mock()

        mock_postgresql_db._mock_session.query.side_effect = mock_query_side_effect

        positions = mock_postgresql_db.get_active_positions()
        position = positions[0]

        numeric_fields = [
            "entry_price",
            "current_price",
            "size",
            "quantity",
            "entry_balance",
            "unrealized_pnl",
            "unrealized_pnl_percent",
            "stop_loss",
            "take_profit",
            "trailing_stop_price",
            "mfe",
            "mae",
            "mfe_price",
            "mae_price",
            "original_size",
            "current_size",
            "last_partial_exit_price",
            "last_scale_in_price",
        ]
        for field in numeric_fields:
            value = position[field]
            assert not isinstance(value, Decimal), f"{field} leaked a Decimal: {value!r}"
            assert isinstance(value, float), f"{field} should be float, got {type(value).__name__}"

        # Values must be preserved through coercion, not just type-changed.
        assert position["entry_price"] == pytest.approx(45000.12345678)
        assert position["stop_loss"] == pytest.approx(43000.0)
        assert position["quantity"] == pytest.approx(0.005)

        # The Decimal * float bug must no longer reproduce on the recovered field.
        assert position["entry_price"] * (1.0 - 0.02) == pytest.approx(44100.12098765)

    def test_get_active_positions_null_numeric_fields_use_defaults(self, mock_postgresql_db):
        """Nullable numeric columns read back as NULL must surface as None (not a
        Decimal, and not a crash from a bare float(None)), while default=0.0 columns
        must surface as 0.0. Guards the nullable branch of the source coercion.
        """
        mock_position = Mock()
        mock_position.id = 7
        mock_position.symbol = "ETHUSDT"
        mock_position.side.value = "LONG"
        mock_position.strategy_name = "TestStrategy"
        mock_position.entry_time = datetime.now(UTC)
        # Required (nullable=False) columns are always present.
        mock_position.entry_price = Decimal("2500.0")
        mock_position.size = Decimal("0.25")
        # Nullable numeric columns are NULL in the DB.
        mock_position.current_price = None
        mock_position.quantity = None
        mock_position.entry_balance = None
        mock_position.stop_loss = None
        mock_position.take_profit = None
        mock_position.trailing_stop_price = None
        mock_position.mfe_price = None
        mock_position.mae_price = None
        mock_position.original_size = None
        mock_position.current_size = None
        mock_position.last_partial_exit_price = None
        mock_position.last_scale_in_price = None
        # default=0.0 columns are NULL (the default should apply).
        mock_position.unrealized_pnl = None
        mock_position.unrealized_pnl_percent = None
        mock_position.mfe = None
        mock_position.mae = None

        def mock_query_side_effect(model):
            if model.__name__ == "Position":
                q = Mock()
                q.filter.return_value.all.return_value = [mock_position]
                return q
            if model.__name__ == "Order":
                q = Mock()
                q.filter.return_value.order_by.return_value.all.return_value = []
                return q
            return Mock()

        mock_postgresql_db._mock_session.query.side_effect = mock_query_side_effect

        position = mock_postgresql_db.get_active_positions()[0]

        # Required fields still coerce to float.
        assert isinstance(position["entry_price"], float)
        assert isinstance(position["size"], float)

        # Nullable fields surface as None (not Decimal, not a crash).
        nullable_fields = [
            "current_price",
            "quantity",
            "entry_balance",
            "stop_loss",
            "take_profit",
            "trailing_stop_price",
            "mfe_price",
            "mae_price",
            "original_size",
            "current_size",
            "last_partial_exit_price",
            "last_scale_in_price",
        ]
        for field in nullable_fields:
            assert position[field] is None, f"{field} should be None, got {position[field]!r}"

        # default=0.0 columns fall back to 0.0 (float), never None.
        for field in ["unrealized_pnl", "unrealized_pnl_percent", "mfe", "mae"]:
            assert position[field] == 0.0
            assert isinstance(position[field], float)

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
        mock_trade.entry_time = datetime.now(UTC)
        mock_trade.exit_time = datetime.now(UTC)
        mock_trade.exit_reason = "Take profit"
        mock_trade.strategy_name = "TestStrategy"

        mock_query = Mock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_trade]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        trades = mock_postgresql_db.get_recent_trades(limit=10)

        assert isinstance(trades, list)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTCUSDT"

    def test_get_recent_trades_coerces_numeric_fields_to_float(self, mock_postgresql_db):
        """Numeric columns come back from PostgreSQL as Decimal; get_recent_trades
        must coerce them to float so dashboard consumers stay JSON-serializable
        (json.dumps raises on Decimal) and any downstream float arithmetic is safe.
        Mirrors the coercion applied in get_active_positions.
        """
        # Mirror real SQLAlchemy Numeric(18, 8) reads: every numeric field is a Decimal.
        mock_trade = Mock()
        mock_trade.id = 1
        mock_trade.symbol = "BTCUSDT"
        mock_trade.side.value = "LONG"
        mock_trade.entry_price = Decimal("45000.12345678")
        mock_trade.exit_price = Decimal("46000.0")
        mock_trade.size = Decimal("0.10000000")
        mock_trade.pnl = Decimal("100.0")
        mock_trade.pnl_percent = Decimal("2.22")
        mock_trade.mfe = Decimal("0.03")
        mock_trade.mae = Decimal("0.01")
        mock_trade.mfe_price = Decimal("46500.0")
        mock_trade.mae_price = Decimal("44500.0")
        mock_trade.entry_time = datetime.now(UTC)
        mock_trade.exit_time = datetime.now(UTC)
        mock_trade.exit_reason = "Take profit"
        mock_trade.strategy_name = "TestStrategy"

        mock_query = Mock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_trade]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        trades = mock_postgresql_db.get_recent_trades(limit=10)
        trade = trades[0]

        numeric_fields = [
            "entry_price",
            "exit_price",
            "size",
            "pnl",
            "pnl_percent",
            "mfe",
            "mae",
            "mfe_price",
            "mae_price",
        ]
        for field in numeric_fields:
            value = trade[field]
            assert not isinstance(value, Decimal), f"{field} leaked a Decimal: {value!r}"
            assert isinstance(value, float), f"{field} should be float, got {type(value).__name__}"

        # Values must be preserved through coercion, not just type-changed.
        assert trade["entry_price"] == pytest.approx(45000.12345678)
        assert trade["pnl"] == pytest.approx(100.0)
        assert trade["mae_price"] == pytest.approx(44500.0)

    def test_get_recent_trades_null_numeric_fields_use_defaults(self, mock_postgresql_db):
        """Nullable trade numeric columns surface as None; default=0.0 columns as 0.0."""
        mock_trade = Mock()
        mock_trade.id = 9
        mock_trade.symbol = "ETHUSDT"
        mock_trade.side.value = "LONG"
        mock_trade.strategy_name = "TestStrategy"
        mock_trade.entry_time = datetime.now(UTC)
        mock_trade.exit_time = datetime.now(UTC)
        mock_trade.exit_reason = "Stop loss"
        # Required (nullable=False) columns are always present.
        mock_trade.entry_price = Decimal("2500.0")
        mock_trade.exit_price = Decimal("2400.0")
        mock_trade.size = Decimal("0.25")
        mock_trade.pnl = Decimal("-25.0")
        mock_trade.pnl_percent = Decimal("-1.0")
        # Nullable numeric columns are NULL.
        mock_trade.mfe_price = None
        mock_trade.mae_price = None
        # default=0.0 columns are NULL (the default should apply).
        mock_trade.mfe = None
        mock_trade.mae = None

        mock_query = Mock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_trade]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        trade = mock_postgresql_db.get_recent_trades(limit=5)[0]

        for field in ["entry_price", "exit_price", "size", "pnl", "pnl_percent"]:
            assert isinstance(trade[field], float)
        assert trade["mfe_price"] is None
        assert trade["mae_price"] is None
        for field in ["mfe", "mae"]:
            assert trade[field] == 0.0
            assert isinstance(trade[field], float)

    def test_get_performance_metrics(self, mock_postgresql_db):
        """Test getting performance metrics"""
        mock_trade = Mock()
        mock_trade.pnl = 100.0
        mock_trade.exit_time = datetime.now(UTC)
        mock_trade.entry_time = datetime.now(UTC) - timedelta(hours=1)

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
