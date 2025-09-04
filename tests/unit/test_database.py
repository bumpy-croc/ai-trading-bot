#!/usr/bin/env python3
"""
Comprehensive unit tests for DatabaseManager
Tests all methods with PostgreSQL mock and real connections
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src")
)

from src.database.manager import DatabaseManager
from src.database.models import OrderStatus, PositionSide, PositionStatus

pytestmark = pytest.mark.unit


class TestDatabaseManager:
    """Comprehensive test suite for DatabaseManager"""

    @pytest.fixture
    def mock_postgresql_db(self):
        """Create DatabaseManager with mocked PostgreSQL for testing"""
        postgresql_url = "postgresql://test_user:test_pass@localhost:5432/test_db"

        with (
            patch("database.manager.create_engine") as mock_create_engine,
            patch("database.manager.sessionmaker") as mock_sessionmaker,
            patch("database.manager.Base"),
        ):
            # Setup mocks
            mock_engine = Mock()
            mock_session_factory = Mock()
            mock_session = Mock()

            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory
            mock_session_factory.return_value = mock_session

            # Mock engine connection
            mock_connection = Mock()
            mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
            mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

            # Mock connection pool
            mock_engine.pool = Mock()
            mock_engine.pool.size = 5
            mock_engine.pool.checkedin = 2
            mock_engine.pool.checkedout = 3
            mock_engine.pool.overflow = 1
            mock_engine.pool.invalid = 0
            mock_engine.pool.status.return_value = "Pool status"
            mock_engine.pool.dispose = Mock()

            # Mock session behavior
            mock_session.execute.return_value = None
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            mock_session.rollback.return_value = None
            mock_session.close.return_value = None
            mock_session.query.return_value = Mock()
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)

            # Create database manager
            db_manager = DatabaseManager(database_url=postgresql_url)
            db_manager._mock_engine = mock_engine
            db_manager._mock_session = mock_session

            yield db_manager


class TestInitialization(TestDatabaseManager):
    """Test DatabaseManager initialization"""

    def test_init_with_postgresql_url(self):
        """Test initialization with valid PostgreSQL URL"""
        postgresql_url = "postgresql://test:test@localhost:5432/test"

        with (
            patch("database.manager.create_engine"),
            patch("database.manager.sessionmaker"),
            patch("database.manager.Base.metadata.create_all"),
        ):
            db_manager = DatabaseManager(database_url=postgresql_url)
            assert db_manager.database_url == postgresql_url

    def test_init_from_environment(self):
        """Test initialization from DATABASE_URL environment variable"""
        postgresql_url = "postgresql://test:test@localhost:5432/test"

        with (
            patch("database.manager.get_config") as mock_config,
            patch("database.manager.create_engine"),
            patch("database.manager.sessionmaker"),
            patch("database.manager.Base.metadata.create_all"),
        ):
            mock_config.return_value.get.return_value = postgresql_url

            db_manager = DatabaseManager()
            assert db_manager.database_url == postgresql_url

    def test_init_fails_without_database_url(self):
        """Test initialization fails without DATABASE_URL"""
        with patch("database.manager.get_config") as mock_config:
            mock_config.return_value.get.return_value = None

            with pytest.raises(ValueError, match="DATABASE_URL environment variable is required"):
                DatabaseManager()

    def test_init_fails_with_non_postgresql_url(self):
        """Test initialization fails with non-PostgreSQL URL"""
        sqlite_url = "mysql://user:pass@localhost:3306/testdb"

        with pytest.raises(ValueError, match="Only PostgreSQL databases are supported"):
            DatabaseManager(database_url=sqlite_url)


class TestConnectionMethods(TestDatabaseManager):
    """Test connection-related methods"""

    def test_test_connection_success(self, mock_postgresql_db):
        """Test successful database connection test"""
        result = mock_postgresql_db.test_connection()
        assert result is True

    def test_test_connection_failure(self, mock_postgresql_db):
        """Test database connection test failure"""
        mock_postgresql_db._mock_session.execute.side_effect = Exception("Connection failed")

        result = mock_postgresql_db.test_connection()
        assert result is False

    def test_get_database_info(self, mock_postgresql_db):
        """Test get_database_info method"""
        info = mock_postgresql_db.get_database_info()

        assert isinstance(info, dict)
        assert "database_url" in info
        assert "database_type" in info
        assert "connection_pool_size" in info
        assert "checked_in_connections" in info
        assert "checked_out_connections" in info

        assert info["database_type"] == "postgresql"
        assert info["connection_pool_size"] == 5

    def test_get_connection_stats(self, mock_postgresql_db):
        """Test get_connection_stats method"""
        stats = mock_postgresql_db.get_connection_stats()

        assert isinstance(stats, dict)
        assert "pool_status" in stats
        assert "checked_in" in stats
        assert "checked_out" in stats
        assert "overflow" in stats
        assert "invalid" in stats

    def test_cleanup_connection_pool(self, mock_postgresql_db):
        """Test cleanup_connection_pool method"""
        mock_postgresql_db.cleanup_connection_pool()
        mock_postgresql_db._mock_engine.pool.dispose.assert_called_once()


class TestSessionManagement(TestDatabaseManager):
    """Test session management methods"""

    def test_create_trading_session(self, mock_postgresql_db):
        """Test creating a trading session"""
        # Mock the session object that gets added
        mock_session_obj = Mock()
        mock_session_obj.id = 123
        mock_postgresql_db._mock_session.add.return_value = None

        # Need to mock the session creation process
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
        # Mock trading session
        mock_session_obj = Mock()
        mock_session_obj.id = 123
        mock_session_obj.session_name = "test_session"
        mock_session_obj.start_time = datetime.utcnow()

        # Mock query results
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_session_obj
        mock_query.filter_by.return_value.all.return_value = []
        mock_postgresql_db._mock_session.query.return_value = mock_query

        mock_postgresql_db._current_session_id = 123
        mock_postgresql_db.end_trading_session()

        mock_postgresql_db._mock_session.commit.assert_called()


class TestTradeLogging(TestDatabaseManager):
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
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow(),
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
            position_id=789, current_price=46000.0, unrealized_pnl=100.0
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


class TestEventLogging(TestDatabaseManager):
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


class TestDataRetrieval(TestDatabaseManager):
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

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_position]
        mock_postgresql_db._mock_session.query.return_value = mock_query

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

        # Use the second get_performance_metrics method (which requires session_id)
        metrics = mock_postgresql_db.get_performance_metrics(session_id=123)

        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        # The method returns: total_trades, winning_trades, losing_trades, win_rate, total_pnl, avg_win, avg_loss, profit_factor, max_drawdown, best_trade, worst_trade, avg_trade_duration
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics


class TestUtilityMethods(TestDatabaseManager):
    """Test utility methods"""

    def test_cleanup_old_data(self, mock_postgresql_db):
        """Test cleaning up old data"""
        mock_session = Mock()
        mock_session.end_time = datetime.utcnow() - timedelta(days=100)
        mock_session.is_active = False

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_session]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        mock_postgresql_db.cleanup_old_data(days_to_keep=90)

        mock_postgresql_db._mock_session.delete.assert_called()
        mock_postgresql_db._mock_session.commit.assert_called()

    def test_execute_query(self, mock_postgresql_db):
        """Test executing raw SQL queries"""
        mock_result = Mock()
        mock_row = {"id": 1, "name": "test"}
        mock_result.mappings.return_value = [mock_row]

        mock_connection = Mock()
        mock_connection.exec_driver_sql.return_value = mock_result
        mock_postgresql_db._mock_engine.connect.return_value.__enter__.return_value = (
            mock_connection
        )

        result = mock_postgresql_db.execute_query("SELECT * FROM test")

        assert isinstance(result, list)


class TestErrorHandling(TestDatabaseManager):
    """Test error handling"""

    def test_session_rollback_on_error(self, mock_postgresql_db):
        """Test session rollback on error"""
        mock_postgresql_db._mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception):  # noqa: B017
            with mock_postgresql_db.get_session() as session:
                session.execute("SELECT 1")

        mock_postgresql_db._mock_session.rollback.assert_called()

    def test_close_position_not_found(self, mock_postgresql_db):
        """Test closing non-existent position"""
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = None
        mock_postgresql_db._mock_session.query.return_value = mock_query

        result = mock_postgresql_db.close_position(999)
        assert result is False


class TestEnumConversion(TestDatabaseManager):
    """Test enum string conversion"""

    def test_trade_side_string_conversion(self, mock_postgresql_db):
        """Test converting side string to enum"""
        mock_trade_obj = Mock()
        mock_trade_obj.id = 456

        with patch("database.manager.Trade") as mock_trade_class:
            mock_trade_class.return_value = mock_trade_obj

            trade_id = mock_postgresql_db.log_trade(
                symbol="BTCUSDT",
                side="LONG",  # String instead of enum
                entry_price=45000.0,
                exit_price=46000.0,
                size=0.1,
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow(),
                pnl=100.0,
                exit_reason="Take profit",
                strategy_name="TestStrategy",
            )

            assert trade_id == 456

    def test_event_type_string_conversion(self, mock_postgresql_db):
        """Test converting event type string to enum"""
        mock_event_obj = Mock()
        mock_event_obj.id = 101

        with patch("database.manager.SystemEvent") as mock_event_class:
            mock_event_class.return_value = mock_event_obj

            event_id = mock_postgresql_db.log_event(
                event_type="TEST",  # String instead of enum
                message="Test event message",
            )

            assert event_id == 101


class TestOrderStatusNormalization(TestDatabaseManager):
    """Tests for order status normalization in DatabaseManager.update_order_status"""

    def test_update_order_status_normalizes_values(self, mock_postgresql_db):
        mock_position = Mock()
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_position
        mock_postgresql_db._mock_session.query.return_value = mock_query

        # Mixed casing and exchange variants map correctly
        assert mock_postgresql_db.update_order_status(1, "open") is True
        assert mock_position.status.value == OrderStatus.OPEN.value

        assert mock_postgresql_db.update_order_status(1, "Partially_Filled") is True
        assert mock_position.status.value == OrderStatus.FILLED.value

        assert mock_postgresql_db.update_order_status(1, "rejected") is True
        assert mock_position.status.value == OrderStatus.FAILED.value

        assert mock_postgresql_db.update_order_status(1, "expired") is True
        assert mock_position.status.value == OrderStatus.CANCELLED.value

    def test_update_order_status_invalid_value(self, mock_postgresql_db):
        mock_position = Mock()
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_position
        mock_postgresql_db._mock_session.query.return_value = mock_query

        assert mock_postgresql_db.update_order_status(1, "unknown_status") is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
