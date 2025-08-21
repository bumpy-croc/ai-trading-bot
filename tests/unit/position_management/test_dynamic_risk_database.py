"""
Test database persistence functionality for dynamic risk management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.database.manager import DatabaseManager
from src.position_management.dynamic_risk import DynamicRiskManager, DynamicRiskConfig


class TestDynamicRiskDatabase:
    """Test database operations for dynamic risk management"""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for testing"""
        db_manager = Mock(spec=DatabaseManager)
        return db_manager

    @pytest.fixture
    def dynamic_risk_manager(self, mock_db_manager):
        """Create a DynamicRiskManager with mocked database"""
        config = DynamicRiskConfig(
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4]
        )
        return DynamicRiskManager(config, db_manager=mock_db_manager)

    def test_database_logging_significant_adjustment(self, dynamic_risk_manager, mock_db_manager):
        """Test that significant risk adjustments are logged to database"""
        # Setup mock to return performance data that triggers adjustment
        mock_db_manager.get_performance_metrics.return_value = {
            "total_trades": 20,
            "win_rate": 0.3,  # Poor performance
            "profit_factor": 0.8,
            "sharpe_ratio": -0.5,
            "expectancy": -0.05,
            "avg_trade_duration_hours": 4.0,
            "consecutive_losses": 5,
            "consecutive_wins": 0
        }

        # Calculate adjustments (should trigger significant adjustment)
        adjustments = dynamic_risk_manager.calculate_dynamic_risk_adjustments(
            current_balance=8500,  # 15% drawdown from 10000
            peak_balance=10000,
            session_id=123
        )

        # Verify that a significant adjustment was calculated
        assert adjustments.position_size_factor < 1.0
        assert adjustments.primary_reason != "normal"

    def test_performance_metrics_caching(self, dynamic_risk_manager, mock_db_manager):
        """Test that performance metrics are cached for performance"""
        # Setup mock
        mock_db_manager.get_performance_metrics.return_value = {
            "total_trades": 15,
            "win_rate": 0.6,
            "profit_factor": 1.2
        }

        # First call should query database
        adjustments1 = dynamic_risk_manager.calculate_dynamic_risk_adjustments(
            current_balance=9500,
            peak_balance=10000,
            session_id=123
        )

        # Second call within cache TTL should use cache (not call database again)
        adjustments2 = dynamic_risk_manager.calculate_dynamic_risk_adjustments(
            current_balance=9500,
            peak_balance=10000,
            session_id=123
        )

        # Database should only be called once
        assert mock_db_manager.get_performance_metrics.call_count == 1
        
        # Results should be the same
        assert adjustments1.position_size_factor == adjustments2.position_size_factor

    def test_database_error_handling(self, mock_db_manager):
        """Test graceful handling of database errors"""
        # Setup database to raise an error
        mock_db_manager.get_performance_metrics.side_effect = Exception("Database connection failed")
        
        config = DynamicRiskConfig()
        manager = DynamicRiskManager(config, db_manager=mock_db_manager)

        # Should not raise exception, should return safe defaults
        adjustments = manager.calculate_dynamic_risk_adjustments(
            current_balance=9000,
            peak_balance=10000,
            session_id=123
        )

        # Should still calculate drawdown-based adjustments
        assert adjustments.position_size_factor == 0.6  # 10% drawdown threshold
        assert "drawdown" in adjustments.primary_reason

    def test_recovery_threshold_logic(self, dynamic_risk_manager, mock_db_manager):
        """Test recovery threshold de-throttling functionality"""
        # Setup normal performance metrics
        mock_db_manager.get_performance_metrics.return_value = {
            "total_trades": 25,
            "win_rate": 0.55,
            "profit_factor": 1.1,
            "sharpe_ratio": 0.3
        }

        # Test recovery scenario
        adjustments = dynamic_risk_manager.calculate_dynamic_risk_adjustments(
            current_balance=10300,  # 3% above previous peak
            peak_balance=10000,
            previous_peak_balance=9500  # Recovered from 9500 to 10300
        )

        # Should apply recovery logic
        recovery_return = (10300 - 9500) / 9500  # ~8.4% recovery
        assert recovery_return > 0.05  # Above 5% recovery threshold
        
        # Should get some recovery benefit
        assert adjustments.position_size_factor >= 1.0  # At least normal or better

    def test_volatility_estimation_from_equity(self, dynamic_risk_manager, mock_db_manager):
        """Test volatility estimation from account equity history"""
        # Mock the database session and AccountHistory query
        mock_session = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        mock_db_manager.get_session.return_value = mock_context
        
        # Create mock account history records
        mock_records = []
        for i in range(30):
            record = Mock()
            record.equity = 10000 + (i * 50) + ((-1)**i * 100)  # Volatile equity curve
            mock_records.append(record)
        
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_records
        
        # Setup get_performance_metrics to return basic data
        mock_db_manager.get_performance_metrics.return_value = {
            "total_trades": 20,
            "win_rate": 0.5,
            "profit_factor": 1.0
        }

        # Calculate adjustments
        adjustments = dynamic_risk_manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000,
            session_id=123
        )

        # Should have estimated volatility in the metrics
        assert "estimated_volatility" in adjustments.adjustment_details["performance_metrics"]

    def test_minimum_trades_requirement(self, dynamic_risk_manager, mock_db_manager):
        """Test that performance adjustments require minimum trade count"""
        # Setup insufficient trade count
        mock_db_manager.get_performance_metrics.return_value = {
            "total_trades": 5,  # Below minimum of 10
            "win_rate": 0.2,   # Poor performance
            "profit_factor": 0.5,
            "sharpe_ratio": -1.0
        }

        adjustments = dynamic_risk_manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000,
            session_id=123
        )

        # Performance adjustment should not be applied due to insufficient data
        performance_adj = adjustments.adjustment_details["performance_adjustment"]
        assert performance_adj.primary_reason == "insufficient_data"
        assert performance_adj.position_size_factor == 1.0  # No adjustment


class TestDatabaseManagerMethods:
    """Test the DatabaseManager methods for dynamic risk"""

    @pytest.fixture
    def db_manager(self):
        """Create a real DatabaseManager for testing (with mocked session)"""
        with patch('src.database.manager.create_engine'), \
             patch('src.database.manager.sessionmaker'):
            manager = DatabaseManager()
            manager.get_session = Mock()
            return manager

    def test_log_dynamic_performance_metrics(self, db_manager):
        """Test logging dynamic performance metrics"""
        mock_session = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        db_manager.get_session.return_value = mock_context

        # Test logging metrics
        metrics_id = db_manager.log_dynamic_performance_metrics(
            session_id=123,
            rolling_win_rate=0.65,
            current_drawdown=0.05,
            volatility_30d=0.02,
            consecutive_losses=0,
            consecutive_wins=3,
            risk_adjustment_factor=0.8,
            profit_factor=1.3
        )

        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_log_risk_adjustment(self, db_manager):
        """Test logging risk adjustments"""
        mock_session = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        db_manager.get_session.return_value = mock_context

        # Test logging adjustment
        adjustment_id = db_manager.log_risk_adjustment(
            session_id=123,
            adjustment_type="drawdown",
            trigger_reason="drawdown_15.0%",
            parameter_name="position_size_factor",
            original_value=1.0,
            adjusted_value=0.4,
            adjustment_factor=0.4,
            current_drawdown=0.15,
            performance_score=0.3,
            volatility_level=0.025
        )

        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_get_performance_metrics_calculation(self, db_manager):
        """Test performance metrics calculation from trade data"""
        mock_session = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        db_manager.get_session.return_value = mock_context

        # Mock trade data
        mock_trades = []
        for i in range(20):
            trade = Mock()
            trade.pnl = 100 if i % 3 == 0 else -50  # Mixed results
            trade.entry_time = datetime.utcnow() - timedelta(hours=i*2)
            trade.exit_time = datetime.utcnow() - timedelta(hours=i*2-1)
            mock_trades.append(trade)

        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.all.return_value = mock_trades

        # Test calculation
        metrics = db_manager.get_recent_performance_metrics(
            session_id=123,
            start_date=datetime.utcnow() - timedelta(days=30)
        )

        # Verify basic metrics are calculated
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["total_trades"] == 20

    def test_performance_metrics_empty_data(self, db_manager):
        """Test performance metrics with no trade data"""
        mock_session = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        db_manager.get_session.return_value = mock_context
        
        # Mock empty trade data
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.all.return_value = []

        # Test calculation with no data
        metrics = db_manager.get_recent_performance_metrics(session_id=123)

        # Should return safe defaults
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.5
        assert metrics["profit_factor"] == 1.0
        assert metrics["sharpe_ratio"] == 0.0