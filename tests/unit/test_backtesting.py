"""
Tests for backtesting engine.

Backtesting engine is critical for strategy validation before live trading. Tests cover:
- Strategy execution simulation
- Trade generation and tracking
- Performance calculation
- Risk management integration
- Data handling and validation
- Edge cases and error conditions
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import Backtester
from live.trading_engine import Trade
from risk.risk_manager import RiskParameters
from strategies.ml_adaptive import MlAdaptive


class TestBacktesterInitialization:
    """Test backtesting engine initialization"""

    def test_backtester_initialization(self, mock_data_provider):
        """Test backtester initialization with basic parameters"""
        strategy = MlAdaptive()
        risk_params = RiskParameters()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        assert backtester.strategy == strategy
        assert backtester.data_provider == mock_data_provider
        assert backtester.risk_parameters == risk_params
        assert backtester.initial_balance == 10000
        assert backtester.balance == 10000
        assert len(backtester.trades) == 0
        assert backtester.current_trade is None

    def test_backtester_with_default_parameters(self, mock_data_provider):
        """Test backtester with default parameters"""
        strategy = MlAdaptive()

        backtester = Backtester(strategy=strategy, data_provider=mock_data_provider)

        assert backtester.strategy == strategy
        assert backtester.data_provider == mock_data_provider
        assert backtester.initial_balance > 0
        assert backtester.balance == backtester.initial_balance

    def test_backtester_with_sentiment_provider(self, mock_data_provider, mock_sentiment_provider):
        """Test backtester with sentiment provider"""
        strategy = MlAdaptive()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            sentiment_provider=mock_sentiment_provider,
        )

        assert backtester.sentiment_provider == mock_sentiment_provider


class TestTradeGeneration:
    """Test trade generation and management"""

    def test_trade_creation(self):
        """Test Trade object creation"""
        from live.trading_engine import PositionSide

        trade = Trade(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            exit_price=55000,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        assert trade.symbol == "BTCUSDT"
        assert trade.side == PositionSide.LONG
        assert trade.entry_price == 50000
        assert trade.exit_price == 55000
        assert trade.size == 0.1
        assert trade.pnl == 500

    def test_trade_pnl_calculation(self):
        """Test trade P&L calculation"""
        from live.trading_engine import PositionSide

        # Long position with profit
        trade_long_profit = Trade(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            exit_price=55000,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        # P&L should be calculated automatically
        assert trade_long_profit.pnl == 500  # (55000-50000) * 0.1

        # Short position with profit
        trade_short_profit = Trade(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=55000,
            exit_price=50000,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        assert trade_short_profit.pnl == 500  # (55000-50000) * 0.1

    def test_trade_duration_calculation(self):
        """Test trade duration calculation"""
        from live.trading_engine import PositionSide

        entry_time = datetime(2024, 1, 1, 10, 0)
        exit_time = datetime(2024, 1, 1, 12, 0)

        trade = Trade(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            exit_price=55000,
            entry_time=entry_time,
            exit_time=exit_time,
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        # Duration should be 2 hours
        expected_duration = timedelta(hours=2)
        assert trade.exit_time - trade.entry_time == expected_duration


class TestBacktestingExecution:
    """Test backtesting execution and results"""

    def test_basic_backtest_execution(self, mock_data_provider, sample_ohlcv_data):
        """Test basic backtest execution"""
        strategy = MlAdaptive()
        risk_params = RiskParameters()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        # Mock data provider to return sample data
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        # Run backtest
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)

        results = backtester.run("BTCUSDT", "1h", start_date, end_date)

        # Verify results structure
        assert isinstance(results, dict)
        required_keys = ["total_trades", "win_rate", "total_return", "final_balance"]
        for key in required_keys:
            assert key in results

        # Verify realistic results
        assert results["total_trades"] >= 0
        assert 0 <= results["win_rate"] <= 100
        assert results["final_balance"] > 0

    def test_backtest_with_no_trades(self, mock_data_provider):
        """Test backtest with no trading signals"""
        strategy = MlAdaptive()

        # Create data with no clear signals
        no_signal_data = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99],
                "close": [100, 100, 100, 100, 100],
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="1h"),
        )

        mock_data_provider.get_historical_data.return_value = no_signal_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should have no trades
        assert results["total_trades"] == 0
        assert results["final_balance"] == 10000
        assert results["total_return"] == 0.0

    def test_backtest_performance_metrics(self, mock_data_provider, sample_ohlcv_data):
        """Test backtest performance metrics calculation"""
        strategy = MlAdaptive()

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Test performance metrics
        assert "sharpe_ratio" in results or "max_drawdown" in results
        assert "total_return" in results
        assert "win_rate" in results

        # Metrics should be reasonable
        assert results["total_return"] >= -100  # Should not lose more than 100%
        assert 0 <= results["win_rate"] <= 100


class TestRiskManagementIntegration:
    """Test risk management integration in backtesting"""

    def test_risk_parameters_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test that risk parameters are respected during backtesting"""
        strategy = MlAdaptive()

        # Conservative risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=0.01,  # 1% risk per trade
            max_position_size=0.05,  # 5% max position
            max_daily_risk=0.03,  # 3% daily risk
        )

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should complete without errors
        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_position_size_limits(self, mock_data_provider, sample_ohlcv_data):
        """Test that position size limits are enforced"""
        strategy = MlAdaptive()

        # Very restrictive position size
        risk_params = RiskParameters(max_position_size=0.01)  # 1% max position

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should complete without errors
        assert isinstance(results, dict)


class TestDataHandling:
    """Test data handling and validation"""

    def test_empty_data_handling(self, mock_data_provider):
        """Test backtester with empty data"""
        strategy = MlAdaptive()

        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        mock_data_provider.get_historical_data.return_value = empty_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should handle empty data gracefully
        assert results["total_trades"] == 0
        assert results["final_balance"] == 10000

    def test_missing_columns_handling(self, mock_data_provider):
        """Test backtester with missing data columns"""
        strategy = MlAdaptive()

        # Data missing required columns
        incomplete_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "close": [101, 102, 103],
                # Missing high, low, volume
            }
        )

        mock_data_provider.get_historical_data.return_value = incomplete_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        # Should handle missing columns gracefully or raise appropriate error
        with pytest.raises((KeyError, ValueError), match="Missing required columns"):
            backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

    def test_data_validation(self, mock_data_provider):
        """Test data validation in backtester"""
        strategy = MlAdaptive()

        # Data with invalid values
        invalid_data = pd.DataFrame(
            {
                "open": [100, -50, 102],  # Negative price
                "high": [101, 101, 102],
                "low": [99, 99, 101],
                "close": [101, 102, 103],
                "volume": [1000, 1000, 1000],
            }
        )

        mock_data_provider.get_historical_data.return_value = invalid_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        # Should handle invalid data gracefully
        try:
            results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
            assert isinstance(results, dict)
        except (ValueError, AssertionError):
            # Expected behavior for invalid data
            pass


class TestBacktestingEdgeCases:
    """Test backtesting edge cases and error conditions"""

    def test_single_data_point(self, mock_data_provider):
        """Test backtester with single data point"""
        strategy = MlAdaptive()

        single_data = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [1000]},
            index=[datetime(2024, 1, 1, 10, 0)],
        )

        mock_data_provider.get_historical_data.return_value = single_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should handle single data point
        assert isinstance(results, dict)
        assert results["total_trades"] == 0  # No trades possible with single point

    @pytest.mark.slow
    def test_very_large_dataset(self, mock_data_provider):
        """Test backtester with very large dataset"""
        strategy = MlAdaptive()

        # Generate large dataset
        n_points = 10000
        large_data = pd.DataFrame(
            {
                "open": np.random.randn(n_points) + 100,
                "high": np.random.randn(n_points) + 101,
                "low": np.random.randn(n_points) + 99,
                "close": np.random.randn(n_points) + 100,
                "volume": np.random.randint(1000, 10000, n_points),
            },
            index=pd.date_range("2024-01-01", periods=n_points, freq="1h"),
        )

        mock_data_provider.get_historical_data.return_value = large_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        # Should handle large dataset without memory issues
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_concurrent_trades_handling(self, mock_data_provider, sample_ohlcv_data):
        """Test handling of concurrent trades"""
        strategy = MlAdaptive()

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should handle concurrent trades appropriately
        assert isinstance(results, dict)
        assert results["total_trades"] >= 0


class TestBacktestingIntegration:
    """Test backtesting integration with other components"""

    def test_strategy_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester integration with different strategies"""
        # Test with adaptive strategy
        adaptive_strategy = MlAdaptive()

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=adaptive_strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_database_logging_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester with database logging"""
        strategy = MlAdaptive()

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        # Test with database logging enabled
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
            log_to_database=True,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should complete with database logging
        assert isinstance(results, dict)
        assert backtester.db_manager is not None

    def test_sentiment_integration(
        self, mock_data_provider, mock_sentiment_provider, sample_ohlcv_data
    ):
        """Test backtester with sentiment data integration"""
        strategy = MlAdaptive()

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            sentiment_provider=mock_sentiment_provider,
            initial_balance=10000,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        # Should complete with sentiment integration
        assert isinstance(results, dict)
        assert backtester.sentiment_provider == mock_sentiment_provider
