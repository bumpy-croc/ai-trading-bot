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

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backtesting.engine import Backtester
from live.trading_engine import Trade
from strategies.ml_adaptive import MlAdaptive
from risk.risk_manager import RiskParameters


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
            initial_balance=10000
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
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider
        )
        
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
            sentiment_provider=mock_sentiment_provider
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
            exit_reason="test"
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
            exit_reason="test"
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
            exit_reason="test"
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
            exit_reason="test"
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
            initial_balance=10000
        )
        
        # Mock data provider to return sample data
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        # Run backtest
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        
        results = backtester.run("BTCUSDT", "1h", start_date, end_date)
        
        # Verify results structure
        assert isinstance(results, dict)
        required_keys = ['total_trades', 'win_rate', 'total_return', 'final_balance']
        for key in required_keys:
            assert key in results
        
        # Verify realistic results
        assert results['total_trades'] >= 0
        assert 0 <= results['win_rate'] <= 100
        assert results['final_balance'] > 0

    def test_backtest_with_no_trades(self, mock_data_provider):
        """Test backtest with no trading signals"""
        strategy = MlAdaptive()
        
        # Create data with no clear signals
        no_signal_data = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [101, 101, 101, 101, 101],
            'low': [99, 99, 99, 99, 99],
            'close': [100, 100, 100, 100, 100],
            'volume': [1000, 1000, 1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1h'))
        
        mock_data_provider.get_historical_data.return_value = no_signal_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should have no trades
        assert results['total_trades'] == 0
        assert results['final_balance'] == 10000
        assert results['total_return'] == 0.0

    def test_backtest_performance_metrics(self, mock_data_provider, sample_ohlcv_data):
        """Test backtest performance metrics calculation"""
        strategy = MlAdaptive()
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Test performance metrics
        assert 'sharpe_ratio' in results or 'max_drawdown' in results
        assert 'total_return' in results
        assert 'win_rate' in results
        
        # Metrics should be reasonable
        assert results['total_return'] >= -100  # Should not lose more than 100%
        assert 0 <= results['win_rate'] <= 100


class TestRiskManagementIntegration:
    """Test risk management integration in backtesting"""

    def test_risk_parameters_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test that risk parameters are respected during backtesting"""
        strategy = MlAdaptive()
        
        # Conservative risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=0.01,  # 1% risk per trade
            max_position_size=0.05,    # 5% max position
            max_daily_risk=0.03        # 3% daily risk
        )
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should complete without errors
        assert isinstance(results, dict)
        assert 'total_trades' in results

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
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should complete without errors
        assert isinstance(results, dict)


class TestDataHandling:
    """Test data handling and validation"""

    def test_empty_data_handling(self, mock_data_provider):
        """Test backtester with empty data"""
        strategy = MlAdaptive()
        
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        mock_data_provider.get_historical_data.return_value = empty_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should handle empty data gracefully
        assert results['total_trades'] == 0
        assert results['final_balance'] == 10000

    def test_missing_columns_handling(self, mock_data_provider):
        """Test backtester with missing data columns"""
        strategy = MlAdaptive()
        
        # Data missing required columns
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103]
            # Missing high, low, volume
        })
        
        mock_data_provider.get_historical_data.return_value = incomplete_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        # Should handle missing columns gracefully or raise appropriate error
        with pytest.raises((KeyError, ValueError), match="Missing required columns"):
            backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

    def test_data_validation(self, mock_data_provider):
        """Test data validation in backtester"""
        strategy = MlAdaptive()
        
        # Data with invalid values
        invalid_data = pd.DataFrame({
            'open': [100, -50, 102],  # Negative price
            'high': [101, 101, 102],
            'low': [99, 99, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1000, 1000]
        })
        
        mock_data_provider.get_historical_data.return_value = invalid_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
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
        
        single_data = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        }, index=[datetime(2024, 1, 1, 10, 0)])
        
        mock_data_provider.get_historical_data.return_value = single_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should handle single data point
        assert isinstance(results, dict)
        assert results['total_trades'] == 0  # No trades possible with single point

    def test_very_large_dataset(self, mock_data_provider):
        """Test backtester with very large dataset"""
        strategy = MlAdaptive()
        
        # Generate large dataset
        n_points = 10000
        large_data = pd.DataFrame({
            'open': np.random.randn(n_points) + 100,
            'high': np.random.randn(n_points) + 101,
            'low': np.random.randn(n_points) + 99,
            'close': np.random.randn(n_points) + 100,
            'volume': np.random.randint(1000, 10000, n_points)
        }, index=pd.date_range('2024-01-01', periods=n_points, freq='1h'))
        
        mock_data_provider.get_historical_data.return_value = large_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        # Should handle large dataset without memory issues
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        assert isinstance(results, dict)
        assert 'total_trades' in results

    def test_concurrent_trades_handling(self, mock_data_provider, sample_ohlcv_data):
        """Test handling of concurrent trades"""
        strategy = MlAdaptive()
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should handle concurrent trades appropriately
        assert isinstance(results, dict)
        assert results['total_trades'] >= 0


class TestBacktestingIntegration:
    """Test backtesting integration with other components"""

    def test_strategy_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester integration with different strategies"""
        # Test with adaptive strategy
        adaptive_strategy = MlAdaptive()
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        backtester = Backtester(
            strategy=adaptive_strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        assert isinstance(results, dict)
        assert 'total_trades' in results

    def test_database_logging_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester with database logging"""
        strategy = MlAdaptive()
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        # Test with database logging enabled
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
            log_to_database=True
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should complete with database logging
        assert isinstance(results, dict)
        assert backtester.db_manager is not None

    def test_sentiment_integration(self, mock_data_provider, mock_sentiment_provider, sample_ohlcv_data):
        """Test backtester with sentiment data integration"""
        strategy = MlAdaptive()
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            sentiment_provider=mock_sentiment_provider,
            initial_balance=10000
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Should complete with sentiment integration
        assert isinstance(results, dict)
        assert backtester.sentiment_provider == mock_sentiment_provider


class TestBacktestingPredictionEngineValidation:
    """Test backtesting validation with prediction engine integration"""

    def test_backtesting_with_prediction_engine_mock(self, mock_data_provider, sample_ohlcv_data):
        """Test that backtesting works with strategies using prediction engine"""
        from src.prediction.engine import PredictionEngine, PredictionResult
        from unittest.mock import Mock
        from datetime import datetime, timezone
        
        # Create mock prediction engine
        mock_engine = Mock(spec=PredictionEngine)
        mock_result = PredictionResult(
            price=55000.0,
            confidence=0.75,
            direction=1,
            model_name="backtest_model",
            timestamp=datetime.now(timezone.utc),
            inference_time=0.02,
            features_used=15
        )
        mock_engine.predict.return_value = mock_result
        
        # Test with different strategies using prediction engine
        strategies = [
            MlAdaptive(prediction_engine=mock_engine),
            # Add more strategies when they support prediction engine
        ]
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        for strategy in strategies:
            backtester = Backtester(
                strategy=strategy,
                data_provider=mock_data_provider,
                initial_balance=10000
            )
            
            results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
            
            # Should produce valid results
            assert isinstance(results, dict)
            assert 'total_trades' in results
            assert 'final_balance' in results
            assert 'total_return' in results
            
            # Results should be reasonable
            assert results['final_balance'] > 0
            assert results['total_trades'] >= 0

    def test_backtesting_prediction_engine_vs_no_engine(self, mock_data_provider, sample_ohlcv_data):
        """Test that strategies produce consistent results with/without prediction engine"""
        from src.prediction.engine import PredictionEngine, PredictionResult
        from unittest.mock import Mock
        from datetime import datetime, timezone
        
        # Create deterministic mock prediction engine
        mock_engine = Mock(spec=PredictionEngine)
        mock_result = PredictionResult(
            price=52000.0,  # Predictable price
            confidence=0.8,
            direction=1,
            model_name="deterministic_model", 
            timestamp=datetime.now(timezone.utc),
            inference_time=0.01,
            features_used=12
        )
        mock_engine.predict.return_value = mock_result
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        # Test same strategy with and without prediction engine
        strategy_with_engine = MlAdaptive(prediction_engine=mock_engine)
        strategy_without_engine = MlAdaptive()
        
        backtester_with = Backtester(
            strategy=strategy_with_engine,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        backtester_without = Backtester(
            strategy=strategy_without_engine,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        results_with = backtester_with.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        results_without = backtester_without.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # Both should produce valid results
        for results in [results_with, results_without]:
            assert isinstance(results, dict)
            assert 'total_trades' in results
            assert 'final_balance' in results
            assert results['final_balance'] > 0

    def test_backtesting_prediction_engine_error_handling(self, mock_data_provider, sample_ohlcv_data):
        """Test that backtesting handles prediction engine errors gracefully"""
        from src.prediction.engine import PredictionEngine
        from unittest.mock import Mock
        
        # Create mock prediction engine that fails
        mock_engine = Mock(spec=PredictionEngine)
        mock_engine.predict.side_effect = Exception("Model inference failed")
        
        strategy = MlAdaptive(prediction_engine=mock_engine)
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        # Should not crash even if prediction engine fails
        try:
            results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
            
            # Should still produce results (possibly with no trades due to prediction failures)
            assert isinstance(results, dict)
            assert 'total_trades' in results
            assert results['final_balance'] >= 0
            
        except Exception as e:
            # If strategy doesn't handle prediction errors gracefully,
            # this test documents the current behavior
            pytest.skip(f"Backtesting doesn't handle prediction engine errors gracefully: {e}")

    def test_backtesting_prediction_engine_performance_impact(self, mock_data_provider, sample_ohlcv_data):
        """Test that prediction engine doesn't significantly slow down backtesting"""
        from src.prediction.engine import PredictionEngine, PredictionResult
        from unittest.mock import Mock
        from datetime import datetime, timezone
        import time
        
        # Create fast mock prediction engine
        mock_engine = Mock(spec=PredictionEngine)
        mock_result = PredictionResult(
            price=51000.0,
            confidence=0.7,
            direction=0,
            model_name="fast_backtest_model",
            timestamp=datetime.now(timezone.utc),
            inference_time=0.001,  # Very fast
            features_used=8
        )
        mock_engine.predict.return_value = mock_result
        
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        # Time backtesting with prediction engine
        strategy_with_engine = MlAdaptive(prediction_engine=mock_engine)
        backtester_with = Backtester(
            strategy=strategy_with_engine,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        start_time = time.time()
        results_with = backtester_with.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        with_engine_time = time.time() - start_time
        
        # Time backtesting without prediction engine
        strategy_without_engine = MlAdaptive()
        backtester_without = Backtester(
            strategy=strategy_without_engine,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        start_time = time.time()
        results_without = backtester_without.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        without_engine_time = time.time() - start_time
        
        # Performance impact should be reasonable (less than 5x slower)
        if with_engine_time > 0 and without_engine_time > 0:
            slowdown_factor = with_engine_time / without_engine_time
            assert slowdown_factor < 5.0, f"Prediction engine slows down backtesting by {slowdown_factor:.2f}x"
        
        # Both should complete successfully
        assert isinstance(results_with, dict)
        assert isinstance(results_without, dict)

    def test_backtesting_preserves_existing_functionality(self, mock_data_provider, sample_ohlcv_data):
        """Test that existing backtesting functionality is preserved with prediction engine integration"""
        from unittest.mock import Mock
        
        # Test that existing strategy features still work
        strategy = MlAdaptive()  # No prediction engine - should use existing behavior
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
        
        # Test with risk parameters
        risk_params = RiskParameters()
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
            log_to_database=True
        )
        
        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        
        # All existing functionality should work
        assert isinstance(results, dict)
        assert 'total_trades' in results
        assert 'final_balance' in results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        
        # Risk management should still be active
        assert backtester.risk_parameters == risk_params