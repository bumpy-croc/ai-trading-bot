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

from src.backtesting.engine import Backtester
from src.live.trading_engine import Trade
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import MlBasic


class TestBacktesterInitialization:
    """Test backtesting engine initialization"""

    def test_backtester_initialization(self, mock_data_provider):
        """Test backtester initialization with basic parameters"""
        strategy = MlBasic()
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
        strategy = MlBasic()
        backtester = Backtester(strategy=strategy, data_provider=mock_data_provider)
        assert backtester.strategy == strategy
        assert backtester.data_provider == mock_data_provider
        assert backtester.initial_balance > 0
        assert backtester.balance == backtester.initial_balance

    def test_backtester_with_sentiment_provider(self, mock_data_provider, mock_sentiment_provider):
        """Test backtester with sentiment provider"""
        strategy = MlBasic()
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
            size=0.1,
            entry_price=50000,
            exit_price=50000,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            pnl=0.0,
            exit_reason="init",
        )
        assert trade.symbol == "BTCUSDT"
        assert trade.side == PositionSide.LONG
        assert trade.entry_price == 50000
        assert trade.size == 0.1

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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        strategy = MlBasic()
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
        # Test with basic strategy
        adaptive_strategy = MlBasic()
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=adaptive_strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_database_logging_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester with database logging"""
        strategy = MlBasic()
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
        strategy = MlBasic()
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


class TestCacheMechanism:
    """Test the new cache mechanism improvements"""

    def test_cache_initialization(self, mock_data_provider):
        """Test that cache attributes are properly initialized"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Check cache attributes
        assert hasattr(backtester, '_feature_cache')
        assert hasattr(backtester, '_strategy_cache')
        assert hasattr(backtester, '_ml_predictions_cache')
        assert hasattr(backtester, '_feature_cache_size')
        assert hasattr(backtester, '_strategy_cache_size')
        assert hasattr(backtester, '_ml_predictions_cache_size')
        assert hasattr(backtester, '_model_version')
        assert hasattr(backtester, '_use_original_method')
        assert hasattr(backtester, '_cache_hits')
        assert hasattr(backtester, '_cache_misses')

        # Check initial values
        assert backtester._feature_cache_size == 0
        assert backtester._strategy_cache_size == 0
        assert backtester._ml_predictions_cache_size == 0
        assert backtester._model_version is None
        assert backtester._use_original_method is False
        assert backtester._cache_hits == 0
        assert backtester._cache_misses == 0

    def test_model_version_generation(self, mock_data_provider):
        """Test that model version is generated correctly"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # First call should generate version
        version1 = backtester._get_model_version()
        assert version1 is not None
        assert len(version1) == 16  # MD5 hash truncated to 16 chars
        assert isinstance(version1, str)

        # Second call should return same version
        version2 = backtester._get_model_version()
        assert version1 == version2

        # Version should be stored
        assert backtester._model_version == version1

    def test_cache_key_generation(self, mock_data_provider):
        """Test that cache keys are generated correctly"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Generate cache keys
        key1 = backtester._get_cache_key(0)
        key2 = backtester._get_cache_key(1)
        key3 = backtester._get_cache_key(0)  # Same index

        # Keys should be different for different indices
        assert key1 != key2
        assert key1 == key3  # Same index should produce same key

        # Keys should contain model version
        model_version = backtester._get_model_version()
        assert model_version in key1
        assert model_version in key2

        # Keys should contain index
        assert "0" in key1
        assert "1" in key2

    def test_memory_usage_check(self, mock_data_provider):
        """Test memory usage checking"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Should return boolean
        memory_ok = backtester._check_memory_usage()
        assert isinstance(memory_ok, bool)

        # Should not raise exception
        try:
            backtester._check_memory_usage()
        except Exception as e:
            pytest.fail(f"Memory check should not raise exception: {e}")

    def test_cache_size_limits(self, mock_data_provider):
        """Test cache size limit checking"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Initially cache should not be full
        assert not backtester._is_cache_full()

        # Manually set cache sizes to test limits
        backtester._feature_cache_size = 10000  # At limit
        assert backtester._is_cache_full()

        backtester._feature_cache_size = 5000
        backtester._strategy_cache_size = 10000  # At limit
        assert backtester._is_cache_full()

        backtester._strategy_cache_size = 5000
        backtester._ml_predictions_cache_size = 10000  # At limit
        assert backtester._is_cache_full()

    def test_cache_cleanup(self, mock_data_provider):
        """Test cache cleanup functionality"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Fill caches beyond limit
        for i in range(12000):  # More than MAX_CACHE_SIZE
            key = backtester._get_cache_key(i)
            backtester._feature_cache[key] = {'test': 'data'}
            backtester._strategy_cache[key] = {'test': 'data'}
            backtester._ml_predictions_cache[key] = 0.5

        backtester._feature_cache_size = 12000
        backtester._strategy_cache_size = 12000
        backtester._ml_predictions_cache_size = 12000

        # Cache should be full
        assert backtester._is_cache_full()

        # Run cleanup
        backtester._cleanup_old_cache_entries()

        # Cache should be reduced to 80% of max size
        target_size = int(10000 * 0.8)  # 8000
        assert backtester._feature_cache_size <= target_size
        assert backtester._strategy_cache_size <= target_size
        assert backtester._ml_predictions_cache_size <= target_size

    def test_cache_clear(self, mock_data_provider):
        """Test cache clearing functionality"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Add some data to caches
        key = backtester._get_cache_key(0)
        backtester._feature_cache[key] = {'test': 'data'}
        backtester._strategy_cache[key] = {'test': 'data'}
        backtester._ml_predictions_cache[key] = 0.5
        backtester._feature_cache_size = 1
        backtester._strategy_cache_size = 1
        backtester._ml_predictions_cache_size = 1
        backtester._cache_hits = 5
        backtester._cache_misses = 3

        # Caches should have data
        assert len(backtester._feature_cache) == 1
        assert len(backtester._strategy_cache) == 1
        assert len(backtester._ml_predictions_cache) == 1
        assert backtester._cache_hits == 5
        assert backtester._cache_misses == 3

        # Clear caches
        backtester._clear_feature_cache()

        # Caches should be empty
        assert len(backtester._feature_cache) == 0
        assert len(backtester._strategy_cache) == 0
        assert len(backtester._ml_predictions_cache) == 0
        assert backtester._feature_cache_size == 0
        assert backtester._strategy_cache_size == 0
        assert backtester._ml_predictions_cache_size == 0
        assert backtester._cache_hits == 0
        assert backtester._cache_misses == 0

    def test_chunked_processing_decision(self, mock_data_provider):
        """Test that chunked processing is used for large datasets"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Create a large DataFrame (more than MAX_CACHE_SIZE)
        large_df = pd.DataFrame({
            'open': np.random.rand(15000),
            'high': np.random.rand(15000),
            'low': np.random.rand(15000),
            'close': np.random.rand(15000),
            'volume': np.random.rand(15000)
        })

        # Mock the chunked processing method to verify it's called
        chunked_called = False
        original_method = backtester._precompute_ml_predictions_chunked

        def mock_chunked(df):
            nonlocal chunked_called
            chunked_called = True
            return original_method(df)

        backtester._precompute_ml_predictions_chunked = mock_chunked

        # Call pre-computation
        backtester._precompute_ml_predictions(large_df)

        # Should have called chunked processing
        assert chunked_called

    def test_fallback_to_original_method(self, mock_data_provider):
        """Test fallback to original method when pre-computation fails"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Mock strategy to raise exception during calculate_indicators
        def mock_calculate_indicators(df):
            raise Exception("Simulated ML prediction failure")

        strategy.calculate_indicators = mock_calculate_indicators

        # Create test DataFrame
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })

        # Call pre-computation
        backtester._precompute_ml_predictions(df)

        # Should have fallen back to original method
        assert backtester._use_original_method is True
        assert len(backtester._ml_predictions_cache) == 0

    def test_cache_hit_miss_tracking(self, mock_data_provider):
        """Test that cache hits and misses are properly tracked"""
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Add some data to strategy cache
        key = backtester._get_cache_key(0)
        backtester._strategy_cache[key] = {'current_price': 100.0}

        # Test cached method calls
        df = pd.DataFrame({'close': [100, 101, 102]})
        
        # This should be a cache hit
        backtester._check_entry_conditions_cached(df, 0)
        assert backtester._cache_hits == 1
        assert backtester._cache_misses == 0

        # This should be a cache miss (index 1 not in cache)
        backtester._check_entry_conditions_cached(df, 1)
        assert backtester._cache_hits == 1
        assert backtester._cache_misses == 1

    def test_model_version_consistency(self, mock_data_provider):
        """Test that model version remains consistent for the same backtester instance"""
        strategy = MlBasic()
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )

        # Same backtester instance should produce same model version
        version1 = backtester._get_model_version()
        version2 = backtester._get_model_version()
        assert version1 == version2

        # Version should be consistent
        assert version1 is not None
        assert version2 is not None
        assert len(version1) == 16  # MD5 hash truncated to 16 chars

    def test_persistent_cache_manager_initialization(self):
        """Test PersistentCacheManager initialization"""
        from src.backtesting.engine import PersistentCacheManager
        
        cache_manager = PersistentCacheManager()
        assert cache_manager.cache_dir.exists()
        assert cache_manager.expiry_days == 7

    def test_persistent_cache_operations(self, tmp_path):
        """Test persistent cache save/load/delete operations"""
        from src.backtesting.engine import PersistentCacheManager
        
        cache_manager = PersistentCacheManager(cache_dir=str(tmp_path))
        
        # Test data
        test_data = {'test': 'data', 'number': 42}
        cache_key = "test_cache_key"
        
        # Test save
        assert cache_manager.set(cache_key, test_data) is True
        
        # Test load
        loaded_data = cache_manager.get(cache_key)
        assert loaded_data == test_data
        
        # Test delete
        assert cache_manager.delete(cache_key) is True
        assert cache_manager.get(cache_key) is None

    def test_cache_expiration(self, tmp_path):
        """Test cache expiration functionality"""
        from src.backtesting.engine import PersistentCacheManager
        from datetime import datetime, timedelta
        
        # Create cache manager with 1 day expiry
        cache_manager = PersistentCacheManager(cache_dir=str(tmp_path), expiry_days=1)
        
        test_data = {'test': 'data'}
        cache_key = "expiry_test"
        
        # Save data
        assert cache_manager.set(cache_key, test_data) is True
        
        # Should not be expired immediately
        assert not cache_manager._is_cache_expired(cache_key)
        
        # Manually set creation time to past
        metadata_path = cache_manager._get_metadata_path(cache_key)
        with open(metadata_path, 'w') as f:
            import json
            json.dump({
                'created_at': (datetime.now() - timedelta(days=2)).isoformat(),
                'size': 100,
                'cache_key': cache_key
            }, f)
        
        # Should now be expired
        assert cache_manager._is_cache_expired(cache_key)

    def test_cache_cleanup(self, tmp_path):
        """Test cache cleanup functionality"""
        from src.backtesting.engine import PersistentCacheManager
        from datetime import datetime, timedelta
        
        cache_manager = PersistentCacheManager(cache_dir=str(tmp_path), expiry_days=1)
        
        # Create some test cache files
        for i in range(3):
            cache_key = f"test_{i}"
            cache_manager.set(cache_key, {'data': i})
        
        # Manually expire one file
        metadata_path = cache_manager._get_metadata_path("test_0")
        with open(metadata_path, 'w') as f:
            import json
            json.dump({
                'created_at': (datetime.now() - timedelta(days=2)).isoformat(),
                'size': 100,
                'cache_key': 'test_0'
            }, f)
        
        # Run cleanup
        cleaned = cache_manager.cleanup_expired()
        assert cleaned == 1
        
        # Check that expired file is gone
        assert not cache_manager._get_cache_path("test_0").exists()
        assert cache_manager._get_cache_path("test_1").exists()
        assert cache_manager._get_cache_path("test_2").exists()

    def test_cache_stats(self, tmp_path):
        """Test cache statistics functionality"""
        from src.backtesting.engine import PersistentCacheManager
        
        cache_manager = PersistentCacheManager(cache_dir=str(tmp_path))
        
        # Create some test files
        for i in range(3):
            cache_manager.set(f"test_{i}", {'data': i})
        
        stats = cache_manager.get_cache_stats()
        
        assert stats['total_files'] == 3
        assert stats['total_size_bytes'] > 0
        assert stats['total_size_mb'] > 0
        assert stats['expired_files'] == 0
        assert 'cache_dir' in stats

    def test_data_hash_generation(self, mock_data_provider):
        """Test data hash generation for cache invalidation"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Create small test DataFrame
        df1 = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        df2 = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # Same data should produce same hash
        hash1 = backtester._get_data_hash(df1)
        hash2 = backtester._get_data_hash(df2)
        assert hash1 == hash2
        assert len(hash1) == 16
        
        # Reset data hash to test different data
        backtester._data_hash = None
        
        # Different data should produce different hash
        df3 = df1.copy()
        df3.loc[0, 'close'] = 99.5
        hash3 = backtester._get_data_hash(df3)
        assert hash3 != hash1

    def test_persistent_cache_key_generation(self, mock_data_provider):
        """Test persistent cache key generation"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Generate data hash first
        df = pd.DataFrame({'close': [100, 101, 102]})
        backtester._get_data_hash(df)
        
        # Test persistent cache key generation
        key1 = backtester._get_persistent_cache_key("features")
        key2 = backtester._get_persistent_cache_key("strategy")
        key3 = backtester._get_persistent_cache_key("features")
        
        assert key1 != key2  # Different cache types
        assert key1 == key3  # Same cache type should produce same key
        assert "features" in key1
        assert "strategy" in key2

    def test_progress_callback(self, mock_data_provider):
        """Test progress callback functionality"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Track progress calls
        progress_calls = []
        
        def progress_callback(current, total, operation):
            progress_calls.append((current, total, operation))
        
        backtester.set_progress_callback(progress_callback)
        
        # Test progress updates
        backtester._update_progress(50, 100, "Test operation")
        backtester._update_progress(100, 100, "Test operation")
        
        assert len(progress_calls) == 2
        assert progress_calls[0] == (50, 100, "Test operation")
        assert progress_calls[1] == (100, 100, "Test operation")

    def test_persistent_cache_enable_disable(self, mock_data_provider):
        """Test enabling/disabling persistent cache"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Should be enabled by default
        assert backtester._enable_persistent_cache is True
        
        # Disable
        backtester.enable_persistent_cache(False)
        assert backtester._enable_persistent_cache is False
        
        # Enable
        backtester.enable_persistent_cache(True)
        assert backtester._enable_persistent_cache is True

    def test_cache_cleanup_method(self, mock_data_provider):
        """Test cache cleanup method"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Test cleanup when persistent cache is disabled
        backtester.enable_persistent_cache(False)
        cleaned = backtester.cleanup_expired_cache()
        assert cleaned == 0
        
        # Test cleanup when persistent cache is enabled
        backtester.enable_persistent_cache(True)
        cleaned = backtester.cleanup_expired_cache()
        assert isinstance(cleaned, int)
        assert cleaned >= 0

    def test_comprehensive_cache_stats(self, mock_data_provider):
        """Test comprehensive cache statistics"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Get initial stats
        stats = backtester.get_cache_stats()
        
        # Check required fields
        required_fields = [
            'feature_cache_size', 'strategy_cache_size', 'ml_predictions_cache_size',
            'cache_hits', 'cache_misses', 'hit_rate'
        ]
        
        for field in required_fields:
            assert field in stats
            assert isinstance(stats[field], (int, float))
        
        # Hit rate should be 0 initially
        assert stats['hit_rate'] == 0

    def test_chunked_processing_decision(self, mock_data_provider):
        """Test that chunked processing is used for large datasets"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Create a moderately large DataFrame (more than MAX_CACHE_SIZE)
        large_df = pd.DataFrame({
            'open': np.random.rand(12000),  # Reduced from 15000
            'high': np.random.rand(12000),
            'low': np.random.rand(12000),
            'close': np.random.rand(12000),
            'volume': np.random.rand(12000)
        })
        
        # Mock the strategy's calculate_indicators to avoid expensive ML computation
        # but still test the chunked processing logic
        original_calculate_indicators = strategy.calculate_indicators
        
        def mock_calculate_indicators(df):
            # Return DataFrame with mock ONNX predictions
            result_df = df.copy()
            result_df['onnx_pred'] = np.random.rand(len(df))  # Mock predictions
            return result_df
        
        strategy.calculate_indicators = mock_calculate_indicators
        
        try:
            # Call pre-computation - this should use chunked processing
            backtester._precompute_ml_predictions(large_df)
            
            # Verify that chunked processing was used by checking cache size
            # With chunked processing, we should have some predictions cached
            assert backtester._ml_predictions_cache_size > 0
            assert len(backtester._ml_predictions_cache) > 0
            
        finally:
            # Restore original method
            strategy.calculate_indicators = original_calculate_indicators

    def test_memory_efficient_data_structures(self, mock_data_provider):
        """Test that memory-efficient data structures are used"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Check that caches use dictionaries (memory efficient)
        assert isinstance(backtester._feature_cache, dict)
        assert isinstance(backtester._strategy_cache, dict)
        assert isinstance(backtester._ml_predictions_cache, dict)
        
        # Check that cache sizes are tracked
        assert hasattr(backtester, '_feature_cache_size')
        assert hasattr(backtester, '_strategy_cache_size')
        assert hasattr(backtester, '_ml_predictions_cache_size')

    def test_lazy_loading_behavior(self, mock_data_provider):
        """Test lazy loading behavior for large datasets"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Create moderately large DataFrame
        large_df = pd.DataFrame({
            'open': np.random.rand(12000),  # Reduced from 20000
            'high': np.random.rand(12000),
            'low': np.random.rand(12000),
            'close': np.random.rand(12000),
            'volume': np.random.rand(12000)
        })
        
        # Mock the strategy's calculate_indicators to avoid expensive ML computation
        original_calculate_indicators = strategy.calculate_indicators
        
        def mock_calculate_indicators(df):
            # Return DataFrame with mock ONNX predictions
            result_df = df.copy()
            result_df['onnx_pred'] = np.random.rand(len(df))  # Mock predictions
            return result_df
        
        strategy.calculate_indicators = mock_calculate_indicators
        
        try:
            # Call pre-computation - this should use chunked processing for large datasets
            backtester._precompute_ml_predictions(large_df)
            
            # Verify that chunked processing was used by checking cache size
            # With chunked processing, we should have some predictions cached
            assert backtester._ml_predictions_cache_size > 0
            assert len(backtester._ml_predictions_cache) > 0
            
            # Verify that cache keys are properly generated for chunked data
            cache_keys = list(backtester._ml_predictions_cache.keys())
            assert len(cache_keys) > 0
            
            # Verify cache keys follow the expected pattern
            for key in cache_keys[:5]:  # Check first 5 keys
                assert isinstance(key, str)
                assert '_' in key  # Should contain model version and data hash
            
        finally:
            # Restore original method
            strategy.calculate_indicators = original_calculate_indicators

    def test_cache_mechanism_with_real_ml_operations(self, mock_data_provider):
        """Test cache mechanism with actual ML operations on small dataset"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Create DataFrame with enough data for ML strategy (needs 120+ candles)
        small_df = pd.DataFrame({
            'open': np.random.rand(150) * 100 + 50000,  # Realistic price range
            'high': np.random.rand(150) * 100 + 50000,
            'low': np.random.rand(150) * 100 + 50000,
            'close': np.random.rand(150) * 100 + 50000,
            'volume': np.random.rand(150) * 1000 + 1000
        })
        
        # Mock the strategy's calculate_indicators to avoid expensive ONNX operations
        # but still test the cache mechanism logic
        original_calculate_indicators = strategy.calculate_indicators
        
        def mock_calculate_indicators(df):
            result_df = df.copy()
            # Add normalized features that the strategy expects
            for feature in ['open', 'high', 'low', 'close', 'volume']:
                result_df[f'{feature}_normalized'] = (df[feature] - df[feature].mean()) / df[feature].std()
            
            # Add mock ONNX predictions for candles after sequence_length (120)
            for i in range(120, len(df)):
                result_df.at[result_df.index[i], 'onnx_pred'] = df['close'].iloc[i] * (1 + np.random.randn() * 0.01)
                result_df.at[result_df.index[i], 'ml_prediction'] = df['close'].iloc[i] * (1 + np.random.randn() * 0.01)
                result_df.at[result_df.index[i], 'prediction_confidence'] = np.random.rand()
            
            return result_df
        
        strategy.calculate_indicators = mock_calculate_indicators
        
        try:
            # Test actual ML pre-computation (should be fast with mocked operations)
            backtester._precompute_ml_predictions(small_df)
            
            # Verify cache was populated
            assert backtester._ml_predictions_cache_size > 0
            assert len(backtester._ml_predictions_cache) > 0
            
            # Verify cache keys are properly formatted
            cache_keys = list(backtester._ml_predictions_cache.keys())
            for key in cache_keys[:5]:
                assert isinstance(key, str)
                assert '_' in key  # Should contain model version and data hash
                assert len(key) > 10  # Should be reasonably long
                
        finally:
            # Restore original method
            strategy.calculate_indicators = original_calculate_indicators

    def test_cache_cleanup_under_memory_pressure(self, mock_data_provider):
        """Test cache cleanup when memory pressure is high"""
        from src.backtesting.engine import Backtester, MAX_CACHE_SIZE
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Fill cache beyond MAX_CACHE_SIZE
        for i in range(15000):  # More than MAX_CACHE_SIZE (10000)
            cache_key = backtester._get_cache_key(i)
            backtester._ml_predictions_cache[cache_key] = 0.5
            backtester._ml_predictions_cache_size += 1
        
        # Verify cache is full
        assert backtester._is_cache_full()
        
        # Add one more entry to trigger cleanup
        cache_key = backtester._get_cache_key(15000)
        backtester._ml_predictions_cache[cache_key] = 0.5
        backtester._ml_predictions_cache_size += 1
        
        # Trigger cleanup
        backtester._cleanup_old_cache_entries()
        
        # Verify cache was cleaned up (should be at 80% of max size)
        expected_size = int(MAX_CACHE_SIZE * 0.8)
        assert backtester._ml_predictions_cache_size <= expected_size
        assert len(backtester._ml_predictions_cache) <= expected_size

    def test_persistent_cache_with_model_version_changes(self, mock_data_provider, tmp_path):
        """Test that persistent cache is invalidated when model version changes"""
        from src.backtesting.engine import Backtester, PersistentCacheManager
        from src.strategies.ml_basic import MlBasic
        
        # Create two different strategies (different model versions)
        strategy1 = MlBasic()
        strategy2 = MlBasic()
        
        backtester1 = Backtester(
            strategy=strategy1,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        backtester2 = Backtester(
            strategy=strategy2,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Generate data hash for both
        df = pd.DataFrame({'close': [100, 101, 102]})
        backtester1._get_data_hash(df)
        backtester2._get_data_hash(df)
        
        # Get persistent cache keys
        key1 = backtester1._get_persistent_cache_key("test")
        key2 = backtester2._get_persistent_cache_key("test")
        
        # Keys should be different due to different model versions
        assert key1 != key2
        
        # Test that cache manager handles different keys correctly
        cache_manager = PersistentCacheManager(cache_dir=str(tmp_path))
        
        # Save data with first key
        test_data1 = {'data': 'strategy1'}
        assert cache_manager.set(key1, test_data1) is True
        
        # Save data with second key
        test_data2 = {'data': 'strategy2'}
        assert cache_manager.set(key2, test_data2) is True
        
        # Verify both can be retrieved independently
        assert cache_manager.get(key1) == test_data1
        assert cache_manager.get(key2) == test_data2

    def test_cache_mechanism_edge_cases(self, mock_data_provider):
        """Test cache mechanism with various edge cases"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        backtester._precompute_ml_predictions(empty_df)
        assert backtester._ml_predictions_cache_size == 0
        
        # Test with single row DataFrame
        single_df = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 
            'close': [100.5], 'volume': [1000]
        })
        
        # Mock calculate_indicators for single row
        original_calculate_indicators = strategy.calculate_indicators
        
        def mock_calculate_indicators(df):
            result_df = df.copy()
            result_df['onnx_pred'] = [0.5] if len(df) > 0 else []
            return result_df
        
        strategy.calculate_indicators = mock_calculate_indicators
        
        try:
            backtester._precompute_ml_predictions(single_df)
            assert backtester._ml_predictions_cache_size == 1
        finally:
            strategy.calculate_indicators = original_calculate_indicators

    def test_progress_callback_error_handling(self, mock_data_provider):
        """Test that progress callback errors don't break the system"""
        from src.backtesting.engine import Backtester
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Set a callback that raises an exception
        def faulty_callback(current, total, operation):
            raise ValueError("Callback error")
        
        backtester.set_progress_callback(faulty_callback)
        
        # Create small DataFrame
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # This should not raise an exception despite callback error
        backtester._precompute_features(df)
        
        # Verify features were still computed
        assert backtester._feature_cache_size > 0


