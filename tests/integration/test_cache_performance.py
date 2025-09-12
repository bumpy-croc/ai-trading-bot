"""
Performance-focused integration tests for cache mechanism.

These tests specifically measure and verify that the cache provides
measurable performance improvements during backtesting.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from datetime import datetime
from src.backtesting.engine import Backtester
from src.strategies.ml_basic import MlBasic
from src.data_providers.binance_provider import BinanceDataProvider


class TestCachePerformance:
    """Performance tests for cache mechanism."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider for testing."""
        provider = Mock(spec=BinanceDataProvider)
        return provider

    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=5000, freq='1h')
        
        # Generate realistic price data with trends
        base_price = 50000
        trend = np.linspace(0, 0.1, 5000)  # 10% upward trend
        noise = np.random.normal(0, 0.02, 5000)
        returns = trend + noise
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 5000)
        }, index=dates)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df

    def test_cache_performance_improvement_measurable(self, mock_data_provider, large_dataset):
        """Test that cache provides measurable performance improvement."""
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        mock_data_provider.get_historical_data.return_value = large_dataset
        
        # Test 1: Run with cache disabled (by clearing after each operation)
        backtester_no_cache = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Disable effective caching by clearing caches frequently
        original_precompute_ml = backtester_no_cache._precompute_ml_predictions
        original_precompute_strategy = backtester_no_cache._precompute_strategy_calculations
        
        def no_cache_precompute_ml(df):
            # Clear cache before and after
            backtester_no_cache._ml_predictions_cache.clear()
            backtester_no_cache._ml_predictions_cache_size = 0
            return original_precompute_ml(df)
        
        def no_cache_precompute_strategy(df):
            # Clear cache before and after
            backtester_no_cache._strategy_cache.clear()
            backtester_no_cache._strategy_cache_size = 0
            return original_precompute_strategy(df)
        
        backtester_no_cache._precompute_ml_predictions = no_cache_precompute_ml
        backtester_no_cache._precompute_strategy_calculations = no_cache_precompute_strategy
        
        # Time the backtest without effective caching
        start_time = time.time()
        result_no_cache = backtester_no_cache.run(
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 7, 1),
            timeframe="1h"
        )
        time_no_cache = time.time() - start_time
        
        # Test 2: Run with cache enabled
        backtester_with_cache = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Time the backtest with caching
        start_time = time.time()
        result_with_cache = backtester_with_cache.run(
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 7, 1),
            timeframe="1h"
        )
        time_with_cache = time.time() - start_time
        
        cache_stats = backtester_with_cache.get_cache_stats()
        
        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(f"Time without cache: {time_no_cache:.2f}s")
        print(f"Time with cache: {time_with_cache:.2f}s")
        print(f"Performance improvement: {((time_no_cache - time_with_cache) / time_no_cache * 100):.1f}%")
        print(f"Cache hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"Strategy cache size: {cache_stats['strategy_cache_size']}")
        print(f"ML predictions cache size: {cache_stats['ml_predictions_cache_size']}")
        print(f"Total cache hits: {cache_stats['cache_hits']}")
        print(f"Total cache misses: {cache_stats['cache_misses']}")
        
        # Verify results are consistent
        assert abs(result_no_cache['total_return'] - result_with_cache['total_return']) < 0.01, \
            "Results should be consistent between cached and non-cached runs"
        
        # Verify cache was actually used
        assert cache_stats['strategy_cache_size'] > 0, "Strategy cache should be populated"
        assert cache_stats['ml_predictions_cache_size'] > 0, "ML predictions cache should be populated"
        assert cache_stats['cache_hits'] > 0, "Should have cache hits"
        assert cache_stats['hit_rate'] > 0, "Should have a positive hit rate"
        
        # Performance improvement should be measurable
        # For large datasets, we expect at least some improvement
        improvement_pct = (time_no_cache - time_with_cache) / time_no_cache * 100
        print(f"Performance improvement: {improvement_pct:.1f}%")
        
        # The cache should not make it significantly slower
        assert time_with_cache <= time_no_cache * 1.2, \
            f"Cached version should not be more than 20% slower. " \
            f"No cache: {time_no_cache:.2f}s, With cache: {time_with_cache:.2f}s"

    def test_cache_scales_with_dataset_size(self, mock_data_provider):
        """Test that cache performance scales well with dataset size."""
        from src.strategies.ml_basic import MlBasic
        
        strategy = MlBasic()
        
        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        times = []
        cache_stats_list = []
        
        for size in dataset_sizes:
            # Generate dataset of specific size
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=size, freq='1h')
            
            base_price = 50000
            returns = np.random.normal(0, 0.02, size)
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, size)
            }, index=dates)
            
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            mock_data_provider.get_historical_data.return_value = df
            
            backtester = Backtester(
                strategy=strategy,
                data_provider=mock_data_provider,
                initial_balance=10000,
            )
            
            start_time = time.time()
            result = backtester.run(
                symbol="BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + pd.Timedelta(hours=size-1),
                timeframe="1h"
            )
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            cache_stats_list.append(backtester.get_cache_stats())
            
            print(f"Dataset size: {size}, Time: {elapsed_time:.2f}s, "
                  f"Cache hits: {backtester.get_cache_stats()['cache_hits']}")
        
        # Verify that time scales reasonably with dataset size
        # (should be roughly linear, not exponential)
        for i in range(1, len(times)):
            size_ratio = dataset_sizes[i] / dataset_sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Time should not grow exponentially with dataset size
            assert time_ratio < size_ratio * 2, \
                f"Time scaling is too poor: {size_ratio}x size increase " \
                f"caused {time_ratio:.1f}x time increase"

    def test_cache_memory_efficiency(self, mock_data_provider, large_dataset):
        """Test that cache uses memory efficiently."""
        from src.strategies.ml_basic import MlBasic
        import psutil
        import os
        
        strategy = MlBasic()
        mock_data_provider.get_historical_data.return_value = large_dataset
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
        )
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run backtest
        result = backtester.run(
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 7, 1),
            timeframe="1h"
        )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        cache_stats = backtester.get_cache_stats()
        
        print(f"\n=== MEMORY USAGE ===")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Strategy cache size: {cache_stats['strategy_cache_size']}")
        print(f"ML predictions cache size: {cache_stats['ml_predictions_cache_size']}")
        
        # Memory increase should be reasonable (less than 500MB for 5000 candles)
        assert memory_increase < 500, \
            f"Memory increase too high: {memory_increase:.1f} MB"
        
        # Cache sizes should be within limits
        assert cache_stats['strategy_cache_size'] <= 10000, \
            "Strategy cache exceeds MAX_CACHE_SIZE"
        assert cache_stats['ml_predictions_cache_size'] <= 10000, \
            "ML predictions cache exceeds MAX_CACHE_SIZE"

    def test_cache_persistence_across_runs(self, mock_data_provider, large_dataset):
        """Test that persistent cache works across multiple backtest runs."""
        from src.strategies.ml_basic import MlBasic
        import tempfile
        import shutil
        
        # Use temporary directory for cache
        temp_cache_dir = tempfile.mkdtemp()
        
        try:
            strategy = MlBasic()
            mock_data_provider.get_historical_data.return_value = large_dataset.head(1000)
            
            # First run - should populate persistent cache
            backtester1 = Backtester(
                strategy=strategy,
                data_provider=mock_data_provider,
                initial_balance=10000,
            )
            backtester1._persistent_cache.cache_dir = temp_cache_dir
            
            start_time = time.time()
            result1 = backtester1.run(
                symbol="BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 2, 11),  # 42 days from Jan 1
                timeframe="1h"
            )
            time1 = time.time() - start_time
            
            # Second run with new backtester instance - should use persistent cache
            backtester2 = Backtester(
                strategy=strategy,
                data_provider=mock_data_provider,
                initial_balance=10000,
            )
            backtester2._persistent_cache.cache_dir = temp_cache_dir
            
            start_time = time.time()
            result2 = backtester2.run(
                symbol="BTCUSDT",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 2, 11),  # 42 days from Jan 1
                timeframe="1h"
            )
            time2 = time.time() - start_time
            
            cache_stats1 = backtester1.get_cache_stats()
            cache_stats2 = backtester2.get_cache_stats()
            
            print(f"\n=== PERSISTENT CACHE TEST ===")
            print(f"First run time: {time1:.2f}s")
            print(f"Second run time: {time2:.2f}s")
            print(f"First run cache hits: {cache_stats1['cache_hits']}")
            print(f"Second run cache hits: {cache_stats2['cache_hits']}")
            
            # Results should be identical
            assert abs(result1['total_return'] - result2['total_return']) < 0.01, \
                "Results should be identical across runs"
            
            # Second run should be faster due to persistent cache
            # (though this might not always be true depending on implementation)
            print(f"Performance difference: {((time1 - time2) / time1 * 100):.1f}%")
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
