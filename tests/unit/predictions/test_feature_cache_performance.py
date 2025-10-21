"""Performance-oriented smoke tests for FeatureCache operations."""

import time

import numpy as np
import pandas as pd
import pytest

from src.prediction.utils.caching import FeatureCache


class TestCachePerformance:
    """Lightweight timing checks to guard regressions."""

    @pytest.fixture
    def large_dataset(self):
        dates = pd.date_range("2020-01-01", periods=10000, freq="1h")
        np.random.seed(42)

        return pd.DataFrame(
            {
                "open": np.random.uniform(29000, 31000, 10000),
                "high": np.random.uniform(30000, 32000, 10000),
                "low": np.random.uniform(28000, 30000, 10000),
                "close": np.random.uniform(29000, 31000, 10000),
                "volume": np.random.uniform(100, 1000, 10000),
                "feature1": np.random.normal(0, 1, 10000),
                "feature2": np.random.uniform(-1, 1, 10000),
            },
            index=dates,
        )

    def test_cache_performance_large_data(self, large_dataset):
        cache = FeatureCache()
        extractor_name = "performance_test"
        config = {"param": "value"}

        start_time = time.time()
        cache.set(large_dataset, extractor_name, config, large_dataset)
        set_time = time.time() - start_time

        start_time = time.time()
        result = cache.get(large_dataset, extractor_name, config)
        get_time = time.time() - start_time

        assert result is not None
        pd.testing.assert_frame_equal(result, large_dataset)

        assert set_time < 5.0
        assert get_time < 1.0

        print(f"Cache set time: {set_time:.3f}s, get time: {get_time:.3f}s")

    def test_cache_hash_performance(self, large_dataset):
        cache = FeatureCache()

        start_time = time.time()
        quick_hash1 = cache._generate_quick_hash(large_dataset)
        quick_hash_time = time.time() - start_time

        start_time = time.time()
        full_hash1 = cache._generate_full_data_hash(large_dataset)
        full_hash_time = time.time() - start_time

        quick_hash2 = cache._generate_quick_hash(large_dataset)
        full_hash2 = cache._generate_full_data_hash(large_dataset)

        assert quick_hash1 == quick_hash2
        assert full_hash1 == full_hash2

        assert quick_hash_time < full_hash_time
        assert quick_hash_time < 0.1
        assert full_hash_time < 2.0

        print(f"Quick hash time: {quick_hash_time:.3f}s, full hash time: {full_hash_time:.3f}s")

    def test_cache_memory_efficiency(self, large_dataset):
        cache = FeatureCache()

        for i in range(5):
            config = {"iteration": i}
            cache.set(large_dataset, f"extractor_{i}", config, large_dataset)

        size_info = cache.get_size_info()

        assert size_info["total_entries"] == 5

        avg_size = size_info["average_entry_size_bytes"]
        total_size = size_info["total_memory_bytes"]

        assert avg_size > 0
        assert total_size > avg_size * 4
        assert total_size < avg_size * 10

        print(f"Average entry size: {avg_size:,} bytes")
        print(f"Total cache size: {total_size:,} bytes")
