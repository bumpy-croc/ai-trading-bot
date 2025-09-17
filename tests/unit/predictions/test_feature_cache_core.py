"""Core behaviour tests for the FeatureCache implementation."""

import time

import numpy as np
import pandas as pd
import pytest

from src.prediction.utils.caching import FeatureCache


class TestFeatureCache:
    """Exercise primary cache operations and statistics."""

    @pytest.fixture
    def cache(self):
        return FeatureCache(default_ttl=300)

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="1h")
        return pd.DataFrame(
            {
                "open": np.random.uniform(29000, 31000, 100),
                "high": np.random.uniform(30000, 32000, 100),
                "low": np.random.uniform(28000, 30000, 100),
                "close": np.random.uniform(29000, 31000, 100),
                "volume": np.random.uniform(100, 1000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_result(self, sample_data):
        result = sample_data.copy()
        result["rsi"] = np.random.uniform(20, 80, len(sample_data))
        result["atr"] = np.random.uniform(100, 500, len(sample_data))
        return result

    def test_cache_initialization(self, cache):
        assert cache.default_ttl == 300
        assert len(cache._cache) == 0

        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0
        assert "quick_hash_efficiency" in stats

    def test_quick_hash_generation(self, cache, sample_data):
        hash1 = cache._generate_quick_hash(sample_data)
        hash2 = cache._generate_quick_hash(sample_data)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        different_data = sample_data.copy()
        different_data["close"] += 1
        hash3 = cache._generate_quick_hash(different_data)
        assert hash1 != hash3

    def test_full_data_hash_generation(self, cache, sample_data):
        hash1 = cache._generate_full_data_hash(sample_data)
        hash2 = cache._generate_full_data_hash(sample_data)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        different_data = sample_data.copy()
        different_data["close"] += 1
        hash3 = cache._generate_full_data_hash(different_data)
        assert hash1 != hash3

    def test_cache_key_generation(self, cache, sample_data):
        config = {"param1": "value1", "param2": 42}

        key1 = cache._generate_cache_key(sample_data, "test_extractor", config)
        key2 = cache._generate_cache_key(sample_data, "test_extractor", config)

        assert key1 == key2

        key3 = cache._generate_cache_key(sample_data, "different_extractor", config)
        assert key1 != key3

        different_config = {"param1": "value2", "param2": 42}
        key4 = cache._generate_cache_key(sample_data, "test_extractor", different_config)
        assert key1 != key4

    def test_cache_set_and_get(self, cache, sample_data, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        result = cache.get(sample_data, extractor_name, config)
        assert result is None

        cache.set(sample_data, extractor_name, config, sample_result)

        cached_result = cache.get(sample_data, extractor_name, config)
        assert cached_result is not None
        pd.testing.assert_frame_equal(cached_result, sample_result)

        stats = cache.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "quick_hash_matches" in stats

    def test_cache_has_method(self, cache, sample_data, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        assert not cache.has(sample_data, extractor_name, config)

        cache.set(sample_data, extractor_name, config, sample_result)

        assert cache.has(sample_data, extractor_name, config)

    def test_cache_expiration(self, sample_data, sample_result):
        cache = FeatureCache(default_ttl=1)
        extractor_name = "test_extractor"
        config = {"param": "value"}

        cache.set(sample_data, extractor_name, config, sample_result)

        assert cache.has(sample_data, extractor_name, config)

        time.sleep(1.1)

        assert not cache.has(sample_data, extractor_name, config)

        result = cache.get(sample_data, extractor_name, config)
        assert result is None

        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_cache_clear(self, cache, sample_data, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        cache.set(sample_data, extractor_name, config, sample_result)

        assert cache.has(sample_data, extractor_name, config)

        cache.clear()

        assert not cache.has(sample_data, extractor_name, config)

        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["sets"] == 0
        assert stats["evictions"] == 0
        assert stats["quick_hash_matches"] == 0

    def test_cache_cleanup_expired(self, sample_data, sample_result):
        cache = FeatureCache(default_ttl=1)
        extractor_name = "test_extractor"
        config = {"param": "value"}

        cache.set(sample_data, extractor_name, config, sample_result)

        assert len(cache._cache) == 1

        time.sleep(1.1)

        removed_count = cache.cleanup_expired()
        assert removed_count == 1
        assert len(cache._cache) == 0

    def test_cache_stats(self, cache, sample_data, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["evictions"] == 0
        assert stats["quick_hash_matches"] == 0
        assert stats["full_hash_verifications"] == 0

        cache.set(sample_data, extractor_name, config, sample_result)
        cache.get(sample_data, extractor_name, config)
        cache.get(sample_data, "different_extractor", config)

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["sets"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["quick_hash_efficiency"] >= 0.0

    def test_cache_size_info(self, cache, sample_data, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        cache.set(sample_data, extractor_name, config, sample_result)

        size_info = cache.get_size_info()

        assert size_info["total_entries"] == 1
        assert size_info["total_memory_bytes"] > 0
        assert size_info["average_entry_size_bytes"] > 0
        assert size_info["largest_entry_size_bytes"] > 0

    def test_cache_data_isolation(self, cache, sample_data, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        cache.set(sample_data, extractor_name, config, sample_result)

        original_copy = sample_data.copy()
        sample_data["close"] += 1000

        cached_result = cache.get(original_copy, extractor_name, config)

        assert cached_result is not None
        assert cached_result["close"].iloc[0] != sample_data["close"].iloc[0]

    def test_cache_different_data_same_extractor(self, cache, sample_result):
        extractor_name = "test_extractor"
        config = {"param": "value"}

        data1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})

        cache.set(data1, extractor_name, config, sample_result)
        cache.set(data2, extractor_name, config, sample_result)

        assert cache.has(data1, extractor_name, config)
        assert cache.has(data2, extractor_name, config)

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
