"""
Tests for Feature Caching

This module contains tests for the feature caching utilities
to ensure proper cache functionality and performance.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch
from src.prediction.utils.caching import (
    FeatureCache, CacheEntry, get_global_feature_cache, clear_global_feature_cache
)


class TestCacheEntry:
    """Test cases for CacheEntry."""
    
    def test_cache_entry_creation(self):
        """Test creation of cache entry."""
        test_data = pd.DataFrame({'a': [1, 2, 3]})
        entry = CacheEntry(
            data=test_data,
            timestamp=time.time(),
            ttl=300,
            data_hash="test_hash",
            quick_hash="quick_test_hash"  # * Added required quick_hash parameter
        )
        
        assert not entry.data.empty
        assert entry.ttl == 300
        assert entry.data_hash == "test_hash"
        assert entry.quick_hash == "quick_test_hash"  # * Verify quick_hash is set
        assert entry.is_valid()  # Should be valid when just created
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        test_data = pd.DataFrame({'a': [1, 2, 3]})
        
        # Create entry with very short TTL
        entry = CacheEntry(
            data=test_data,
            timestamp=time.time() - 10,  # 10 seconds ago
            ttl=5,  # 5 second TTL
            data_hash="test_hash",
            quick_hash="quick_test_hash"  # * Added required quick_hash parameter
        )
        
        assert entry.is_expired()
        assert not entry.is_valid()
    
    def test_cache_entry_not_expired(self):
        """Test cache entry that hasn't expired."""
        test_data = pd.DataFrame({'a': [1, 2, 3]})
        
        # Create entry with long TTL
        entry = CacheEntry(
            data=test_data,
            timestamp=time.time(),
            ttl=3600,  # 1 hour TTL
            data_hash="test_hash",
            quick_hash="quick_test_hash"  # * Added required quick_hash parameter
        )
        
        assert not entry.is_expired()
        assert entry.is_valid()


class TestFeatureCache:
    """Test cases for FeatureCache."""
    
    @pytest.fixture
    def cache(self):
        """Create a fresh FeatureCache instance."""
        return FeatureCache(default_ttl=300)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        return pd.DataFrame({
            'open': np.random.uniform(29000, 31000, 100),
            'high': np.random.uniform(30000, 32000, 100),
            'low': np.random.uniform(28000, 30000, 100),
            'close': np.random.uniform(29000, 31000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
    
    @pytest.fixture
    def sample_result(self, sample_data):
        """Create sample feature extraction result."""
        result = sample_data.copy()
        result['rsi'] = np.random.uniform(20, 80, len(sample_data))
        result['atr'] = np.random.uniform(100, 500, len(sample_data))
        return result
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.default_ttl == 300
        assert len(cache._cache) == 0
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['hit_rate'] == 0.0
        assert stats['total_requests'] == 0
        assert 'quick_hash_efficiency' in stats  # * Verify new stat is present
    
    def test_quick_hash_generation(self, cache, sample_data):
        """Test quick hash generation."""
        hash1 = cache._generate_quick_hash(sample_data)
        hash2 = cache._generate_quick_hash(sample_data)
        
        # Same data should produce same quick hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Different data should produce different quick hash
        different_data = sample_data.copy()
        different_data['close'] += 1
        hash3 = cache._generate_quick_hash(different_data)
        assert hash1 != hash3
    
    def test_full_data_hash_generation(self, cache, sample_data):
        """Test full data hash generation."""
        hash1 = cache._generate_full_data_hash(sample_data)
        hash2 = cache._generate_full_data_hash(sample_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Different data should produce different hash
        different_data = sample_data.copy()
        different_data['close'] += 1
        hash3 = cache._generate_full_data_hash(different_data)
        assert hash1 != hash3
    
    def test_cache_key_generation(self, cache, sample_data):
        """Test cache key generation."""
        config = {'param1': 'value1', 'param2': 42}
        
        key1 = cache._generate_cache_key(sample_data, "test_extractor", config)
        key2 = cache._generate_cache_key(sample_data, "test_extractor", config)
        
        # Same inputs should produce same key
        assert key1 == key2
        
        # Different extractor name should produce different key
        key3 = cache._generate_cache_key(sample_data, "different_extractor", config)
        assert key1 != key3
        
        # Different config should produce different key
        different_config = {'param1': 'value2', 'param2': 42}
        key4 = cache._generate_cache_key(sample_data, "test_extractor", different_config)
        assert key1 != key4
    
    def test_cache_set_and_get(self, cache, sample_data, sample_result):
        """Test basic cache set and get operations."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Initially should miss cache
        result = cache.get(sample_data, extractor_name, config)
        assert result is None
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Should now hit cache
        cached_result = cache.get(sample_data, extractor_name, config)
        assert cached_result is not None
        pd.testing.assert_frame_equal(cached_result, sample_result)
        
        # Check stats
        stats = cache.get_stats()
        assert stats['sets'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert 'quick_hash_matches' in stats  # * Verify new stats are present
    
    def test_cache_has_method(self, cache, sample_data, sample_result):
        """Test cache has method."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Initially should not have entry
        assert not cache.has(sample_data, extractor_name, config)
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Should now have entry
        assert cache.has(sample_data, extractor_name, config)
    
    def test_cache_expiration(self, sample_data, sample_result):
        """Test cache expiration functionality."""
        # Create cache with very short TTL
        cache = FeatureCache(default_ttl=1)
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Should initially hit cache
        assert cache.has(sample_data, extractor_name, config)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should now miss cache due to expiration
        assert not cache.has(sample_data, extractor_name, config)
        
        # Get should return None and clean up expired entry
        result = cache.get(sample_data, extractor_name, config)
        assert result is None
        
        # Stats should reflect eviction
        stats = cache.get_stats()
        assert stats['evictions'] > 0
    
    def test_cache_clear(self, cache, sample_data, sample_result):
        """Test cache clear functionality."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Should have entry
        assert cache.has(sample_data, extractor_name, config)
        
        # Clear cache
        cache.clear()
        
        # Should not have entry after clear
        assert not cache.has(sample_data, extractor_name, config)
        
        # Stats should be reset (note: has() calls get() internally, so misses=1)
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 1  # * has() method calls get() which increments misses
        assert stats['sets'] == 0
        assert stats['evictions'] == 0
        assert stats['quick_hash_matches'] == 0  # * Verify new stats are reset
    
    def test_cache_cleanup_expired(self, sample_data, sample_result):
        """Test cache cleanup of expired entries."""
        # Create cache with very short TTL
        cache = FeatureCache(default_ttl=1)
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Should have entry initially
        assert len(cache._cache) == 1
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup should remove expired entries
        removed_count = cache.cleanup_expired()
        assert removed_count == 1
        assert len(cache._cache) == 0
    
    def test_cache_stats(self, cache, sample_data, sample_result):
        """Test cache statistics."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['hit_rate'] == 0.0
        assert stats['total_requests'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['sets'] == 0
        assert stats['evictions'] == 0
        assert stats['quick_hash_matches'] == 0  # * Verify new stats are present
        assert stats['full_hash_verifications'] == 0
        
        # After some operations
        cache.set(sample_data, extractor_name, config, sample_result)
        cache.get(sample_data, extractor_name, config)
        cache.get(sample_data, "different_extractor", config)  # Miss
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        assert stats['sets'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['total_requests'] == 2
        assert stats['hit_rate'] == 0.5
        assert stats['quick_hash_efficiency'] >= 0.0  # * Verify efficiency calculation
    
    def test_cache_size_info(self, cache, sample_data, sample_result):
        """Test cache size information."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Get size info
        size_info = cache.get_size_info()
        
        assert size_info['total_entries'] == 1
        assert size_info['total_memory_bytes'] > 0
        assert size_info['average_entry_size_bytes'] > 0
        assert size_info['largest_entry_size_bytes'] > 0
    
    def test_cache_data_isolation(self, cache, sample_data, sample_result):
        """Test that cached data is isolated from original."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Set cache
        cache.set(sample_data, extractor_name, config, sample_result)
        
        # Modify original data
        original_copy = sample_data.copy()
        sample_data['close'] += 1000
        
        # Get cached result
        cached_result = cache.get(original_copy, extractor_name, config)
        
        # Cached result should not be affected by original data modification
        assert cached_result is not None
        assert cached_result['close'].iloc[0] != sample_data['close'].iloc[0]
    
    def test_cache_different_data_same_extractor(self, cache, sample_result):
        """Test that different data with same extractor produces different cache entries."""
        extractor_name = "test_extractor"
        config = {'param': 'value'}
        
        # Create two different datasets
        data1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        data2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})  # Different value
        
        # Set cache for both
        cache.set(data1, extractor_name, config, sample_result)
        cache.set(data2, extractor_name, config, sample_result)
        
        # Should have two separate entries
        assert cache.has(data1, extractor_name, config)
        assert cache.has(data2, extractor_name, config)
        
        # Should have 2 entries total
        stats = cache.get_stats()
        assert stats['total_entries'] == 2


class TestGlobalFeatureCache:
    """Test cases for global feature cache functionality."""
    
    def test_get_global_cache_singleton(self):
        """Test that global cache is a singleton."""
        cache1 = get_global_feature_cache()
        cache2 = get_global_feature_cache()
        
        assert cache1 is cache2
        
        # Clean up
        clear_global_feature_cache()
    
    def test_clear_global_cache(self):
        """Test clearing global cache."""
        cache = get_global_feature_cache()
        sample_data = pd.DataFrame({'a': [1, 2, 3]})
        
        # Add some data
        cache.set(sample_data, "test", {}, sample_data)
        assert cache.has(sample_data, "test", {})
        
        # Clear global cache
        clear_global_feature_cache()
        
        # Should not have data after clear
        assert not cache.has(sample_data, "test", {})
    
    def test_global_cache_persistence_across_calls(self):
        """Test that global cache persists across function calls."""
        sample_data = pd.DataFrame({'a': [1, 2, 3]})
        
        # Get cache and add data
        cache1 = get_global_feature_cache()
        cache1.set(sample_data, "test", {}, sample_data)
        
        # Get cache again and verify data is still there
        cache2 = get_global_feature_cache()
        assert cache2.has(sample_data, "test", {})
        
        # Clean up
        clear_global_feature_cache()


class TestCachePerformance:
    """Performance tests for caching functionality."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing."""
        dates = pd.date_range('2020-01-01', periods=10000, freq='1h')
        np.random.seed(42)
        
        return pd.DataFrame({
            'open': np.random.uniform(29000, 31000, 10000),
            'high': np.random.uniform(30000, 32000, 10000),
            'low': np.random.uniform(28000, 30000, 10000),
            'close': np.random.uniform(29000, 31000, 10000),
            'volume': np.random.uniform(100, 1000, 10000),
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.uniform(-1, 1, 10000)
        }, index=dates)
    
    def test_cache_performance_large_data(self, large_dataset):
        """Test cache performance with large datasets."""
        cache = FeatureCache()
        extractor_name = "performance_test"
        config = {'param': 'value'}
        
        # Time cache set operation
        start_time = time.time()
        cache.set(large_dataset, extractor_name, config, large_dataset)
        set_time = time.time() - start_time
        
        # Time cache get operation (first time - should be fast)
        start_time = time.time()
        result = cache.get(large_dataset, extractor_name, config)
        get_time = time.time() - start_time
        
        # Verify result is correct
        assert result is not None
        pd.testing.assert_frame_equal(result, large_dataset)
        
        # Cache operations should be reasonably fast
        assert set_time < 5.0  # Should cache within 5 seconds
        assert get_time < 1.0  # Should retrieve within 1 second
        
        print(f"Cache set time: {set_time:.3f}s, get time: {get_time:.3f}s")
    
    def test_cache_hash_performance(self, large_dataset):
        """Test hash generation performance."""
        cache = FeatureCache()
        
        # Time quick hash generation
        start_time = time.time()
        quick_hash1 = cache._generate_quick_hash(large_dataset)
        quick_hash_time = time.time() - start_time
        
        # Time full hash generation
        start_time = time.time()
        full_hash1 = cache._generate_full_data_hash(large_dataset)
        full_hash_time = time.time() - start_time
        
        # Generate again to test consistency
        quick_hash2 = cache._generate_quick_hash(large_dataset)
        full_hash2 = cache._generate_full_data_hash(large_dataset)
        
        # Hashes should be identical
        assert quick_hash1 == quick_hash2
        assert full_hash1 == full_hash2
        
        # Quick hash should be faster than full hash
        assert quick_hash_time < full_hash_time
        assert quick_hash_time < 0.1  # Should be very fast
        assert full_hash_time < 2.0  # Should hash within 2 seconds
        
        print(f"Quick hash time: {quick_hash_time:.3f}s, full hash time: {full_hash_time:.3f}s")
    
    def test_cache_memory_efficiency(self, large_dataset):
        """Test cache memory usage."""
        cache = FeatureCache()
        
        # Add multiple entries
        for i in range(5):
            config = {'iteration': i}
            cache.set(large_dataset, f"extractor_{i}", config, large_dataset)
        
        # Check memory usage
        size_info = cache.get_size_info()
        
        # Should have 5 entries
        assert size_info['total_entries'] == 5
        
        # Memory usage should be reasonable (not excessive duplication)
        # Each entry should be roughly the same size
        avg_size = size_info['average_entry_size_bytes']
        total_size = size_info['total_memory_bytes']
        
        assert avg_size > 0
        assert total_size > avg_size * 4  # At least 4 entries worth
        assert total_size < avg_size * 10  # Not excessively large
        
        print(f"Average entry size: {avg_size:,} bytes")
        print(f"Total cache size: {total_size:,} bytes")