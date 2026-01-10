"""
Integration tests for prediction caching functionality.

This module tests the complete prediction caching pipeline including
database operations, cache hit/miss scenarios, and performance characteristics.
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from src.database.manager import DatabaseManager
from src.database.models import PredictionCache
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine
from src.prediction.utils.caching import PredictionCacheManager


class TestPredictionCachingIntegration:
    """Integration tests for prediction caching"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create test configuration
        self.config = PredictionConfig()
        self.config.prediction_cache_enabled = True
        self.config.prediction_cache_ttl = 60
        self.config.prediction_cache_max_size = 100

        # Sample test data
        self.test_data = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        self.features = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager"""
        mock_db = MagicMock(spec=DatabaseManager)
        mock_session = MagicMock(spec=Session)

        # Mock session context manager
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        return mock_db, mock_session

    def test_cache_manager_integration(self, mock_db_manager):
        """Test complete cache manager integration"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Test cache miss
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = cache_manager.get(self.features, "test_model", {"param": "value"})
        assert result is None
        assert cache_manager._stats["misses"] == 1

        # Test cache set
        cache_manager.set(self.features, "test_model", {"param": "value"}, 100.5, 0.8, 1)
        assert cache_manager._stats["sets"] == 1

        # Verify session.add was called
        mock_session.add.assert_called_once()
        added_entry = mock_session.add.call_args[0][0]
        assert isinstance(added_entry, PredictionCache)
        assert added_entry.predicted_price == 100.5
        assert added_entry.confidence == 0.8
        assert added_entry.direction == 1

    def test_cache_hit_scenario(self, mock_db_manager):
        """Test cache hit scenario"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Mock existing cache entry
        mock_entry = MagicMock()
        mock_entry.predicted_price = 100.5
        mock_entry.confidence = 0.8
        mock_entry.direction = 1
        mock_entry.access_count = 5
        mock_entry.expires_at = datetime.now(UTC) + timedelta(seconds=30)

        mock_session.query.return_value.filter.return_value.first.return_value = mock_entry

        # Test cache hit
        result = cache_manager.get(self.features, "test_model", {"param": "value"})

        assert result is not None
        assert result["price"] == 100.5
        assert result["confidence"] == 0.8
        assert result["direction"] == 1
        assert result["cache_hit"] is True
        assert cache_manager._stats["hits"] == 1

    def test_cache_expiration(self, mock_db_manager):
        """Test cache expiration handling"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Mock the query to return None for expired entries (this is handled by the database filter)
        # The actual cache manager queries with expires_at > datetime.now(UTC), so expired entries aren't returned
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Test expired cache - should return None since expired entries are filtered out by the query
        result = cache_manager.get(self.features, "test_model", {"param": "value"})

        assert result is None
        assert cache_manager._stats["misses"] == 1

    def test_cache_size_limit_enforcement(self, mock_db_manager):
        """Test cache size limit enforcement"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager with small size limit
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=2)

        # Mock current count exceeding limit
        mock_session.query.return_value.count.return_value = 3

        # Mock oldest entries to remove
        mock_entries = [MagicMock() for _ in range(1)]
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_entries
        )

        # Test size limit enforcement
        result = cache_manager._enforce_size_limit(mock_session)

        assert result == 1
        assert cache_manager._stats["evictions"] == 1
        mock_session.delete.assert_called_once()

    def test_cache_invalidation(self, mock_db_manager):
        """Test cache invalidation"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Mock invalidation
        mock_session.query.return_value.filter.return_value.delete.return_value = 5

        # Test model invalidation
        result = cache_manager.invalidate_model("test_model")
        assert result == 5

        # Test config invalidation
        result = cache_manager.invalidate_config("test_model", {"param": "value"})
        assert result == 5

    def test_cache_statistics(self, mock_db_manager):
        """Test cache statistics collection"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Mock statistics
        mock_session.query.return_value.count.return_value = 50
        mock_session.query.return_value.filter.return_value.count.return_value = 5

        # Get statistics
        stats = cache_manager.get_stats()

        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert "hit_rate" in stats
        assert "total_requests" in stats
        assert stats["total_entries"] == 50
        assert stats["expired_entries"] == 5

    @patch("src.prediction.engine.PredictionCacheManager")
    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_engine_cache_integration(
        self, mock_feature_pipeline, mock_model_registry, mock_cache_manager_class
    ):
        """Test prediction engine cache integration"""
        mock_db_manager = MagicMock()
        mock_cache_manager = MagicMock()
        mock_cache_manager_class.return_value = mock_cache_manager

        # Create prediction engine
        engine = PredictionEngine(self.config, mock_db_manager)

        # Verify cache manager was created
        mock_cache_manager_class.assert_called_once_with(
            mock_db_manager,
            ttl=self.config.prediction_cache_ttl,
            max_size=self.config.prediction_cache_max_size,
        )

        # Test cache management methods
        engine.clear_caches()
        mock_cache_manager.clear.assert_called_once()

        engine.get_cache_stats()
        mock_cache_manager.get_stats.assert_called_once()

    def test_cache_performance_characteristics(self, mock_db_manager):
        """Test cache performance characteristics"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Test cache miss performance (should be fast)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        start_time = time.time()
        num_miss_ops = 100
        for _ in range(num_miss_ops):
            cache_manager.get(self.features, "test_model", {"param": "value"})
        cache_miss_time = time.time() - start_time

        # Cache miss overhead should be minimal
        # Increased threshold for CI tolerance: 500ms for 100 operations (5ms per op)
        assert cache_miss_time < 0.5, (
            f"Cache miss operations too slow: {cache_miss_time:.3f}s for {num_miss_ops} ops "
            f"({cache_miss_time/num_miss_ops*1000:.2f}ms per op, expected <5ms per op)"
        )

        # Test cache hit performance (should be very fast)
        mock_entry = MagicMock()
        mock_entry.predicted_price = 100.5
        mock_entry.confidence = 0.8
        mock_entry.direction = 1
        mock_entry.access_count = 0
        mock_entry.expires_at = datetime.now(UTC) + timedelta(seconds=30)

        mock_session.query.return_value.filter.return_value.first.return_value = mock_entry

        start_time = time.time()
        num_hit_ops = 10
        for _ in range(num_hit_ops):
            cache_manager.get(self.features, "test_model", {"param": "value"})
        cache_hit_time = time.time() - start_time

        # Cache hit should be reasonably fast
        # Increased threshold for CI tolerance: 200ms for 10 operations (20ms per op)
        assert cache_hit_time < 0.2, (
            f"Cache hit operations too slow: {cache_hit_time:.3f}s for {num_hit_ops} ops "
            f"({cache_hit_time/num_hit_ops*1000:.2f}ms per op, expected <20ms per op)"
        )

        # Verify cache stats are being tracked correctly
        assert cache_manager._stats["misses"] == num_miss_ops
        assert cache_manager._stats["hits"] == num_hit_ops

    def test_cache_consistency(self, mock_db_manager):
        """Test cache consistency across multiple operations"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Test that same inputs always produce same cache key
        key1 = cache_manager._generate_cache_key(self.features, "model1", {"param": "value"})
        key2 = cache_manager._generate_cache_key(self.features, "model1", {"param": "value"})
        assert key1 == key2

        # Test that different inputs produce different cache keys
        different_features = np.array([[1.0, 2.0, 3.1]], dtype=np.float32)
        key3 = cache_manager._generate_cache_key(different_features, "model1", {"param": "value"})
        assert key1 != key3

        # Test that different models produce different cache keys
        key4 = cache_manager._generate_cache_key(self.features, "model2", {"param": "value"})
        assert key1 != key4

    def test_cache_error_handling(self, mock_db_manager):
        """Test cache error handling"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Test database error handling
        mock_session.query.side_effect = Exception("Database error")

        # Should handle errors gracefully and return None
        result = cache_manager.get(self.features, "test_model", {"param": "value"})
        assert result is None

        # Should still increment miss counter
        assert cache_manager._stats["misses"] == 1

    def test_cache_configuration_validation(self):
        """Test cache configuration validation"""
        # Test valid configuration
        config = PredictionConfig()
        config.prediction_cache_enabled = True
        config.prediction_cache_ttl = 60
        config.prediction_cache_max_size = 100

        # Should not raise any exceptions
        config.validate()

        # Test invalid TTL
        config.prediction_cache_ttl = -1
        with pytest.raises(ValueError):
            config.validate()

        # Test invalid max size
        config.prediction_cache_ttl = 60
        config.prediction_cache_max_size = 0
        with pytest.raises(ValueError):
            config.validate()

    def test_cache_hash_collision_resistance(self, mock_db_manager):
        """Test cache hash collision resistance"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager
        cache_manager = PredictionCacheManager(mock_db, ttl=60, max_size=100)

        # Generate many different feature sets and verify unique hashes
        hashes = set()
        for i in range(100):
            features = np.array([[float(i), float(i + 1), float(i + 2)]], dtype=np.float32)
            hash_val = cache_manager._generate_features_hash(features)
            hashes.add(hash_val)

        # All hashes should be unique (no collisions)
        assert len(hashes) == 100

    def test_cache_ttl_behavior(self, mock_db_manager):
        """Test cache TTL behavior"""
        mock_db, mock_session = mock_db_manager

        # Create cache manager with short TTL
        cache_manager = PredictionCacheManager(mock_db, ttl=1, max_size=100)

        # Mock cache entry that is not expired initially
        mock_entry = MagicMock()
        mock_entry.predicted_price = 100.5
        mock_entry.confidence = 0.8
        mock_entry.direction = 1
        mock_entry.access_count = 0
        mock_entry.expires_at = datetime.now(UTC) + timedelta(seconds=0.5)

        # Set up mock to return entry initially, then None after expiration
        mock_query_result = mock_session.query.return_value.filter.return_value.first
        mock_query_result.side_effect = [mock_entry, None]

        # Should find entry initially
        result = cache_manager.get(self.features, "test_model", {"param": "value"})
        assert result is not None

        # Wait for expiration (simulate time passing)
        time.sleep(1.1)

        # Should not find expired entry (mock returns None)
        result = cache_manager.get(self.features, "test_model", {"param": "value"})
        assert result is None
