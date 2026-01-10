"""
Unit tests for prediction caching functionality.

This module tests the prediction cache manager, cache integration with OnnxRunner,
and cache management features.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
from sqlalchemy.orm import Session

from src.database.models import PredictionCache
from src.prediction.config import PredictionConfig
from src.prediction.models.onnx_runner import OnnxRunner
from src.prediction.utils.caching import PredictionCacheManager


class TestPredictionCacheManager:
    """Test the PredictionCacheManager class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_db_manager = MagicMock()
        self.mock_session = MagicMock(spec=Session)
        self.mock_db_manager.get_session.return_value.__enter__.return_value = self.mock_session
        self.mock_db_manager.get_session.return_value.__exit__.return_value = None

        self.cache_manager = PredictionCacheManager(self.mock_db_manager, ttl=60, max_size=100)

        # Sample test data
        self.features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        self.model_name = "test_model"
        self.config = {"param1": "value1", "param2": "value2"}

    def test_generate_features_hash(self):
        """Test feature hash generation"""
        hash1 = self.cache_manager._generate_features_hash(self.features)
        hash2 = self.cache_manager._generate_features_hash(self.features)

        # Same features should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length

        # Different features should produce different hash
        different_features = np.array([[1.0, 2.0, 3.1]], dtype=np.float32)
        hash3 = self.cache_manager._generate_features_hash(different_features)
        assert hash1 != hash3

    def test_generate_config_hash(self):
        """Test configuration hash generation"""
        hash1 = self.cache_manager._generate_config_hash(self.model_name, self.config)
        hash2 = self.cache_manager._generate_config_hash(self.model_name, self.config)

        # Same config should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64

        # Different config should produce different hash
        different_config = {"param1": "value1", "param2": "value3"}
        hash3 = self.cache_manager._generate_config_hash(self.model_name, different_config)
        assert hash1 != hash3

    def test_generate_cache_key(self):
        """Test cache key generation"""
        key1 = self.cache_manager._generate_cache_key(self.features, self.model_name, self.config)
        key2 = self.cache_manager._generate_cache_key(self.features, self.model_name, self.config)

        # Same inputs should produce same key
        assert key1 == key2

        # Different inputs should produce different keys
        different_features = np.array([[1.0, 2.0, 3.1]], dtype=np.float32)
        key3 = self.cache_manager._generate_cache_key(
            different_features, self.model_name, self.config
        )
        assert key1 != key3

    def test_get_cache_hit(self):
        """Test successful cache retrieval"""
        # Mock cache entry
        mock_entry = MagicMock()
        mock_entry.predicted_price = 100.5
        mock_entry.confidence = 0.8
        mock_entry.direction = 1
        mock_entry.access_count = 5
        mock_entry.expires_at = datetime.now(UTC) + timedelta(seconds=30)

        self.mock_session.query.return_value.filter.return_value.first.return_value = mock_entry

        result = self.cache_manager.get(self.features, self.model_name, self.config)

        assert result is not None
        assert result["price"] == 100.5
        assert result["confidence"] == 0.8
        assert result["direction"] == 1
        assert result["cache_hit"] is True
        assert result["access_count"] == 6  # Incremented

    def test_get_cache_miss(self):
        """Test cache miss scenario"""
        self.mock_session.query.return_value.filter.return_value.first.return_value = None

        result = self.cache_manager.get(self.features, self.model_name, self.config)

        assert result is None
        assert self.cache_manager._stats["misses"] == 1

    def test_get_cache_expired(self):
        """Test expired cache entry"""
        # Mock expired cache entry - should not be returned by query due to expiration filter
        self.mock_session.query.return_value.filter.return_value.first.return_value = None

        result = self.cache_manager.get(self.features, self.model_name, self.config)

        # Should return None for expired entry
        assert result is None
        assert self.cache_manager._stats["misses"] == 1

    def test_set_cache_new_entry(self):
        """Test setting new cache entry"""
        self.mock_session.query.return_value.filter.return_value.first.return_value = None

        self.cache_manager.set(self.features, self.model_name, self.config, 100.5, 0.8, 1)

        # Verify session.add was called with new entry
        self.mock_session.add.assert_called_once()
        added_entry = self.mock_session.add.call_args[0][0]
        assert isinstance(added_entry, PredictionCache)
        assert added_entry.predicted_price == 100.5
        assert added_entry.confidence == 0.8
        assert added_entry.direction == 1

    def test_set_cache_update_existing(self):
        """Test updating existing cache entry"""
        # Mock existing entry
        mock_entry = MagicMock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = mock_entry

        self.cache_manager.set(self.features, self.model_name, self.config, 100.5, 0.8, 1)

        # Verify existing entry was updated
        assert mock_entry.predicted_price == 100.5
        assert mock_entry.confidence == 0.8
        assert mock_entry.direction == 1

    def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        self.mock_session.query.return_value.filter.return_value.delete.return_value = 5

        result = self.cache_manager._cleanup_expired(self.mock_session)

        assert result == 5
        assert self.cache_manager._stats["expired_cleanups"] == 5

    def test_enforce_size_limit(self):
        """Test size limit enforcement"""
        # Create a cache manager with small size limit
        cache_manager = PredictionCacheManager(self.mock_db_manager, ttl=60, max_size=100)

        # Test when current count is within limit (should return 0)
        self.mock_session.query.return_value.count.return_value = 50

        result = cache_manager._enforce_size_limit(self.mock_session)

        # Should return 0 when within limit
        assert result == 0
        assert cache_manager._stats["evictions"] == 0

    def test_invalidate_model(self):
        """Test model cache invalidation"""
        self.mock_session.query.return_value.filter.return_value.delete.return_value = 10

        result = self.cache_manager.invalidate_model("test_model")

        assert result == 10

    def test_invalidate_config(self):
        """Test configuration cache invalidation"""
        self.mock_session.query.return_value.filter.return_value.delete.return_value = 5

        result = self.cache_manager.invalidate_config(self.model_name, self.config)

        assert result == 5

    def test_clear_cache(self):
        """Test cache clearing"""
        self.mock_session.query.return_value.delete.return_value = 25

        result = self.cache_manager.clear()

        assert result == 25

    def test_get_stats(self):
        """Test cache statistics retrieval"""
        self.mock_session.query.return_value.count.return_value = 50
        self.mock_session.query.return_value.filter.return_value.count.return_value = 5

        stats = self.cache_manager.get_stats()

        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert "hit_rate" in stats
        assert "total_requests" in stats
        assert stats["total_entries"] == 50
        assert stats["expired_entries"] == 5


class TestOnnxRunnerCaching:
    """Test OnnxRunner integration with caching"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = PredictionConfig()
        self.config.prediction_cache_enabled = True
        self.config.prediction_cache_ttl = 60
        self.config.prediction_cache_max_size = 100

        self.mock_cache_manager = MagicMock()
        self.features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.providers_patcher = patch(
            "src.prediction.models.onnx_runner.get_preferred_providers",
            return_value=["CPUExecutionProvider"],
        )
        self.providers_patcher.start()

    def teardown_method(self):
        """Stop patched providers"""
        self.providers_patcher.stop()

    @patch("src.prediction.models.onnx_runner.ort.InferenceSession")
    def test_predict_with_cache_hit(self, mock_inference_session):
        """Test prediction with cache hit"""
        # Mock cache hit
        cache_result = {"price": 100.5, "confidence": 0.8, "direction": 1, "cache_hit": True}
        self.mock_cache_manager.get.return_value = cache_result

        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_inference_session.return_value = mock_session

        # Use proper path format instead of filename string
        model_path = "/tmp/test_model.onnx"
        runner = OnnxRunner(model_path, self.config, self.mock_cache_manager)

        # Mock model metadata
        runner.model_metadata = {"normalization_params": {}}

        result = runner.predict(self.features)

        # Should return cached result without running inference
        assert result.price == 100.5
        assert result.confidence == 0.8
        assert result.direction == 1
        # model_name should be basename of the path
        assert result.model_name == "test_model.onnx"

        # Verify cache was checked
        self.mock_cache_manager.get.assert_called_once()

    @patch("src.prediction.models.onnx_runner.ort.InferenceSession")
    def test_predict_with_cache_miss(self, mock_inference_session):
        """Test prediction with cache miss"""
        # Mock cache miss
        self.mock_cache_manager.get.return_value = None

        # Mock ONNX session and inference
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.5]])]  # Mock output
        mock_inference_session.return_value = mock_session

        # Use proper path format instead of filename string
        model_path = "/tmp/test_model.onnx"
        runner = OnnxRunner(model_path, self.config, self.mock_cache_manager)

        # Mock model metadata
        runner.model_metadata = {"normalization_params": {}}

        result = runner.predict(self.features)

        # Should run inference and cache result
        assert result.price == 0.5
        # model_name should be basename of the path
        assert result.model_name == "test_model.onnx"

        # Verify cache was checked and result was cached
        self.mock_cache_manager.get.assert_called_once()
        self.mock_cache_manager.set.assert_called_once()

    @patch("src.prediction.models.onnx_runner.ort.InferenceSession")
    def test_predict_without_cache(self, mock_inference_session):
        """Test prediction without cache manager"""
        # Mock ONNX session and inference
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.5]])]  # Mock output
        mock_inference_session.return_value = mock_session

        # Use proper path format instead of filename string
        model_path = "/tmp/test_model.onnx"
        runner = OnnxRunner(model_path, self.config, cache_manager=None)

        # Mock model metadata
        runner.model_metadata = {"normalization_params": {}}

        result = runner.predict(self.features)

        # Should run inference without caching
        assert result.price == 0.5
        # model_name should be basename of the path
        assert result.model_name == "test_model.onnx"

    def test_cache_disabled(self):
        """Test prediction when cache is disabled"""
        self.config.prediction_cache_enabled = False

        with patch(
            "src.prediction.models.onnx_runner.ort.InferenceSession"
        ) as mock_inference_session:
            # Mock ONNX session and inference
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_session.run.return_value = [np.array([[0.5]])]  # Mock output
            mock_inference_session.return_value = mock_session

            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config, self.mock_cache_manager)

            # Mock model metadata
            runner.model_metadata = {"normalization_params": {}}

            result = runner.predict(self.features)

            # Should run inference without checking cache
            assert result.price == 0.5
            # model_name should be basename of the path
            assert result.model_name == "test_model.onnx"
            self.mock_cache_manager.get.assert_not_called()
            self.mock_cache_manager.set.assert_not_called()


class TestPredictionEngineCaching:
    """Test PredictionEngine integration with caching"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = PredictionConfig()
        self.config.prediction_cache_enabled = True
        self.config.prediction_cache_ttl = 60
        self.config.prediction_cache_max_size = 100

    @patch("src.prediction.engine.PredictionCacheManager")
    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_engine_with_cache_manager(
        self, mock_feature_pipeline, mock_model_registry, mock_cache_manager_class
    ):
        """Test engine initialization with cache manager"""
        mock_db_manager = MagicMock()
        mock_cache_manager = MagicMock()
        mock_cache_manager_class.return_value = mock_cache_manager

        from src.prediction.engine import PredictionEngine

        PredictionEngine(self.config, mock_db_manager)

        # Verify cache manager was created
        mock_cache_manager_class.assert_called_once_with(
            mock_db_manager,
            ttl=self.config.prediction_cache_ttl,
            max_size=self.config.prediction_cache_max_size,
        )

        # Verify model registry was created with cache manager
        mock_model_registry.assert_called_once_with(self.config, mock_cache_manager)

    @patch("src.prediction.engine.PredictionCacheManager")
    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_engine_without_database_manager(
        self, mock_feature_pipeline, mock_model_registry, mock_cache_manager_class
    ):
        """Test engine initialization without database manager"""
        from src.prediction.engine import PredictionEngine

        PredictionEngine(self.config, database_manager=None)

        # Verify cache manager was not created
        mock_cache_manager_class.assert_not_called()

        # Verify model registry was created without cache manager
        mock_model_registry.assert_called_once_with(self.config, None)

    @patch("src.prediction.engine.PredictionCacheManager")
    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_cache_disabled(
        self, mock_feature_pipeline, mock_model_registry, mock_cache_manager_class
    ):
        """Test engine initialization with cache disabled"""
        self.config.prediction_cache_enabled = False
        mock_db_manager = MagicMock()

        from src.prediction.engine import PredictionEngine

        PredictionEngine(self.config, mock_db_manager)

        # Verify cache manager was not created
        mock_cache_manager_class.assert_not_called()

        # Verify model registry was created without cache manager
        mock_model_registry.assert_called_once_with(self.config, None)

    @patch("src.prediction.engine.PredictionCacheManager")
    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_cache_management_methods(
        self, mock_feature_pipeline, mock_model_registry, mock_cache_manager_class
    ):
        """Test cache management methods"""
        mock_db_manager = MagicMock()
        mock_cache_manager = MagicMock()
        mock_cache_manager_class.return_value = mock_cache_manager
        mock_model_registry_instance = MagicMock()
        mock_model_registry.return_value = mock_model_registry_instance

        from src.prediction.engine import PredictionEngine

        engine = PredictionEngine(self.config, mock_db_manager)

        # Test clear_caches
        engine.clear_caches()
        mock_cache_manager.clear.assert_called_once()

        # Test get_cache_stats
        engine.get_cache_stats()
        mock_cache_manager.get_stats.assert_called_once()

        # Test invalidate_model_cache
        engine.invalidate_model_cache("test_model")
        mock_model_registry_instance.invalidate_cache.assert_called_once_with("test_model")

        # Test reload_models_and_clear_cache
        engine.reload_models_and_clear_cache()
        mock_cache_manager.clear.assert_called()
        mock_model_registry_instance.reload_models.assert_called_once()
