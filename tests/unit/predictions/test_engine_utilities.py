"""Tests covering utility features of PredictionEngine."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from src.prediction.engine import PredictionEngine, PredictionResult


class TestPredictionEngineUtilities:
    """Test PredictionEngine utility methods"""

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_get_available_models(self, mock_pipeline, mock_registry):
        """Test getting available models"""
        engine = PredictionEngine()
        engine.model_registry.list_models.return_value = ["model1", "model2", "model3"]

        models = engine.get_available_models()
        assert models == ["model1", "model2", "model3"]

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_get_model_info(self, mock_pipeline, mock_registry):
        """Test getting model information"""
        engine = PredictionEngine()

        mock_model = Mock()
        mock_model.model_path = "/path/to/model.onnx"
        mock_model.model_metadata = {"version": "1.0", "type": "price"}
        engine.model_registry.get_model.return_value = mock_model

        info = engine.get_model_info("test_model")

        assert info["name"] == "test_model"
        assert info["path"] == "/path/to/model.onnx"
        assert info["metadata"] == {"version": "1.0", "type": "price"}
        assert info["loaded"] is True

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_get_model_info_not_found(self, mock_pipeline, mock_registry):
        """Test getting info for non-existent model"""
        engine = PredictionEngine()
        engine.model_registry.get_model.return_value = None

        info = engine.get_model_info("nonexistent")
        assert info == {}

    def test_get_performance_stats(self):
        """Test getting performance statistics"""
        engine = PredictionEngine()

        engine._prediction_count = 10
        engine._total_inference_time = 5.0
        engine._cache_hits = 7
        engine._cache_misses = 3

        stats = engine.get_performance_stats()

        assert stats["prediction_count"] == 10
        assert stats["total_inference_time"] == 5.0
        assert stats["cache_hits"] == 7
        assert stats["cache_misses"] == 3
        assert "cache_hit_rate" in stats

    def test_clear_caches(self):
        """Test clearing caches"""
        engine = PredictionEngine()

        mock_cache = MagicMock()
        engine.feature_pipeline.cache = mock_cache

        mock_cache_manager = MagicMock()
        engine.cache_manager = mock_cache_manager

        engine._cache_hits = 5
        engine._cache_misses = 3
        engine._feature_extraction_time = 1.5
        engine._total_feature_extraction_time = 10.0
        engine._feature_extraction_count = 7

        engine.clear_caches()

        mock_cache.clear.assert_called_once()
        mock_cache_manager.clear.assert_called_once()

        assert engine._cache_hits == 0
        assert engine._cache_misses == 0
        assert engine._feature_extraction_time == 0.0
        assert engine._total_feature_extraction_time == 0.0
        assert engine._feature_extraction_count == 0

    def test_reload_models(self):
        """Test reloading models"""
        engine = PredictionEngine()

        mock_registry = MagicMock()
        engine.model_registry = mock_registry

        engine.reload_models_and_clear_cache()

        mock_registry.reload_models.assert_called_once()

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_update_performance_stats(self, mock_pipeline, mock_registry):
        """Test performance statistics update"""
        engine = PredictionEngine()

        result = PredictionResult(
            price=100.0,
            confidence=0.8,
            direction=1,
            model_name="test",
            timestamp=datetime.now(timezone.utc),
            inference_time=0.1,
            features_used=10,
            cache_hit=True,
        )

        initial_count = engine._prediction_count
        initial_time = engine._total_inference_time
        initial_hits = engine._cache_hits

        engine._update_performance_stats(result)

        assert engine._prediction_count == initial_count + 1
        assert engine._total_inference_time == initial_time + 0.1
        assert engine._cache_hits == initial_hits + 1
