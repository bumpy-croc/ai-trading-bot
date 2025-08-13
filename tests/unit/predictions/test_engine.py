"""
Comprehensive tests for the PredictionEngine class.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.exceptions import (
    InvalidInputError,
)
from src.prediction.models.onnx_runner import ModelPrediction


class TestPredictionResult:
    """Test PredictionResult dataclass"""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult"""
        timestamp = datetime.now(timezone.utc)
        result = PredictionResult(
            price=100.5,
            confidence=0.85,
            direction=1,
            model_name="test_model",
            timestamp=timestamp,
            inference_time=0.05,
            features_used=10,
        )

        assert result.price == 100.5
        assert result.confidence == 0.85
        assert result.direction == 1
        assert result.model_name == "test_model"
        assert result.timestamp == timestamp
        assert result.inference_time == 0.05
        assert result.features_used == 10
        assert result.cache_hit is False
        assert result.error is None
        assert result.metadata == {}

    def test_prediction_result_with_error(self):
        """Test creating a PredictionResult with error"""
        result = PredictionResult(
            price=0.0,
            confidence=0.0,
            direction=0,
            model_name="test_model",
            timestamp=datetime.now(timezone.utc),
            inference_time=0.1,
            features_used=0,
            error="Test error message",
        )

        assert result.error == "Test error message"
        assert result.price == 0.0
        assert result.confidence == 0.0


class TestPredictionEngineInit:
    """Test PredictionEngine initialization"""

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_init_with_config(self, mock_pipeline, mock_registry):
        """Test initialization with provided config"""
        config = PredictionConfig()
        config.enable_sentiment = True
        config.enable_market_microstructure = True
        config.feature_cache_ttl = 600

        engine = PredictionEngine(config)

        assert engine.config == config
        mock_pipeline.assert_called_once_with(
            enable_sentiment=True, enable_market_microstructure=True, cache_ttl=600
        )
        mock_registry.assert_called_once_with(config)

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    @patch("src.prediction.config.PredictionConfig.from_config_manager")
    def test_init_without_config(self, mock_config, mock_pipeline, mock_registry):
        """Test initialization without provided config"""
        default_config = PredictionConfig()
        mock_config.return_value = default_config

        engine = PredictionEngine()

        mock_config.assert_called_once()
        assert engine.config == default_config
        mock_pipeline.assert_called_once()
        mock_registry.assert_called_once_with(default_config)


class TestPredictionEnginePredict:
    """Test PredictionEngine predict method"""

    def create_test_data(self, num_rows=120):
        """Create test market data"""
        return pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, num_rows),
                "high": np.random.uniform(110, 120, num_rows),
                "low": np.random.uniform(90, 100, num_rows),
                "close": np.random.uniform(100, 110, num_rows),
                "volume": np.random.uniform(1000, 2000, num_rows),
            }
        )

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_success(self, mock_pipeline, mock_registry):
        """Test successful prediction"""
        # Setup mocks
        config = PredictionConfig()
        engine = PredictionEngine(config)

        # Mock feature extraction
        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        # Mock model prediction
        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=105.5, confidence=0.85, direction=1, model_name="test_model", inference_time=0.02
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_default_model.return_value = mock_model

        # Test prediction
        data = self.create_test_data()
        result = engine.predict(data)

        # Verify result
        assert isinstance(result, PredictionResult)
        assert result.price == 105.5
        assert result.confidence == 0.85
        assert result.direction == 1
        assert result.model_name == "test_model"
        assert result.features_used == 10
        assert result.error is None
        assert result.inference_time > 0

        # Verify method calls
        engine.feature_pipeline.transform.assert_called_once_with(data, use_cache=True)
        mock_model.predict.assert_called_once_with(mock_features)

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_with_specific_model(self, mock_pipeline, mock_registry):
        """Test prediction with specific model name"""
        config = PredictionConfig()
        engine = PredictionEngine(config)

        # Mock model registry
        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=100.0,
            confidence=0.7,
            direction=-1,
            model_name="specific_model",
            inference_time=0.03,
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_model.return_value = mock_model

        # Mock feature extraction
        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        # Test prediction with specific model
        data = self.create_test_data()
        result = engine.predict(data, model_name="specific_model")

        # Verify model selection
        engine.model_registry.get_model.assert_called_once_with("specific_model")
        assert result.model_name == "specific_model"

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_invalid_input(self, mock_pipeline, mock_registry):
        """Test prediction with invalid input data"""
        engine = PredictionEngine()

        # Test with insufficient data
        small_data = self.create_test_data(num_rows=50)
        result = engine.predict(small_data)

        assert result.error is not None
        assert "Insufficient data" in result.error
        assert result.price == 0.0
        assert result.confidence == 0.0

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_missing_columns(self, mock_pipeline, mock_registry):
        """Test prediction with missing required columns"""
        engine = PredictionEngine()

        # Test with missing columns
        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102] * 40,
                "high": [102, 103, 104] * 40,
                # Missing low, close, volume
            }
        )

        result = engine.predict(invalid_data)

        assert result.error is not None
        assert "Missing required columns" in result.error

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_model_not_found(self, mock_pipeline, mock_registry):
        """Test prediction when model is not found"""
        engine = PredictionEngine()
        engine.model_registry.get_model.return_value = None

        # Mock feature extraction to return numpy array
        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        data = self.create_test_data()
        result = engine.predict(data, model_name="nonexistent_model")

        assert result.error is not None
        assert "not found" in result.error

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_feature_extraction_error(self, mock_pipeline, mock_registry):
        """Test prediction when feature extraction fails"""
        engine = PredictionEngine()
        engine.feature_pipeline.transform.side_effect = Exception("Feature extraction failed")

        data = self.create_test_data()
        result = engine.predict(data)

        assert result.error is not None
        assert "Feature extraction failed" in result.error

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_timeout(self, mock_pipeline, mock_registry):
        """Test prediction timeout logic"""
        config = PredictionConfig()
        config.max_prediction_latency = 0.001  # Very short timeout (1ms)
        engine = PredictionEngine(config)

        # Mock slow feature extraction that takes longer than timeout
        def slow_transform(*args, **kwargs):
            import time

            time.sleep(0.05)  # Sleep much longer than timeout (50ms > 1ms)
            return np.random.random((1, 10))

        engine.feature_pipeline.transform.side_effect = slow_transform

        # Mock model
        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=100.0, confidence=0.8, direction=1, model_name="test_model", inference_time=0.01
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_default_model.return_value = mock_model

        data = self.create_test_data()
        result = engine.predict(data)

        # Predictions that exceed max_prediction_latency should return an error result,
        # even if the prediction completed successfully.
        assert result.error is not None
        assert "Prediction timeout" in result.error
        assert result.inference_time > config.max_prediction_latency
        assert result.price == 0.0
        assert result.confidence == 0.0
        assert result.direction == 0
        assert result.metadata["error_type"] == "PredictionTimeoutError"

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_timeout_with_original_error(self, mock_pipeline, mock_registry):
        """Test timeout logic preserves original error when both timeout and exception occur"""
        config = PredictionConfig()
        config.max_prediction_latency = 0.001  # Very short timeout (1ms)
        engine = PredictionEngine(config)

        # Mock slow feature extraction that raises an exception and takes longer than timeout
        def slow_transform_with_error(*args, **kwargs):
            import time

            time.sleep(0.05)  # Sleep longer than timeout (50ms > 1ms)
            raise ValueError("Feature extraction failed")

        engine.feature_pipeline.transform.side_effect = slow_transform_with_error

        data = self.create_test_data()
        result = engine.predict(data)

        # Test that when both a timeout and an exception occur, the error message includes both timeout and original error information
        assert result.error is not None
        assert "Prediction timeout" in result.error
        assert "Feature extraction failed" in result.error
        assert result.inference_time > config.max_prediction_latency
        assert result.price == 0.0
        assert result.confidence == 0.0
        assert result.direction == 0
        assert result.metadata["error_type"] == "PredictionTimeoutError+FeatureExtractionError"


class TestPredictionEngineBatch:
    """Test PredictionEngine batch prediction"""

    def create_test_data(self, num_rows=120):
        """Create test market data"""
        return pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, num_rows),
                "high": np.random.uniform(110, 120, num_rows),
                "low": np.random.uniform(90, 100, num_rows),
                "close": np.random.uniform(100, 110, num_rows),
                "volume": np.random.uniform(1000, 2000, num_rows),
            }
        )

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_batch_success(self, mock_pipeline, mock_registry):
        """Test successful batch prediction"""
        engine = PredictionEngine()

        # Mock feature extraction
        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        # Mock model prediction
        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=105.5, confidence=0.85, direction=1, model_name="test_model", inference_time=0.02
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_default_model.return_value = mock_model

        # Test batch prediction
        data_batches = [self.create_test_data(), self.create_test_data(), self.create_test_data()]
        results = engine.predict_batch(data_batches)

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, PredictionResult)
            assert result.price == 105.5
            assert result.confidence == 0.85
            assert result.direction == 1
            assert result.metadata["batch_index"] == i
            assert result.metadata["batch_size"] == 3

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_batch_model_error(self, mock_pipeline, mock_registry):
        """Test batch prediction when model loading fails"""
        engine = PredictionEngine()
        engine.model_registry.get_default_model.side_effect = Exception("Model loading failed")

        data_batches = [self.create_test_data(), self.create_test_data()]
        results = engine.predict_batch(data_batches)

        # All results should be error results
        assert len(results) == 2
        for result in results:
            assert result.error is not None
            assert "Model loading failed" in result.error

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_batch_mixed_results(self, mock_pipeline, mock_registry):
        """Test batch prediction with some failures"""
        engine = PredictionEngine()

        # Mock feature extraction - fail on second call
        call_count = [0]

        def mock_transform(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Feature extraction failed")
            return np.random.random((1, 10))

        engine.feature_pipeline.transform.side_effect = mock_transform

        # Mock model
        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=105.5, confidence=0.85, direction=1, model_name="test_model", inference_time=0.02
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_default_model.return_value = mock_model

        data_batches = [self.create_test_data(), self.create_test_data(), self.create_test_data()]
        results = engine.predict_batch(data_batches)

        # First and third should succeed, second should fail
        assert len(results) == 3
        assert results[0].error is None
        assert results[1].error is not None
        assert results[2].error is None


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

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_get_performance_stats(self, mock_pipeline, mock_registry):
        """Test getting performance statistics"""
        engine = PredictionEngine()

        # Simulate some predictions
        engine._prediction_count = 10
        engine._total_inference_time = 1.0
        engine._cache_hits = 7
        engine._cache_misses = 3
        engine.model_registry.list_models.return_value = ["model1", "model2"]

        stats = engine.get_performance_stats()

        assert stats["total_predictions"] == 10
        assert stats["avg_inference_time"] == 0.1
        assert stats["cache_hit_rate"] == 0.7
        assert stats["cache_hits"] == 7
        assert stats["cache_misses"] == 3
        assert stats["available_models"] == 2

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_clear_caches(self, mock_pipeline, mock_registry):
        """Test clearing caches"""
        engine = PredictionEngine()
        engine._cache_hits = 5
        engine._cache_misses = 3

        engine.clear_caches()

        engine.feature_pipeline.clear_cache.assert_called_once()
        assert engine._cache_hits == 0
        assert engine._cache_misses == 0

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_reload_models(self, mock_pipeline, mock_registry):
        """Test reloading models"""
        engine = PredictionEngine()

        engine.reload_models()

        engine.model_registry.reload_models.assert_called_once()


class TestPredictionEngineHealthCheck:
    """Test PredictionEngine health check"""

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_health_check_healthy(self, mock_pipeline, mock_registry):
        """Test health check when all components are healthy"""
        engine = PredictionEngine()

        # Mock feature pipeline health
        mock_features = np.random.random((120, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        # Mock model registry health
        engine.model_registry.list_models.return_value = ["model1", "model2"]
        mock_model = Mock()
        mock_model.model_path = "/path/to/model.onnx"
        engine.model_registry.get_default_model.return_value = mock_model

        health = engine.health_check()

        assert health["status"] == "healthy"
        assert health["components"]["feature_pipeline"]["status"] == "healthy"
        assert health["components"]["model_registry"]["status"] == "healthy"
        assert health["components"]["configuration"]["status"] == "healthy"
        assert health["components"]["feature_pipeline"]["test_features_count"] == 10
        assert health["components"]["model_registry"]["available_models"] == 2

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_health_check_degraded(self, mock_pipeline, mock_registry):
        """Test health check when some components have errors"""
        engine = PredictionEngine()

        # Mock feature pipeline failure
        engine.feature_pipeline.transform.side_effect = Exception("Feature pipeline error")

        # Mock model registry health
        engine.model_registry.list_models.return_value = ["model1"]
        mock_model = Mock()
        mock_model.model_path = "/path/to/model.onnx"
        engine.model_registry.get_default_model.return_value = mock_model

        health = engine.health_check()

        assert health["status"] == "degraded"
        assert health["components"]["feature_pipeline"]["status"] == "error"
        assert health["components"]["model_registry"]["status"] == "healthy"
        assert health["components"]["configuration"]["status"] == "healthy"
        assert "Feature pipeline error" in health["components"]["feature_pipeline"]["error"]

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_validate_input_data_valid(self, mock_pipeline, mock_registry):
        """Test input data validation with valid data"""
        engine = PredictionEngine()

        valid_data = pd.DataFrame(
            {
                "open": [100.0] * 120,
                "high": [102.0] * 120,
                "low": [99.0] * 120,
                "close": [101.0] * 120,
                "volume": [1000.0] * 120,
            }
        )

        # Should not raise any exception
        engine._validate_input_data(valid_data)

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_validate_input_data_invalid(self, mock_pipeline, mock_registry):
        """Test input data validation with invalid data"""
        engine = PredictionEngine()

        # Test with non-DataFrame
        with pytest.raises(InvalidInputError, match="Input data must be a pandas DataFrame"):
            engine._validate_input_data("not a dataframe")

        # Test with missing columns
        invalid_data = pd.DataFrame({"open": [100.0] * 120})
        with pytest.raises(InvalidInputError, match="Missing required columns"):
            engine._validate_input_data(invalid_data)

        # Test with insufficient data
        small_data = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [102.0] * 50,
                "low": [99.0] * 50,
                "close": [101.0] * 50,
                "volume": [1000.0] * 50,
            }
        )
        with pytest.raises(InvalidInputError, match="Insufficient data"):
            engine._validate_input_data(small_data)

        # Test with null values
        null_data = pd.DataFrame(
            {
                "open": [100.0, None] * 60,
                "high": [102.0, 103.0] * 60,
                "low": [99.0, 100.0] * 60,
                "close": [101.0, 102.0] * 60,
                "volume": [1000.0, 1100.0] * 60,
            }
        )
        with pytest.raises(InvalidInputError, match="Input data contains null values"):
            engine._validate_input_data(null_data)

        # Test with non-positive values
        negative_data = pd.DataFrame(
            {
                "open": [100.0, -1.0] * 60,
                "high": [102.0, 103.0] * 60,
                "low": [99.0, 100.0] * 60,
                "close": [101.0, 102.0] * 60,
                "volume": [1000.0, 1100.0] * 60,
            }
        )
        with pytest.raises(InvalidInputError, match="Input data contains non-positive values"):
            engine._validate_input_data(negative_data)

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_get_config_version(self, mock_pipeline, mock_registry):
        """Test config version generation"""
        config = PredictionConfig()
        engine = PredictionEngine(config)

        version = engine._get_config_version()
        assert version.startswith("v1.0-")
        assert len(version) > 5  # Should have hash appended

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
