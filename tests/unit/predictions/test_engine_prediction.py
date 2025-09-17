"""Tests covering single prediction flows for PredictionEngine."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.models.onnx_runner import ModelPrediction


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
        config = PredictionConfig()
        engine = PredictionEngine(config)

        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=105.5,
            confidence=0.85,
            direction=1,
            model_name="test_model",
            inference_time=0.02,
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_default_model.return_value = mock_model

        data = self.create_test_data()
        result = engine.predict(data)

        assert isinstance(result, PredictionResult)
        assert result.price == 105.5
        assert result.confidence == 0.85
        assert result.direction == 1
        assert result.model_name == "test_model"
        assert result.features_used == 10
        assert result.error is None
        assert result.inference_time > 0

        engine.feature_pipeline.transform.assert_called_once_with(data, use_cache=True)
        mock_model.predict.assert_called_once_with(mock_features)

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_with_specific_model(self, mock_pipeline, mock_registry):
        """Test prediction with specific model name"""
        config = PredictionConfig()
        engine = PredictionEngine(config)

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

        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        data = self.create_test_data()
        result = engine.predict(data, model_name="specific_model")

        engine.model_registry.get_model.assert_called_once_with("specific_model")
        assert result.model_name == "specific_model"

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_invalid_input(self, mock_pipeline, mock_registry):
        """Test prediction with invalid input data"""
        engine = PredictionEngine()

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

        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102] * 40,
                "high": [102, 103, 104] * 40,
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
        config.max_prediction_latency = 0.001
        engine = PredictionEngine(config)

        def slow_transform(*args, **kwargs):
            import time

            time.sleep(0.05)
            return np.random.random((1, 10))

        engine.feature_pipeline.transform.side_effect = slow_transform

        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=100.0,
            confidence=0.8,
            direction=1,
            model_name="test_model",
            inference_time=0.01,
        )
        mock_model.predict.return_value = mock_prediction
        engine.model_registry.get_default_model.return_value = mock_model

        data = self.create_test_data()
        result = engine.predict(data)

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
        config.max_prediction_latency = 0.001
        engine = PredictionEngine(config)

        def slow_transform_with_error(*args, **kwargs):
            import time

            time.sleep(0.05)
            raise ValueError("Feature extraction failed")

        engine.feature_pipeline.transform.side_effect = slow_transform_with_error

        data = self.create_test_data()
        result = engine.predict(data)

        assert result.error is not None
        assert "Prediction timeout" in result.error
        assert "Feature extraction failed" in result.error
        assert result.inference_time > config.max_prediction_latency
        assert result.price == 0.0
        assert result.confidence == 0.0
        assert result.direction == 0
        assert result.metadata["error_type"] == "PredictionTimeoutError+FeatureExtractionError"
