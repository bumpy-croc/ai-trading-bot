"""Tests covering batch prediction flows for PredictionEngine."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.models.onnx_runner import ModelPrediction
from src.prediction.models.registry import StrategyModel


def _make_bundle(
    runner: Mock,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    model_type: str = "basic",
    version: str = "v1",
    metadata: dict | None = None,
) -> StrategyModel:
    return StrategyModel(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        version_id=version,
        directory=Path(f"/tmp/{symbol}/{model_type}/{version}"),
        metadata=metadata or {},
        feature_schema=None,
        metrics=None,
        runner=runner,
    )


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
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

        data_batches = [self.create_test_data(), self.create_test_data(), self.create_test_data()]
        results = engine.predict_batch(data_batches)

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
        engine.model_registry.list_bundles.return_value = []

        data_batches = [self.create_test_data(), self.create_test_data()]
        results = engine.predict_batch(data_batches)

        assert len(results) == 2
        for result in results:
            assert result.error is not None
            assert "No prediction models available" in result.error

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_predict_batch_mixed_results(self, mock_pipeline, mock_registry):
        """Test batch prediction with some failures"""
        engine = PredictionEngine()

        call_count = [0]

        def mock_transform(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Feature extraction failed")
            return np.random.random((1, 10))

        engine.feature_pipeline.transform.side_effect = mock_transform

        mock_model = Mock()
        mock_prediction = ModelPrediction(
            price=105.5,
            confidence=0.85,
            direction=1,
            model_name="test_model",
            inference_time=0.02,
        )
        mock_model.predict.return_value = mock_prediction
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

        data_batches = [self.create_test_data(), self.create_test_data(), self.create_test_data()]
        results = engine.predict_batch(data_batches)

        assert len(results) == 3
        assert results[0].error is None
        assert results[1].error is not None
        assert results[2].error is None
