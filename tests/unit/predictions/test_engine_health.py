"""Tests covering health checks and validation in PredictionEngine."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine
from src.prediction.exceptions import InvalidInputError
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


class TestPredictionEngineHealthCheck:
    """Test PredictionEngine health check"""

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_health_check_healthy(self, mock_pipeline, mock_registry):
        """Test health check when all components are healthy"""
        engine = PredictionEngine()

        mock_features = np.random.random((120, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        mock_runner = Mock()
        mock_runner.model_path = "/path/to/model.onnx"
        bundle_one = _make_bundle(mock_runner, metadata={"name": "model1"})
        bundle_two = _make_bundle(Mock(), symbol="ETHUSDT", metadata={"name": "model2"})
        engine.model_registry.list_bundles.return_value = [bundle_one, bundle_two]
        engine.model_registry.get_default_bundle.return_value = bundle_one

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

        engine.feature_pipeline.transform.side_effect = Exception("Feature pipeline error")

        mock_runner = Mock()
        mock_runner.model_path = "/path/to/model.onnx"
        bundle = _make_bundle(mock_runner, metadata={"name": "model1"})
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

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

        engine._validate_input_data(valid_data)

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_validate_input_data_invalid(self, mock_pipeline, mock_registry):
        """Test input data validation with invalid data"""
        engine = PredictionEngine()

        with pytest.raises(InvalidInputError, match="Input data must be a pandas DataFrame"):
            engine._validate_input_data("not a dataframe")

        invalid_data = pd.DataFrame({"open": [100.0] * 120})
        with pytest.raises(InvalidInputError, match="Missing required columns"):
            engine._validate_input_data(invalid_data)

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
        assert len(version) > 5
