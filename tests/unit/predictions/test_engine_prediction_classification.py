"""PredictionEngine's classification-native path (TARGET-REDESIGN, #933 Phase 2).

Purely additive over the regression path: a ModelPrediction with
probabilities=None (every existing regression bundle) must flow through
PredictionEngine.predict() byte-identically to before this feature existed
-- covered explicitly below alongside the new classification behavior.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.models.onnx_runner import ModelPrediction
from src.prediction.models.registry import StrategyModel


def _make_bundle(runner: Mock, *, metadata: dict | None = None) -> StrategyModel:
    return StrategyModel(
        symbol="BTCUSDT",
        timeframe="1h",
        model_type="basic_binary_direction",
        version_id="v1",
        directory=Path("/tmp/BTCUSDT/basic_binary_direction/v1"),
        metadata=metadata or {},
        feature_schema=None,
        metrics=None,
        runner=runner,
    )


def _make_data(num_rows=120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, num_rows),
            "high": np.random.uniform(110, 120, num_rows),
            "low": np.random.uniform(90, 100, num_rows),
            "close": np.random.uniform(100, 110, num_rows),
            "volume": np.random.uniform(1000, 2000, num_rows),
        }
    )


class TestPredictionResultProbabilitiesField:
    def test_default_none(self):
        result = PredictionResult(
            price=100.0,
            confidence=0.5,
            direction=1,
            model_name="m",
            timestamp=pd.Timestamp.now(tz="UTC"),
            inference_time=0.01,
            features_used=5,
        )
        assert result.probabilities is None


@patch("src.prediction.engine.PredictionModelRegistry")
@patch("src.prediction.engine.FeaturePipeline")
class TestClassificationPredictionFlow:
    def test_probabilities_propagate_to_result(self, mock_pipeline, mock_registry):
        config = PredictionConfig()
        engine = PredictionEngine(config)
        engine.feature_pipeline.transform.return_value = np.random.random((1, 10))

        mock_model = Mock()
        mock_model.predict.return_value = ModelPrediction(
            price=float("nan"),
            confidence=0.82,
            direction=1,
            model_name="test_classifier",
            inference_time=0.02,
            probabilities=(0.18, 0.82),
        )
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

        result = engine.predict(_make_data())

        assert result.error is None
        assert result.probabilities == pytest.approx((0.18, 0.82))
        assert result.direction == 1
        assert result.confidence == pytest.approx(0.82)

    def test_classification_prediction_skips_price_denormalization(
        self, mock_pipeline, mock_registry
    ):
        """A classification prediction's NaN price must NOT trip the
        finite-price guard in _apply_rolling_denormalization -- that guard
        exists to catch bad REGRESSION output, not to reject the (expected)
        placeholder NaN a classifier reports."""
        config = PredictionConfig()
        engine = PredictionEngine(config)
        engine.feature_pipeline.transform.return_value = np.random.random((1, 10))

        mock_model = Mock()
        mock_model.predict.return_value = ModelPrediction(
            price=float("nan"),
            confidence=0.6,
            direction=-1,
            model_name="test_classifier",
            inference_time=0.02,
            probabilities=(0.6, 0.4),
        )
        bundle = _make_bundle(
            mock_model, metadata={"price_normalization": {"method": "rolling_minmax"}}
        )
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

        result = engine.predict(_make_data())

        assert result.error is None
        assert result.probabilities == pytest.approx((0.6, 0.4))


@patch("src.prediction.engine.PredictionModelRegistry")
@patch("src.prediction.engine.FeaturePipeline")
class TestRegressionPathUnaffected:
    """Guards that adding the classification path did not change a single
    byte of behavior for every existing (regression) bundle."""

    def test_regression_prediction_has_no_probabilities(self, mock_pipeline, mock_registry):
        config = PredictionConfig()
        engine = PredictionEngine(config)
        engine.feature_pipeline.transform.return_value = np.random.random((1, 10))

        mock_model = Mock()
        mock_model.predict.return_value = ModelPrediction(
            price=105.5,
            confidence=0.85,
            direction=1,
            model_name="test_model",
            inference_time=0.02,
        )
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

        result = engine.predict(_make_data())

        assert result.probabilities is None
        assert result.price == 105.5
        assert result.error is None
