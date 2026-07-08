"""Tests covering single prediction flows for PredictionEngine."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.exceptions import ModelInferenceError
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
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

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
        assert mock_model.predict.call_count == 1
        np.testing.assert_allclose(
            mock_model.predict.call_args.args[0],
            mock_features.astype(np.float32),
        )

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
        default_bundle = _make_bundle(Mock(), version="default")
        specific_bundle = _make_bundle(
            mock_model,
            symbol="ETHUSDT",
            model_type="specific",
            version="v1",
        )
        engine.model_registry.list_bundles.return_value = [default_bundle, specific_bundle]
        engine.model_registry.get_default_bundle.return_value = default_bundle

        mock_features = np.random.random((1, 10))
        engine.feature_pipeline.transform.return_value = mock_features

        data = self.create_test_data()
        result = engine.predict(data, model_name=specific_bundle.key)

        assert mock_model.predict.call_count == 1
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
        bundle = _make_bundle(Mock())
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

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
    def test_slow_prediction_is_returned_intact(self, mock_pipeline, mock_registry):
        """A completed prediction is never substituted because it was slow.

        max_prediction_latency is a latency-ALERTING budget; exceeding it must
        not zero out a successful prediction (that substitution made backtest
        results depend on CPU load).
        """
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
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle

        data = self.create_test_data()
        result = engine.predict(data)

        assert result.error is None
        assert result.price == 100.0
        assert result.confidence == 0.8
        assert result.direction == 1

    @patch("src.prediction.engine.PredictionModelRegistry")
    @patch("src.prediction.engine.FeaturePipeline")
    def test_slow_failing_prediction_keeps_original_error(self, mock_pipeline, mock_registry):
        """Errors are reported as-is; latency no longer relabels them as timeouts."""
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
        assert "Feature extraction failed" in result.error
        assert "Prediction timeout" not in result.error
        assert result.price == 0.0
        assert result.confidence == 0.0
        assert result.direction == 0
        assert result.metadata["error_type"] == "FeatureExtractionError"


class TestDeterministicInference:
    """Backtest/research context: inference must never race a wall clock."""

    def create_test_data(self, num_rows=120):
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "open": rng.uniform(100, 110, num_rows),
                "high": rng.uniform(110, 120, num_rows),
                "low": rng.uniform(90, 100, num_rows),
                "close": rng.uniform(100, 110, num_rows),
                "volume": rng.uniform(1000, 2000, num_rows),
            }
        )

    def _make_engine_with_slow_model(self, config: PredictionConfig, sleep_seconds: float):
        with (
            patch("src.prediction.engine.PredictionModelRegistry"),
            patch("src.prediction.engine.FeaturePipeline"),
        ):
            engine = PredictionEngine(config)

        engine.feature_pipeline.transform.return_value = np.random.random((1, 10))

        mock_model = Mock()

        def slow_predict(features):
            import time

            time.sleep(sleep_seconds)
            return ModelPrediction(
                price=105.5,
                confidence=0.85,
                direction=1,
                model_name="slow_model",
                inference_time=sleep_seconds,
            )

        mock_model.predict.side_effect = slow_predict
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle
        return engine

    def test_slow_inference_never_times_out_in_deterministic_context(self):
        """Inference slower than every latency budget still yields the real
        prediction — identically on repeated calls (determinism guarantee)."""
        config = PredictionConfig()
        config.max_prediction_latency = 0.001  # tighter than the sleep below
        config.live_inference_timeout = 0.01  # would abort if (wrongly) applied
        engine = self._make_engine_with_slow_model(config, sleep_seconds=0.15)

        data = self.create_test_data()
        first = engine.predict(data)
        second = engine.predict(data)

        for result in (first, second):
            assert result.error is None
            assert result.price == 105.5
            assert result.direction == 1
        assert (first.price, first.confidence, first.direction) == (
            second.price,
            second.confidence,
            second.direction,
        )
        assert engine.get_performance_stats()["inference_timeouts"] == 0

    def test_deterministic_context_does_not_log_latency_alerts(self, caplog):
        """Wall-clock noise must not leak into backtest logs as warnings."""
        import logging

        config = PredictionConfig()
        config.max_prediction_latency = 0.001
        engine = self._make_engine_with_slow_model(config, sleep_seconds=0.05)

        with caplog.at_level(logging.WARNING, logger="src.prediction.engine"):
            result = engine.predict(self.create_test_data())

        assert result.error is None
        assert not [r for r in caplog.records if "latency" in r.message.lower()]


class TestLiveInferenceTimeoutAccounting:
    """Live context: bounded latency budget with loud, attributable accounting."""

    def create_test_data(self, num_rows=120):
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "open": rng.uniform(100, 110, num_rows),
                "high": rng.uniform(110, 120, num_rows),
                "low": rng.uniform(90, 100, num_rows),
                "close": rng.uniform(100, 110, num_rows),
                "volume": rng.uniform(1000, 2000, num_rows),
            }
        )

    def _make_engine_with_slow_model(self, config: PredictionConfig, sleep_seconds: float):
        with (
            patch("src.prediction.engine.PredictionModelRegistry"),
            patch("src.prediction.engine.FeaturePipeline"),
        ):
            engine = PredictionEngine(config)

        engine.feature_pipeline.transform.return_value = np.random.random((1, 10))

        mock_model = Mock()

        def slow_predict(features):
            import time

            time.sleep(sleep_seconds)
            return ModelPrediction(
                price=105.5,
                confidence=0.85,
                direction=1,
                model_name="slow_model",
                inference_time=sleep_seconds,
            )

        mock_model.predict.side_effect = slow_predict
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle
        return engine

    def test_live_timeout_is_loud_and_stamped(self, caplog):
        import logging

        from src.prediction.inference_context import InferenceContext, set_inference_context

        set_inference_context(InferenceContext.LIVE)
        config = PredictionConfig()
        config.live_inference_timeout = 0.05
        engine = self._make_engine_with_slow_model(config, sleep_seconds=0.5)

        with caplog.at_level(logging.WARNING, logger="src.prediction.engine"):
            result = engine.predict(self.create_test_data())

        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.metadata["timed_out"] is True
        assert result.price == 0.0
        assert result.confidence == 0.0
        assert result.direction == 0
        assert engine.get_performance_stats()["inference_timeouts"] == 1
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("timeout" in m.lower() for m in warning_messages)

    def test_live_fast_prediction_within_budget_succeeds(self):
        from src.prediction.inference_context import InferenceContext, set_inference_context

        set_inference_context(InferenceContext.LIVE)
        config = PredictionConfig()
        config.live_inference_timeout = 5.0
        engine = self._make_engine_with_slow_model(config, sleep_seconds=0.0)

        result = engine.predict(self.create_test_data())

        assert result.error is None
        assert result.price == 105.5
        assert engine.get_performance_stats()["inference_timeouts"] == 0

    def test_live_latency_alert_warns_without_substitution(self, caplog):
        """Exceeding the alerting budget logs a WARNING but returns the result."""
        import logging

        from src.prediction.inference_context import InferenceContext, set_inference_context

        set_inference_context(InferenceContext.LIVE)
        config = PredictionConfig()
        config.max_prediction_latency = 0.001
        config.live_inference_timeout = 5.0
        engine = self._make_engine_with_slow_model(config, sleep_seconds=0.05)

        with caplog.at_level(logging.WARNING, logger="src.prediction.engine"):
            result = engine.predict(self.create_test_data())

        assert result.error is None
        assert result.price == 105.5
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("latency" in m.lower() for m in warning_messages)


class TestApplyRollingDenormalization:
    """Tests for _apply_rolling_denormalization financial calculation."""

    def _make_engine(self):
        """Create engine with mocked registry/pipeline."""
        with (
            patch("src.prediction.engine.PredictionModelRegistry"),
            patch("src.prediction.engine.FeaturePipeline"),
        ):
            return PredictionEngine(PredictionConfig())

    def _make_input_data(self, close_values):
        """Create minimal input DataFrame with close prices."""
        return pd.DataFrame({"close": close_values})

    def test_valid_denormalization_math(self):
        """Verify inverse MinMax formula: price = normalized * (max - min) + min."""
        engine = self._make_engine()
        metadata = {"price_normalization": {"method": "rolling_minmax", "target_feature": "close"}}
        bundle = _make_bundle(Mock(), metadata=metadata)
        input_data = self._make_input_data([100.0, 150.0, 200.0])

        # normalized=0.5 with min=100, max=200 should yield 150
        result = engine._apply_rolling_denormalization(0.5, bundle, input_data)
        assert result == pytest.approx(150.0)

        # normalized=0.0 yields min
        result = engine._apply_rolling_denormalization(0.0, bundle, input_data)
        assert result == pytest.approx(100.0)

        # normalized=1.0 yields max
        result = engine._apply_rolling_denormalization(1.0, bundle, input_data)
        assert result == pytest.approx(200.0)

    def test_nan_input_raises_model_inference_error(self):
        """Verify NaN prediction raises error instead of returning 0.0."""
        engine = self._make_engine()
        metadata = {"price_normalization": {"method": "rolling_minmax", "target_feature": "close"}}
        bundle = _make_bundle(Mock(), metadata=metadata)
        input_data = self._make_input_data([100.0, 200.0])

        with pytest.raises(ModelInferenceError, match="non-finite prediction"):
            engine._apply_rolling_denormalization(float("nan"), bundle, input_data)

    def test_inf_input_raises_model_inference_error(self):
        """Verify Inf prediction raises error instead of returning 0.0."""
        engine = self._make_engine()
        metadata = {"price_normalization": {"method": "rolling_minmax", "target_feature": "close"}}
        bundle = _make_bundle(Mock(), metadata=metadata)
        input_data = self._make_input_data([100.0, 200.0])

        with pytest.raises(ModelInferenceError, match="non-finite prediction"):
            engine._apply_rolling_denormalization(float("inf"), bundle, input_data)

    def test_nonfinite_window_raises_model_inference_error(self):
        """Verify non-finite min/max in window raises error."""
        engine = self._make_engine()
        metadata = {"price_normalization": {"method": "rolling_minmax", "target_feature": "close"}}
        bundle = _make_bundle(Mock(), metadata=metadata)
        # All-NaN window produces NaN min/max (pandas skips NaN in partial windows)
        input_data = self._make_input_data([float("nan"), float("nan"), float("nan")])

        with pytest.raises(ModelInferenceError, match="non-finite values in window"):
            engine._apply_rolling_denormalization(0.5, bundle, input_data)

    def test_constant_window_returns_window_min(self):
        """Verify constant price window returns the constant value."""
        engine = self._make_engine()
        metadata = {"price_normalization": {"method": "rolling_minmax", "target_feature": "close"}}
        bundle = _make_bundle(Mock(), metadata=metadata)
        input_data = self._make_input_data([50000.0, 50000.0, 50000.0])

        result = engine._apply_rolling_denormalization(0.5, bundle, input_data)
        assert result == pytest.approx(50000.0)

    def test_no_metadata_returns_normalized_price(self):
        """Verify passthrough when no metadata is present."""
        engine = self._make_engine()
        bundle = _make_bundle(Mock(), metadata={})
        input_data = self._make_input_data([100.0, 200.0])

        result = engine._apply_rolling_denormalization(0.75, bundle, input_data)
        assert result == pytest.approx(0.75)

    def test_non_rolling_minmax_returns_normalized_price(self):
        """Verify passthrough when normalization method is not rolling_minmax."""
        engine = self._make_engine()
        metadata = {"price_normalization": {"method": "z_score"}}
        bundle = _make_bundle(Mock(), metadata=metadata)
        input_data = self._make_input_data([100.0, 200.0])

        result = engine._apply_rolling_denormalization(0.75, bundle, input_data)
        assert result == pytest.approx(0.75)
