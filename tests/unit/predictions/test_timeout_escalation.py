"""Tests for consecutive live-inference-timeout escalation (#927).

Repeated live inference timeouts previously degraded every bar to HOLD with
only per-bar WARNINGs — ``health_check()`` never consulted the timeout
counters, so a permanently hung model looked healthy while the bot silently
stopped trading. The prediction engine now tracks consecutive timeouts and
reports itself degraded past a configurable threshold. This is an
observability escalation, never a trading halt.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.constants import DEFAULT_INFERENCE_TIMEOUT_ESCALATION_THRESHOLD
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine
from src.prediction.exceptions import ModelInferenceTimeoutError
from src.prediction.inference_context import InferenceContext, set_inference_context
from src.prediction.models.onnx_runner import ModelPrediction
from tests.unit.predictions.test_engine_prediction import _make_bundle

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_engine(config: PredictionConfig | None = None) -> PredictionEngine:
    with (
        patch("src.prediction.engine.PredictionModelRegistry"),
        patch("src.prediction.engine.FeaturePipeline"),
    ):
        return PredictionEngine(config or PredictionConfig())


def _slow(features):
    time.sleep(0.2)
    return "late"


def _fast(features):
    return "ok"


class TestEscalationConfig:
    def test_default_threshold(self):
        assert DEFAULT_INFERENCE_TIMEOUT_ESCALATION_THRESHOLD == 10
        config = PredictionConfig()
        assert (
            config.inference_timeout_escalation_threshold
            == DEFAULT_INFERENCE_TIMEOUT_ESCALATION_THRESHOLD
        )

    def test_zero_threshold_disables_escalation(self):
        config = PredictionConfig(inference_timeout_escalation_threshold=0)
        config.validate()  # 0 (disabled) is valid
        engine = _make_engine(config)
        engine._consecutive_timeout_count = 10_000
        assert engine.timeout_escalation_active is False

    def test_negative_threshold_rejected(self):
        config = PredictionConfig(inference_timeout_escalation_threshold=-1)
        with pytest.raises(ValueError, match="inference_timeout_escalation_threshold"):
            config.validate()

    def test_non_integer_threshold_rejected(self):
        config = PredictionConfig(inference_timeout_escalation_threshold=2.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="inference_timeout_escalation_threshold"):
            config.validate()


class TestConsecutiveTimeoutTracking:
    def _live_config(self, threshold: int = 2) -> PredictionConfig:
        config = PredictionConfig()
        config.live_inference_timeout = 0.01
        config.inference_timeout_escalation_threshold = threshold
        return config

    def test_counter_increments_on_each_live_timeout(self):
        set_inference_context(InferenceContext.LIVE)
        engine = _make_engine(self._live_config())

        for expected in (1, 2, 3):
            with pytest.raises(ModelInferenceTimeoutError):
                engine._run_inference(_slow, ("features",), operation_name="op")
            assert engine.consecutive_inference_timeouts == expected

        assert engine.inference_timeout_count == 3

    def test_successful_inference_resets_consecutive_but_not_total(self):
        set_inference_context(InferenceContext.LIVE)
        engine = _make_engine(self._live_config())

        with pytest.raises(ModelInferenceTimeoutError):
            engine._run_inference(_slow, ("features",), operation_name="op")
        assert engine.consecutive_inference_timeouts == 1

        engine._run_inference(_fast, ("features",), operation_name="op")
        assert engine.consecutive_inference_timeouts == 0
        assert engine.inference_timeout_count == 1

    def test_deterministic_success_also_breaks_the_streak(self):
        set_inference_context(InferenceContext.LIVE)
        engine = _make_engine(self._live_config())
        with pytest.raises(ModelInferenceTimeoutError):
            engine._run_inference(_slow, ("features",), operation_name="op")

        set_inference_context(InferenceContext.DETERMINISTIC)
        engine._run_inference(_fast, ("features",), operation_name="op")
        assert engine.consecutive_inference_timeouts == 0

    def test_escalation_flag_flips_at_threshold(self):
        set_inference_context(InferenceContext.LIVE)
        engine = _make_engine(self._live_config(threshold=2))

        with pytest.raises(ModelInferenceTimeoutError):
            engine._run_inference(_slow, ("features",), operation_name="op")
        assert engine.timeout_escalation_active is False

        with pytest.raises(ModelInferenceTimeoutError):
            engine._run_inference(_slow, ("features",), operation_name="op")
        assert engine.timeout_escalation_active is True

    def test_stats_expose_consecutive_timeouts(self):
        engine = _make_engine()
        stats = engine.get_performance_stats()
        assert stats["consecutive_inference_timeouts"] == 0
        assert stats["inference_timeouts"] == 0


class TestHealthCheckConsultsTimeouts:
    """A hung model must not look healthy (#927)."""

    def _make_healthy_engine(self, threshold: int = 2) -> PredictionEngine:
        config = PredictionConfig()
        config.inference_timeout_escalation_threshold = threshold
        engine = _make_engine(config)

        engine.feature_pipeline.transform.return_value = np.random.default_rng(3).random((120, 5))
        mock_model = Mock()
        mock_model.predict.return_value = ModelPrediction(
            price=105.5,
            confidence=0.85,
            direction=1,
            model_name="m",
            inference_time=0.001,
        )
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle
        return engine

    def test_healthy_engine_reports_inference_component_healthy(self):
        engine = self._make_healthy_engine()
        health = engine.health_check()

        assert health["status"] == "healthy"
        inference = health["components"]["inference"]
        assert inference["status"] == "healthy"
        assert inference["consecutive_timeouts"] == 0
        assert inference["inference_timeouts"] == 0
        assert inference["escalation_threshold"] == 2

    def test_persistent_timeouts_degrade_health(self):
        engine = self._make_healthy_engine(threshold=2)
        engine._consecutive_timeout_count = 2
        engine._inference_timeout_count = 2

        health = engine.health_check()

        assert health["status"] == "degraded"
        inference = health["components"]["inference"]
        assert inference["status"] == "error"
        assert inference["consecutive_timeouts"] == 2

    def test_below_threshold_stays_healthy(self):
        engine = self._make_healthy_engine(threshold=2)
        engine._consecutive_timeout_count = 1
        engine._inference_timeout_count = 5

        health = engine.health_check()

        assert health["status"] == "healthy"
        assert health["components"]["inference"]["status"] == "healthy"

    def test_full_predict_path_degrades_health_after_threshold_timeouts(self):
        """End to end: a hung model driven through predict() flips health."""
        set_inference_context(InferenceContext.LIVE)
        engine = self._make_healthy_engine(threshold=2)
        engine.config.live_inference_timeout = 0.01

        default_bundle = engine.model_registry.get_default_bundle.return_value

        def slow_predict(features):
            time.sleep(0.2)
            return ModelPrediction(
                price=105.5,
                confidence=0.85,
                direction=1,
                model_name="hung",
                inference_time=0.2,
            )

        default_bundle.runner.predict.side_effect = slow_predict

        data = pd.DataFrame(
            {
                "open": np.full(120, 100.0),
                "high": np.full(120, 101.0),
                "low": np.full(120, 99.0),
                "close": np.full(120, 100.5),
                "volume": np.full(120, 1000.0),
            }
        )

        for _ in range(2):
            result = engine.predict(data)
            assert result.error is not None
            assert result.metadata["timed_out"] is True

        assert engine.timeout_escalation_active is True
        assert engine.health_check()["status"] == "degraded"
