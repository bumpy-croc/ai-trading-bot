"""Tests for the inference execution context (deterministic vs live).

Backtest results must be identical run-to-run regardless of CPU load, so the
deterministic context never imposes a wall-clock deadline on inference. Live
trading opts into a bounded latency budget so the trading loop cannot block
indefinitely on a hung model.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config.constants import DEFAULT_LIVE_INFERENCE_TIMEOUT
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine
from src.prediction.inference_context import (
    InferenceContext,
    get_inference_context,
    set_inference_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestInferenceContextModule:
    """Process-wide context switch semantics."""

    def test_default_context_is_deterministic(self):
        assert get_inference_context() is InferenceContext.DETERMINISTIC

    def test_set_and_get_live_context(self):
        set_inference_context(InferenceContext.LIVE)
        assert get_inference_context() is InferenceContext.LIVE

    def test_reset_restores_deterministic(self):
        from src.prediction.inference_context import reset_inference_context

        set_inference_context(InferenceContext.LIVE)
        reset_inference_context()
        assert get_inference_context() is InferenceContext.DETERMINISTIC

    def test_set_rejects_invalid_context(self):
        with pytest.raises(ValueError):
            set_inference_context("yolo")  # type: ignore[arg-type]


class TestDefaults:
    """Default values are safe without any configuration."""

    def test_live_inference_timeout_default_is_seconds_scale(self):
        # 0.1s was a latency-alerting budget mistakenly used as an inference
        # deadline; the live hard deadline must be orders of magnitude above
        # typical CPU ONNX inference (~30-400ms under load).
        assert DEFAULT_LIVE_INFERENCE_TIMEOUT == 5.0

    def test_prediction_config_default_live_timeout(self):
        config = PredictionConfig()
        assert config.live_inference_timeout == DEFAULT_LIVE_INFERENCE_TIMEOUT

    def test_prediction_config_validates_live_timeout(self):
        config = PredictionConfig(live_inference_timeout=0.0)
        with pytest.raises(ValueError, match="live_inference_timeout"):
            config.validate()

    def test_max_prediction_latency_still_validated(self):
        config = PredictionConfig(max_prediction_latency=-0.1)
        with pytest.raises(ValueError, match="max_prediction_latency"):
            config.validate()


class TestEngineTimeoutResolution:
    """PredictionEngine derives its inference deadline from the context."""

    def _make_engine(self) -> PredictionEngine:
        with (
            patch("src.prediction.engine.PredictionModelRegistry"),
            patch("src.prediction.engine.FeaturePipeline"),
        ):
            return PredictionEngine(PredictionConfig())

    def test_deterministic_context_has_no_deadline(self):
        engine = self._make_engine()
        assert engine._get_timeout_seconds() is None

    def test_live_context_uses_config_live_timeout(self):
        engine = self._make_engine()
        set_inference_context(InferenceContext.LIVE)
        assert engine._get_timeout_seconds() == engine.config.live_inference_timeout


class TestEnginesPinContext:
    """Both engines select the context-appropriate policy at construction."""

    def test_backtester_pins_deterministic_context(self, mock_data_provider):
        set_inference_context(InferenceContext.LIVE)

        from src.engines.backtest.engine import Backtester

        mock_strategy = MagicMock()
        mock_strategy.get_risk_overrides.return_value = None
        mock_strategy.name = "MockStrategy"

        Backtester(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            initial_balance=10_000,
            log_to_database=False,
        )

        assert get_inference_context() is InferenceContext.DETERMINISTIC

    def test_live_engine_pins_live_context(self):
        with patch("src.engines.live.trading_engine.DatabaseManager"):
            from src.engines.live.trading_engine import LiveTradingEngine

            mock_strategy = MagicMock()
            mock_strategy.get_risk_overrides.return_value = {}
            mock_strategy.__class__.__name__ = "MockStrategy"
            mock_strategy.config = {}

            mock_dp = MagicMock()
            mock_dp.get_current_price.return_value = 50000.0
            mock_dp.get_live_data.return_value = pd.DataFrame(
                {
                    "open": [50000.0] * 5,
                    "high": [51000.0] * 5,
                    "low": [49000.0] * 5,
                    "close": [50500.0] * 5,
                    "volume": [100.0] * 5,
                },
                index=pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            )

            LiveTradingEngine(
                strategy=mock_strategy,
                data_provider=mock_dp,
                enable_live_trading=False,
                initial_balance=1000.0,
                enable_dynamic_risk=False,
                enable_hot_swapping=False,
            )

        assert get_inference_context() is InferenceContext.LIVE
