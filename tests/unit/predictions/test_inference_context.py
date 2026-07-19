"""Tests for the inference execution context (deterministic vs live).

Backtest results must be identical run-to-run regardless of CPU load, so the
deterministic context never imposes a wall-clock deadline on inference. Live
trading opts into a bounded latency budget so the trading loop cannot block
indefinitely on a hung model.

The context is scoped (contextvars), not process-global (#926): constructing
a Backtester in the same process as a live engine must never strip the live
inference deadline, and a backtest run nested inside a live process restores
the live policy on exit.
"""

import threading
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.constants import DEFAULT_LIVE_INFERENCE_TIMEOUT
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine
from src.prediction.inference_context import (
    InferenceContext,
    get_inference_context,
    inference_scope,
    set_inference_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_prediction_engine() -> PredictionEngine:
    with (
        patch("src.prediction.engine.PredictionModelRegistry"),
        patch("src.prediction.engine.FeaturePipeline"),
    ):
        return PredictionEngine(PredictionConfig())


def _make_live_engine():
    """Construct a LiveTradingEngine with mocked externals (DB, provider)."""
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

        return LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_dp,
            enable_live_trading=False,
            initial_balance=1000.0,
            enable_dynamic_risk=False,
            enable_hot_swapping=False,
        )


def _make_backtester(mock_data_provider):
    from src.engines.backtest.engine import Backtester

    mock_strategy = MagicMock()
    mock_strategy.get_risk_overrides.return_value = None
    mock_strategy.name = "MockStrategy"

    return Backtester(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        initial_balance=10_000,
        log_to_database=False,
    )


class TestInferenceContextModule:
    """Context accessor semantics."""

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

    def test_set_does_not_leak_across_threads(self):
        """The context is scoped, not process-global: pinning LIVE in one
        thread must not change the policy another thread observes."""
        set_inference_context(InferenceContext.LIVE)

        seen: dict[str, InferenceContext] = {}

        def probe():
            seen["context"] = get_inference_context()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()

        assert seen["context"] is InferenceContext.DETERMINISTIC


class TestInferenceScope:
    """Scoped (context-manager) policy selection."""

    def test_scope_applies_and_restores(self):
        with inference_scope(InferenceContext.LIVE):
            assert get_inference_context() is InferenceContext.LIVE
        assert get_inference_context() is InferenceContext.DETERMINISTIC

    def test_nested_scope_restores_outer_policy(self):
        """A deterministic scope nested inside a live one (the in-process
        validation-gate shape) restores the live deadline on exit."""
        with inference_scope(InferenceContext.LIVE):
            with inference_scope(InferenceContext.DETERMINISTIC):
                assert get_inference_context() is InferenceContext.DETERMINISTIC
            assert get_inference_context() is InferenceContext.LIVE

    def test_scope_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with inference_scope(InferenceContext.LIVE):
                raise RuntimeError("boom")
        assert get_inference_context() is InferenceContext.DETERMINISTIC

    def test_scope_rejects_invalid_context(self):
        with pytest.raises(ValueError):
            with inference_scope("yolo"):  # type: ignore[arg-type]
                pass


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

    def test_deterministic_context_has_no_deadline(self):
        engine = _make_prediction_engine()
        assert engine._get_timeout_seconds() is None

    def test_live_context_uses_config_live_timeout(self):
        engine = _make_prediction_engine()
        set_inference_context(InferenceContext.LIVE)
        assert engine._get_timeout_seconds() == engine.config.live_inference_timeout


class TestTwoEnginesOneProcess:
    """Constructing a Backtester must never strip a live engine's deadline (#926).

    The live 5s inference deadline previously rested on an undocumented
    one-engine-per-process invariant: the Backtester constructor set a
    process-wide DETERMINISTIC policy, silently removing the deadline from
    any live engine in the same process (last-writer-wins).
    """

    def test_backtester_construction_preserves_live_deadline(self, mock_data_provider):
        _make_live_engine()
        assert get_inference_context() is InferenceContext.LIVE

        prediction_engine = _make_prediction_engine()
        assert (
            prediction_engine._get_timeout_seconds()
            == prediction_engine.config.live_inference_timeout
        )

        # A Backtester constructed in the same process (e.g. a future
        # in-process validation gate) must not clobber the live policy.
        _make_backtester(mock_data_provider)

        assert get_inference_context() is InferenceContext.LIVE
        assert (
            prediction_engine._get_timeout_seconds()
            == prediction_engine.config.live_inference_timeout
        )

    def test_backtester_run_is_deterministic_scoped_and_restores(self, mock_data_provider):
        """Backtester.run executes under the deterministic policy even inside
        a live process, and restores the live policy on exit."""
        from datetime import datetime

        backtester = _make_backtester(mock_data_provider)
        seen: dict[str, InferenceContext] = {}

        def fake_fetch(*args, **kwargs):
            seen["context"] = get_inference_context()
            return pd.DataFrame()

        set_inference_context(InferenceContext.LIVE)
        with patch.object(backtester, "_fetch_and_prepare_data", side_effect=fake_fetch):
            backtester.run("BTCUSDT", "1h", start=datetime(2024, 1, 1))

        assert seen["context"] is InferenceContext.DETERMINISTIC
        assert get_inference_context() is InferenceContext.LIVE


class TestEnginesPinContext:
    """Engines select the context-appropriate policy where inference runs."""

    def test_live_engine_pins_live_context(self):
        _make_live_engine()
        assert get_inference_context() is InferenceContext.LIVE

    def test_live_trading_loop_thread_runs_under_live_context(self):
        """All live predictions happen on the trading-loop thread, which does
        not inherit the constructor thread's context — the loop must scope
        itself LIVE or the deadline silently disappears."""
        from src.engines.live.trading_engine import LiveTradingEngine

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        seen: dict[str, InferenceContext] = {}

        def fake_loop(symbol, timeframe, max_steps=None):
            seen["context"] = get_inference_context()

        engine._trading_loop = fake_loop  # type: ignore[method-assign]

        thread = threading.Thread(target=engine._run_trading_loop, args=("BTCUSDT", "1h"))
        thread.start()
        thread.join()

        assert seen["context"] is InferenceContext.LIVE


class TestContextParity:
    """Fast-path decisions are identical under both contexts (#923 arch P2).

    The live deadline only aborts inference that overruns it; a prediction
    completing within budget must be bit-identical to the deterministic
    context's result, or backtest-live parity is broken.
    """

    def _make_engine_with_fast_model(self) -> PredictionEngine:
        from src.prediction.models.onnx_runner import ModelPrediction
        from tests.unit.predictions.test_engine_prediction import _make_bundle

        engine = _make_prediction_engine()
        engine.feature_pipeline.transform.return_value = np.random.default_rng(11).random((1, 10))

        mock_model = Mock()
        mock_model.predict.return_value = ModelPrediction(
            price=105.5,
            confidence=0.85,
            direction=1,
            model_name="fast_model",
            inference_time=0.001,
        )
        bundle = _make_bundle(mock_model)
        engine.model_registry.list_bundles.return_value = [bundle]
        engine.model_registry.get_default_bundle.return_value = bundle
        return engine

    def _create_test_data(self, num_rows=120):
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

    def test_fast_path_predictions_identical_across_contexts(self):
        engine = self._make_engine_with_fast_model()
        data = self._create_test_data()

        with inference_scope(InferenceContext.DETERMINISTIC):
            deterministic = engine.predict(data)
        with inference_scope(InferenceContext.LIVE):
            live = engine.predict(data)

        for result in (deterministic, live):
            assert result.error is None

        assert (
            deterministic.price,
            deterministic.confidence,
            deterministic.direction,
            deterministic.model_name,
            deterministic.probabilities,
        ) == (
            live.price,
            live.confidence,
            live.direction,
            live.model_name,
            live.probabilities,
        )
