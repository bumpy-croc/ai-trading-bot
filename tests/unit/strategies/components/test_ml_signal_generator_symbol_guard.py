"""Symbol-keyed model selection and cross-symbol guards for MLSignalGenerator.

MLSignalGenerator (ml_adaptive / ml_sentiment / ensembles) used to score with
the prediction engine's default bundle regardless of trading symbol. It now
mirrors MLBasicSignalGenerator's registry selection and guard semantics.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.models.exceptions import ModelNotAvailableError
from src.strategies.components.ml_signal_generator import MLSignalGenerator

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _bundle(symbol="BTCUSDT", timeframe="1h", model_type="basic", version="v1"):
    return SimpleNamespace(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        version_id=version,
        key=f"{symbol}:{timeframe}:{model_type}:{version}",
    )


def _engine(registry=None, result_model_name="BTCUSDT:1h:basic:v1"):
    engine = MagicMock()
    engine.health_check.return_value = {"status": "healthy"}
    result = Mock()
    result.price = 51000.0
    result.model_name = result_model_name
    result.error = None
    result.metadata = {}
    engine.predict.return_value = result
    if registry is not None:
        engine.model_registry = registry
    return engine


def _df(length=150, base_price=50000.0):
    rng = np.random.default_rng(42)
    prices = base_price * np.cumprod(1 + rng.normal(0, 0.005, length))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": rng.uniform(1000, 10000, length),
        },
        index=pd.date_range("2023-01-01", periods=length, freq="1h"),
    )


def _registry_with(*bundles):
    """Registry whose select_bundle serves only the given bundles."""
    registry = MagicMock()

    def select(*, symbol, model_type, timeframe, stage=None):
        for bundle in bundles:
            if (bundle.symbol, bundle.model_type, bundle.timeframe) == (
                symbol,
                model_type,
                timeframe,
            ):
                return bundle
        raise ModelNotAvailableError(f"No model bundle for {symbol} {timeframe} {model_type}.")

    registry.select_bundle.side_effect = select
    registry.list_bundles.return_value = list(bundles)
    return registry


class TestSymbolKeyedSelection:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_scores_with_the_trading_symbols_bundle(self, _cfg, engine_cls, caplog):
        registry = _registry_with(_bundle("BTCUSDT"), _bundle("ETHUSDT"))
        engine = _engine(registry, result_model_name="ETHUSDT:1h:basic:v1")
        engine_cls.return_value = engine
        generator = MLSignalGenerator(symbol="ETHUSDT")

        with caplog.at_level(logging.WARNING):
            signal = generator.generate_signal(_df(), 130)

        assert engine.predict.call_args.kwargs["model_name"] == "ETHUSDT:1h:basic:v1"
        assert signal.metadata["trading_symbol"] == "ETHUSDT"
        assert signal.metadata["model_symbol"] == "ETHUSDT"
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_model_type_and_timeframe_select_the_bundle(self, _cfg, engine_cls):
        registry = _registry_with(_bundle("ETHUSDT", "4h", "sentiment"))
        engine = _engine(registry, result_model_name="ETHUSDT:4h:sentiment:v1")
        engine_cls.return_value = engine
        generator = MLSignalGenerator(symbol="ETHUSDT", model_type="sentiment", timeframe="4h")

        generator.generate_signal(_df(), 130)

        assert engine.predict.call_args.kwargs["model_name"] == "ETHUSDT:4h:sentiment:v1"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_explicit_model_name_bypasses_registry_selection(self, _cfg, engine_cls):
        registry = _registry_with(_bundle("BTCUSDT"))
        engine = _engine(registry, result_model_name="BTCUSDT:1h:basic:v1")
        engine_cls.return_value = engine

        generator = MLSignalGenerator(symbol="ETHUSDT", model_name="BTCUSDT:1h:basic:v1")
        generator.generate_signal(_df(), 130)

        assert engine.predict.call_args.kwargs["model_name"] == "BTCUSDT:1h:basic:v1"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_degraded_engine_does_not_fail_fast(self, _cfg, engine_cls):
        engine_cls.side_effect = RuntimeError("engine down")

        generator = MLSignalGenerator(symbol="ETHUSDT")

        assert generator.prediction_engine is None


class TestMissingModelFailFast:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_missing_model_raises_with_available_models(self, _cfg, engine_cls, monkeypatch):
        monkeypatch.delenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", raising=False)
        engine_cls.return_value = _engine(_registry_with(_bundle("BTCUSDT")))

        with pytest.raises(ModelNotAvailableError) as exc_info:
            MLSignalGenerator(symbol="ETHUSDT")

        message = str(exc_info.value)
        assert "ETHUSDT" in message
        assert "BTCUSDT:1h:basic:v1" in message
        assert "FEATURE_ALLOW_CROSS_SYMBOL_MODEL" in message

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_flag_allows_substitution_and_stamps_metadata(
        self, _cfg, engine_cls, monkeypatch, caplog
    ):
        monkeypatch.setenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", "true")
        engine = _engine(_registry_with(_bundle("BTCUSDT")))
        engine_cls.return_value = engine

        with caplog.at_level(logging.CRITICAL):
            generator = MLSignalGenerator(symbol="ETHUSDT")
        assert len([r for r in caplog.records if r.levelno == logging.CRITICAL]) == 1

        signal = generator.generate_signal(_df(), 130)

        assert signal.metadata["trading_symbol"] == "ETHUSDT"
        assert signal.metadata["model_symbol"] == "BTCUSDT"
        assert engine.predict.call_args.kwargs["model_name"] == "BTCUSDT:1h:basic:v1"


class TestMismatchGuard:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_mismatch_from_explicit_override_logs_error_once(self, _cfg, engine_cls, caplog):
        engine_cls.return_value = _engine(
            _registry_with(_bundle("BTCUSDT")), result_model_name="BTCUSDT:1h:basic:v1"
        )
        generator = MLSignalGenerator(symbol="ETHUSDT", model_name="BTCUSDT:1h:basic:v1")

        with caplog.at_level(logging.ERROR):
            signal = generator.generate_signal(_df(), 130)
            generator.generate_signal(_df(), 131)

        assert signal.metadata["model_symbol"] == "BTCUSDT"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "MISMATCH" in errors[0].getMessage()

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_bundle_vanishing_after_startup_holds(self, _cfg, engine_cls, caplog):
        registry = _registry_with(_bundle("ETHUSDT"))
        engine_cls.return_value = _engine(registry, result_model_name="ETHUSDT:1h:basic:v1")
        generator = MLSignalGenerator(symbol="ETHUSDT")
        registry.select_bundle.side_effect = ModelNotAvailableError("bundle gone")

        with caplog.at_level(logging.ERROR):
            signal = generator.generate_signal(_df(), 130)

        assert signal.metadata["reason"] == "prediction_failed"
        assert signal.metadata["model_symbol"] is None
        assert any("refusing cross-symbol fallback" in r.getMessage() for r in caplog.records)

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_hold_paths_stamp_symbols(self, _cfg, engine_cls):
        engine_cls.return_value = _engine(
            _registry_with(_bundle("BTCUSDT")), result_model_name="BTCUSDT:1h:basic:v1"
        )
        generator = MLSignalGenerator(symbol="BTCUSDT")

        signal = generator.generate_signal(_df(), 10)

        assert signal.metadata["reason"] == "insufficient_history"
        assert signal.metadata["trading_symbol"] == "BTCUSDT"
        assert signal.metadata["model_symbol"] is None


class TestFactoryThreading:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_ensemble_factory_threads_symbol_to_ml_generators(self, _cfg, engine_cls):
        engine_cls.return_value = _engine(
            _registry_with(_bundle("ETHUSDT")), result_model_name="ETHUSDT:1h:basic:v1"
        )
        from src.strategies.ensemble_weighted import create_ensemble_weighted_strategy

        strategy = create_ensemble_weighted_strategy(
            symbol="ETHUSDT", use_ml_basic=True, use_ml_adaptive=True, use_ml_sentiment=False
        )

        symbols = {g.symbol for g in strategy.signal_generator.generators}
        assert symbols == {"ETHUSDT"}

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_ml_sentiment_factory_threads_model_type_before_validation(self, _cfg, engine_cls):
        registry = _registry_with(_bundle("ETHUSDT", "1h", "sentiment"))
        engine_cls.return_value = _engine(registry)
        from src.strategies.ml_sentiment import create_ml_sentiment_strategy

        strategy = create_ml_sentiment_strategy(symbol="ETHUSDT", model_type="sentiment")

        assert strategy.signal_generator.model_type == "sentiment"


class TestReviewRegressions:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_config_model_name_is_an_operator_override(self, _cfg, engine_cls, monkeypatch):
        """PREDICTION_ENGINE_MODEL_NAME keeps selecting the model, as before."""
        monkeypatch.setenv("PREDICTION_ENGINE_MODEL_NAME", "BTCUSDT:1h:basic:v1")
        registry = _registry_with(_bundle("BTCUSDT"))
        engine = _engine(registry)
        engine_cls.return_value = engine

        generator = MLSignalGenerator(symbol="BTCUSDT")
        generator.generate_signal(_df(), 130)

        assert generator.model_name == "BTCUSDT:1h:basic:v1"
        assert generator._explicit_model_name is True
        registry.select_bundle.assert_not_called()

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_registry_lookup_error_holds_instead_of_engine_default(self, _cfg, engine_cls):
        registry = _registry_with(_bundle("ETHUSDT"))
        engine = _engine(registry, result_model_name="ETHUSDT:1h:basic:v1")
        engine_cls.return_value = engine
        generator = MLSignalGenerator(symbol="ETHUSDT")
        registry.select_bundle.side_effect = KeyError("boom")

        signal = generator.generate_signal(_df(), 130)

        assert signal.metadata["reason"] == "prediction_failed"
        engine.predict.assert_not_called()

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_signal_metadata_records_the_bundle_that_scored(self, _cfg, engine_cls):
        registry = _registry_with(_bundle("ETHUSDT"))
        engine_cls.return_value = _engine(registry, result_model_name="ETHUSDT:1h:basic:v1")
        generator = MLSignalGenerator(symbol="ETHUSDT")

        signal = generator.generate_signal(_df(), 130)

        assert signal.metadata["engine_model_name"] == "ETHUSDT:1h:basic:v1"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_sentiment_preset_defaults_to_the_basic_bundle(self, _cfg, engine_cls):
        """The price-only pipeline cannot feed a sentiment bundle (would HOLD every bar)."""
        engine_cls.return_value = _engine(_registry_with(_bundle("BTCUSDT")))
        from src.strategies.components.strategy_factory import StrategyFactory

        strategy = StrategyFactory.create_ml_sentiment_strategy(symbol="BTCUSDT")

        assert strategy.signal_generator.model_type == "basic"
