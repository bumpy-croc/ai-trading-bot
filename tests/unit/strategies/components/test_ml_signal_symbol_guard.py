"""Tests for trading-symbol threading and cross-symbol model guards.

Covers the P0 wiring bug where the live runner's --symbol never reached
MLBasicSignalGenerator, so e.g. HyperGrowth on ETHUSDT silently scored with
the BTCUSDT model. Guards live at the generator/registry seam so backtest
and live get identical behavior.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.models.exceptions import ModelNotAvailableError
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _make_bundle(symbol="BTCUSDT", timeframe="1h", model_type="basic", version="v1"):
    return SimpleNamespace(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        version_id=version,
        key=f"{symbol}:{timeframe}:{model_type}:{version}",
    )


def _make_engine(registry=None, predicted_price=51000.0):
    engine = MagicMock()
    engine.health_check.return_value = {"status": "healthy"}
    result = Mock()
    result.price = predicted_price
    result.model_name = "BTCUSDT:1h:basic:v1"
    result.error = None
    result.metadata = {}
    engine.predict.return_value = result
    if registry is not None:
        engine.model_registry = registry
    return engine


def _make_df(length=150, base_price=50000.0):
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


class TestSymbolThreadingThroughFactories:
    """The trading symbol must reach MLBasicSignalGenerator via each factory."""

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_hyper_growth_threads_symbol(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()
        from src.strategies.hyper_growth import create_hyper_growth_strategy

        strategy = create_hyper_growth_strategy(symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_ml_basic_threads_symbol(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()
        from src.strategies.ml_basic import create_ml_basic_strategy

        strategy = create_ml_basic_strategy(symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_leveraged_regime_threads_symbol(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()
        from src.strategies.leveraged_regime import create_leveraged_regime_strategy

        strategy = create_leveraged_regime_strategy(signal_source="ml", symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_ensemble_weighted_threads_symbol(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()
        from src.strategies.ensemble_weighted import create_ensemble_weighted_strategy

        strategy = create_ensemble_weighted_strategy(
            symbol="ETHUSDT", use_ml_adaptive=False, use_ml_sentiment=False
        )

        ml_basic_gens = [
            gen
            for gen in strategy.signal_generator.generators
            if isinstance(gen, MLBasicSignalGenerator)
        ]
        assert len(ml_basic_gens) == 1
        assert ml_basic_gens[0].symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_hyper_growth_default_symbol_unchanged(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()
        from src.strategies.hyper_growth import create_hyper_growth_strategy

        strategy = create_hyper_growth_strategy()

        assert strategy.signal_generator.symbol == "BTCUSDT"


class TestDirectConstructionBackwardCompat:
    """Direct constructions without a symbol keep the BTCUSDT default."""

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_default_symbol_is_btcusdt(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()

        generator = MLBasicSignalGenerator()

        assert generator.symbol == "BTCUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_symbol_normalized_to_binance_style(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()

        generator = MLBasicSignalGenerator(symbol="ETH-USD")

        assert generator.symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_degraded_engine_does_not_fail_fast(self, _cfg, engine_cls):
        """Engine init failure keeps the legacy fail-safe (HOLD) path."""
        engine_cls.side_effect = RuntimeError("no onnxruntime")

        generator = MLBasicSignalGenerator(symbol="ETHUSDT")

        assert generator.prediction_engine is None

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_invalid_symbol_raises_clear_error(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()

        with pytest.raises(ValueError, match="[Ii]nvalid trading symbol.*ETH-USD-X"):
            MLBasicSignalGenerator(symbol="ETH-USD-X")


class TestMissingModelFailFast:
    """No model for (symbol, type, timeframe) must fail fast at startup."""

    def _registry_without_eth(self):
        registry = MagicMock()
        registry.select_bundle.side_effect = ModelNotAvailableError(
            "No model bundle for ETHUSDT 1h basic."
        )
        registry.list_bundles.return_value = [_make_bundle(symbol="BTCUSDT")]
        return registry

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_missing_model_raises_with_available_models(self, _cfg, engine_cls, monkeypatch):
        monkeypatch.delenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", raising=False)
        engine_cls.return_value = _make_engine(registry=self._registry_without_eth())

        with pytest.raises(ModelNotAvailableError) as exc_info:
            MLBasicSignalGenerator(symbol="ETHUSDT")

        message = str(exc_info.value)
        assert "ETHUSDT" in message
        assert "BTCUSDT:1h:basic:v1" in message
        assert "FEATURE_ALLOW_CROSS_SYMBOL_MODEL" in message

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_flag_allows_substitution_and_logs_critical(
        self, _cfg, engine_cls, monkeypatch, caplog
    ):
        monkeypatch.setenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", "true")
        engine_cls.return_value = _make_engine(registry=self._registry_without_eth())

        with caplog.at_level(logging.CRITICAL):
            generator = MLBasicSignalGenerator(symbol="ETHUSDT")

        assert generator._cross_symbol_bundle_key == "BTCUSDT:1h:basic:v1"
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) == 1
        assert "ETHUSDT" in critical_records[0].getMessage()
        assert "BTCUSDT" in critical_records[0].getMessage()

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_flag_without_any_fallback_still_raises(self, _cfg, engine_cls, monkeypatch):
        monkeypatch.setenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", "true")
        registry = MagicMock()
        registry.select_bundle.side_effect = ModelNotAvailableError("nothing")
        registry.list_bundles.return_value = []
        engine_cls.return_value = _make_engine(registry=registry)

        with pytest.raises(ModelNotAvailableError):
            MLBasicSignalGenerator(symbol="ETHUSDT")

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_flag_substitution_warns_per_resolution_and_stamps_metadata(
        self, _cfg, engine_cls, monkeypatch, caplog
    ):
        monkeypatch.setenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", "true")
        engine = _make_engine(registry=self._registry_without_eth())
        engine_cls.return_value = engine

        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        with caplog.at_level(logging.WARNING):
            signal = generator.generate_signal(df, 130)

        assert signal.metadata["trading_symbol"] == "ETHUSDT"
        assert signal.metadata["model_symbol"] == "BTCUSDT"
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "cross-symbol" in r.getMessage().lower()
        ]
        assert len(warnings) == 1
        # Substituted bundle key must be what the engine scores with
        _, kwargs = engine.predict.call_args
        assert kwargs.get("model_name") == "BTCUSDT:1h:basic:v1"


class TestMismatchGuardAtResolution:
    """Resolved bundle symbol != trading symbol logs ERROR and stamps metadata."""

    def _mismatched_registry(self):
        registry = MagicMock()
        registry.select_bundle.return_value = _make_bundle(symbol="BTCUSDT")
        registry.list_bundles.return_value = [_make_bundle(symbol="BTCUSDT")]
        return registry

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_mismatch_logs_error_and_stamps_metadata(self, _cfg, engine_cls, caplog):
        engine_cls.return_value = _make_engine(registry=self._mismatched_registry())
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        with caplog.at_level(logging.ERROR):
            signal = generator.generate_signal(df, 130)

        assert signal.metadata["trading_symbol"] == "ETHUSDT"
        assert signal.metadata["model_symbol"] == "BTCUSDT"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "ETHUSDT" in errors[0].getMessage()
        assert "BTCUSDT" in errors[0].getMessage()

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_mismatch_error_is_rate_limited(self, _cfg, engine_cls, caplog):
        engine_cls.return_value = _make_engine(registry=self._mismatched_registry())
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        with caplog.at_level(logging.ERROR):
            generator.generate_signal(df, 130)
            generator.generate_signal(df, 131)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_matching_symbol_produces_no_guard_logs(self, _cfg, engine_cls, caplog):
        registry = MagicMock()
        registry.select_bundle.return_value = _make_bundle(symbol="BTCUSDT")
        registry.list_bundles.return_value = [_make_bundle(symbol="BTCUSDT")]
        engine_cls.return_value = _make_engine(registry=registry)
        generator = MLBasicSignalGenerator(symbol="BTCUSDT")
        df = _make_df(150)

        with caplog.at_level(logging.WARNING):
            signal = generator.generate_signal(df, 130)

        assert signal.metadata["trading_symbol"] == "BTCUSDT"
        assert signal.metadata["model_symbol"] == "BTCUSDT"
        guard_records = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "symbol" in r.getMessage().lower()
        ]
        assert guard_records == []

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_bundle_vanishing_after_startup_fails_safe(self, _cfg, engine_cls, caplog):
        """Registry reload removing the bundle must not silently substitute."""
        registry = MagicMock()
        registry.select_bundle.return_value = _make_bundle(symbol="ETHUSDT")
        registry.list_bundles.return_value = [_make_bundle(symbol="ETHUSDT")]
        engine = _make_engine(registry=registry)
        engine_cls.return_value = engine
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        registry.select_bundle.side_effect = ModelNotAvailableError("bundle gone")
        with caplog.at_level(logging.ERROR):
            signal = generator.generate_signal(df, 130)

        # Prediction must fail (HOLD) rather than scoring another symbol's model
        assert signal.metadata["reason"] == "prediction_failed"
        # HOLD path still carries the guard stamps for auditability
        assert signal.metadata["trading_symbol"] == "ETHUSDT"
        assert signal.metadata["model_symbol"] is None
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("ETHUSDT" in r.getMessage() for r in errors)

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_distinct_guard_conditions_use_separate_rate_limit_clocks(
        self, _cfg, engine_cls, caplog
    ):
        """A mismatch log must not swallow the first bundle-vanished log."""
        registry = MagicMock()
        registry.select_bundle.return_value = _make_bundle(symbol="BTCUSDT")
        registry.list_bundles.return_value = [_make_bundle(symbol="BTCUSDT")]
        engine_cls.return_value = _make_engine(registry=registry)
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        with caplog.at_level(logging.ERROR):
            generator.generate_signal(df, 130)  # mismatch ERROR
            registry.select_bundle.side_effect = ModelNotAvailableError("bundle gone")
            generator.generate_signal(df, 131)  # vanished ERROR — separate clock

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("MISMATCH" in message for message in errors)
        assert any("refusing cross-symbol fallback" in message for message in errors)


class TestHoldPathMetadataStamps:
    """All HOLD branches stamp trading_symbol/model_symbol for auditability."""

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_insufficient_history_stamps_symbols(self, _cfg, engine_cls):
        engine_cls.return_value = _make_engine()
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(50)

        signal = generator.generate_signal(df, 30)

        assert signal.metadata["reason"] == "insufficient_history"
        assert signal.metadata["trading_symbol"] == "ETHUSDT"
        assert signal.metadata["model_symbol"] is None

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_prediction_failed_stamps_symbols(self, _cfg, engine_cls):
        engine = _make_engine()
        engine.predict.side_effect = RuntimeError("inference exploded")
        engine_cls.return_value = engine
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.metadata["reason"] == "prediction_failed"
        assert signal.metadata["trading_symbol"] == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_invalid_prediction_stamps_symbols(self, _cfg, engine_cls):
        engine = _make_engine(predicted_price=float("nan"))
        engine_cls.return_value = engine
        generator = MLBasicSignalGenerator(symbol="ETHUSDT")
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.metadata["reason"] == "invalid_prediction_or_price"
        assert signal.metadata["trading_symbol"] == "ETHUSDT"
