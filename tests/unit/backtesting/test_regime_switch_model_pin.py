"""GH #1258: a regime switch inside a pinned backtest must honor the run's model pin.

Without the pin reaching ``RegimeHandler``, a switched-in ML strategy resolved
``latest`` while the initial strategy scored with the pinned version, so a
"point-in-time" comparison silently mixed two models.
"""

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.engines.backtest.engine import Backtester
from src.engines.backtest.regime.regime_handler import RegimeHandler
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    HoldSignalGenerator,
    Strategy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"
HANDLER_LOGGER = "src.engines.backtest.regime.regime_handler.logger"
PIN = "2026-01-01_00h_v1"

# Switch targets the handler can build, minus ml_basic (which honors the pin).
# Filtered on the real signatures so a factory that later gains pin support
# drops out of the refusal check instead of failing it.
_SWITCH_TARGET_FACTORIES = {
    "ml_adaptive": ("src.strategies.ml_adaptive", "create_ml_adaptive_strategy"),
    "ml_sentiment": ("src.strategies.ml_sentiment", "create_ml_sentiment_strategy"),
    "ensemble_weighted": ("src.strategies.ensemble_weighted", "create_ensemble_weighted_strategy"),
    "momentum_leverage": ("src.strategies.momentum_leverage", "create_momentum_leverage_strategy"),
    "leveraged_regime": ("src.strategies.leveraged_regime", "create_leveraged_regime_strategy"),
}
_FACTORIES_WITHOUT_PIN_SUPPORT = [
    name
    for name, (module, factory) in _SWITCH_TARGET_FACTORIES.items()
    if "model_version"
    not in inspect.signature(getattr(importlib.import_module(module), factory)).parameters
]


def _handler(model_version=None):
    return RegimeHandler(
        regime_switcher=MagicMock(),
        strategy_manager=MagicMock(),
        initial_strategy_name="ml_basic",
        model_version=model_version,
    )


def _pinnable_registry(symbol="ETHUSDT", timeframe="1h"):
    """Registry whose pinned lookup works but whose ``latest`` lookup must never run."""
    bundle = SimpleNamespace(
        symbol=symbol,
        timeframe=timeframe,
        model_type="basic",
        version_id=PIN,
        key=f"{symbol}:{timeframe}:basic:{PIN}",
    )
    registry = MagicMock()
    registry.get_bundle_by_key.side_effect = lambda key: bundle if key == bundle.key else None
    registry.select_bundle.side_effect = AssertionError("pinned run resolved latest")
    registry.list_bundles.return_value = [bundle]
    return registry


def _engine(registry):
    engine = MagicMock()
    engine.health_check.return_value = {"status": "healthy"}
    engine.model_registry = registry
    return engine


class TestHandlerAppliesPin:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_switched_in_ml_strategy_scores_with_pinned_version(self, _cfg, engine_cls):
        engine_cls.return_value = _engine(_pinnable_registry())

        strategy = _handler(PIN)._load_strategy("ml_basic", symbol="ETHUSDT", timeframe="1h")

        generator = getattr(strategy.signal_generator, "_inner", strategy.signal_generator)
        assert generator.pinned_model_key == f"ETHUSDT:1h:basic:{PIN}"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_unresolvable_pin_fails_the_load_instead_of_using_latest(self, _cfg, engine_cls):
        registry = _pinnable_registry()
        registry.get_bundle_by_key.side_effect = None
        registry.get_bundle_by_key.return_value = None
        engine_cls.return_value = _engine(registry)

        with patch(HANDLER_LOGGER) as log:
            result = _handler(PIN)._load_strategy("ml_basic", symbol="ETHUSDT", timeframe="1h")

        assert result is None
        assert "Pinned model version not found" in str(log.error.call_args)
        registry.select_bundle.assert_not_called()

    def test_factory_without_pin_support_is_refused_not_built_unpinned(self):
        factory = MagicMock(__name__="create_ml_adaptive_strategy")

        def no_pin_factory(symbol=None, timeframe="1h"):
            factory(symbol=symbol, timeframe=timeframe)

        with (
            patch("src.strategies.ml_adaptive.create_ml_adaptive_strategy", no_pin_factory),
            patch(HANDLER_LOGGER) as log,
        ):
            result = _handler(PIN)._load_strategy("ml_adaptive", symbol="ETHUSDT", timeframe="1h")

        assert result is None
        factory.assert_not_called()
        assert "does not support model_version pinning" in str(log.error.call_args)
        assert PIN in str(log.error.call_args)

    @pytest.mark.parametrize("name", _FACTORIES_WITHOUT_PIN_SUPPORT)
    def test_real_factories_without_pin_support_are_never_switched_to(self, name):
        with patch(HANDLER_LOGGER) as log:
            result = _handler(PIN)._load_strategy(name, symbol="ETHUSDT", timeframe="1h")

        assert result is None
        assert "does not support model_version pinning" in str(log.error.call_args)

    def test_failed_pinned_load_is_not_retried(self):
        handler = _handler(PIN)

        with (
            patch(
                "src.engines.backtest.regime.regime_handler.call_strategy_factory",
                side_effect=ValueError("cannot honor pin"),
            ) as factory,
            patch(HANDLER_LOGGER),
        ):
            handler._load_strategy("ml_adaptive", symbol="ETHUSDT", timeframe="1h")
            handler._load_strategy("ml_adaptive", symbol="ETHUSDT", timeframe="1h")

        assert factory.call_count == 1

    def test_unpinned_run_passes_no_pin(self):
        with patch(
            "src.engines.backtest.regime.regime_handler.call_strategy_factory",
            return_value="strategy",
        ) as factory:
            _handler()._load_strategy("ml_basic", symbol="ETHUSDT", timeframe="4h")

        assert factory.call_args.kwargs == {
            "symbol": "ETHUSDT",
            "model_version": None,
            "timeframe": "4h",
        }

    def test_pin_reaches_factory_for_pinned_run(self):
        with patch(
            "src.engines.backtest.regime.regime_handler.call_strategy_factory",
            return_value="strategy",
        ) as factory:
            _handler(PIN)._load_strategy("ml_basic", symbol="ETHUSDT", timeframe="4h")

        assert factory.call_args.kwargs["model_version"] == PIN


class TestBacktesterThreadsPin:
    @staticmethod
    def _strategy():
        return Strategy(
            name="stub",
            signal_generator=HoldSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=FixedFractionSizer(),
        )

    @staticmethod
    def _backtester(model_version):
        return Backtester(
            strategy=TestBacktesterThreadsPin._strategy(),
            data_provider=MagicMock(),
            log_to_database=False,
            enable_regime_switching=True,
            model_version=model_version,
            _regime_switcher_class=MagicMock(),
            _strategy_manager=MagicMock(),
        )

    def test_pin_reaches_regime_handler(self):
        backtester = self._backtester(PIN)

        assert backtester.regime_handler is not None
        assert backtester.regime_handler.model_version == PIN

    def test_unpinned_backtester_builds_unpinned_handler(self):
        backtester = self._backtester(None)

        assert backtester.regime_handler is not None
        assert backtester.regime_handler.model_version is None
