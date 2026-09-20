"""Regression tests for GH #1253: the run timeframe must reach ML strategies.

No runner used to pass its timeframe to the ML strategy factories, so their
registry ``model_timeframe`` stayed at the "1h" default and a 4h run scored 4h
bars with a 1h model (or failed at startup with the wrong error).
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.prediction.models.exceptions import ModelNotAvailableError
from src.strategies import call_strategy_factory
from src.strategies.ensemble_weighted import create_ensemble_weighted_strategy
from src.strategies.hyper_growth import create_hyper_growth_strategy
from src.strategies.leveraged_regime import create_leveraged_regime_strategy
from src.strategies.ml_adaptive import create_ml_adaptive_strategy
from src.strategies.ml_basic import create_ml_basic_strategy
from src.strategies.ml_sentiment import create_ml_sentiment_strategy

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _bundle(symbol, timeframe, model_type):
    return SimpleNamespace(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        version_id="v1",
        key=f"{symbol}:{timeframe}:{model_type}:v1",
    )


def _registry_with(*bundles):
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


def _engine(registry):
    engine = Mock()
    engine.health_check.return_value = {"status": "healthy"}
    engine.model_registry = registry
    return engine


def _signal_generator(strategy):
    generator = strategy.signal_generator
    return getattr(generator, "_inner", generator)


class TestCallStrategyFactoryTimeframe:
    def test_timeframe_threaded_when_factory_declares_it(self):
        def factory(timeframe="1h"):
            return timeframe

        assert call_strategy_factory(factory, timeframe="4h") == "4h"

    def test_not_threaded_through_var_kwargs(self):
        seen = {}

        def factory(**kwargs):
            seen.update(kwargs)
            return "strategy"

        call_strategy_factory(factory, symbol="ETHUSDT", timeframe="4h")

        assert seen == {"symbol": "ETHUSDT"}

    def test_factory_without_timeframe_is_called_unchanged(self):
        def factory(symbol=None):
            return symbol

        assert call_strategy_factory(factory, symbol="ETHUSDT", timeframe="4h") == "ETHUSDT"

    def test_none_timeframe_keeps_factory_default(self):
        def factory(timeframe="1h"):
            return timeframe

        assert call_strategy_factory(factory) == "1h"


class TestTimeframeReachesSignalGenerator:
    @pytest.mark.parametrize(
        "factory",
        [create_ml_basic_strategy, create_ml_adaptive_strategy, create_ml_sentiment_strategy],
    )
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_non_1h_run_selects_that_timeframe(self, _cfg, engine_cls, factory):
        engine_cls.return_value = _engine(
            _registry_with(_bundle("ETHUSDT", "1h", "basic"), _bundle("ETHUSDT", "4h", "basic"))
        )

        strategy = call_strategy_factory(factory, symbol="ETHUSDT", timeframe="4h")

        assert _signal_generator(strategy).model_timeframe == "4h"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_adaptive_default_is_still_1h(self, _cfg, engine_cls):
        engine_cls.return_value = _engine(_registry_with(_bundle("ETHUSDT", "1h", "basic")))

        strategy = call_strategy_factory(create_ml_adaptive_strategy, symbol="ETHUSDT")

        assert _signal_generator(strategy).model_timeframe == "1h"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_hyper_growth_and_ensemble_pick_up_timeframe(self, _cfg, engine_cls):
        engine_cls.return_value = _engine(
            _registry_with(_bundle("ETHUSDT", "4h", "basic"), _bundle("ETHUSDT", "1h", "basic"))
        )

        hyper = call_strategy_factory(
            create_hyper_growth_strategy, symbol="ETHUSDT", timeframe="4h"
        )
        ensemble = call_strategy_factory(
            create_ensemble_weighted_strategy, symbol="ETHUSDT", timeframe="4h"
        )

        leveraged = call_strategy_factory(
            partial(create_leveraged_regime_strategy, signal_source="ml"),
            symbol="ETHUSDT",
            timeframe="4h",
        )
        assert _signal_generator(hyper).model_timeframe == "4h"
        assert _signal_generator(leveraged).model_timeframe == "4h"
        generators = list(ensemble.signal_generator.generators)
        assert generators
        assert {g.model_timeframe for g in generators} == {"4h"}


class TestTimeframeMismatchFailsLoudly:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_no_bundle_for_run_timeframe_raises_instead_of_using_1h(
        self, _cfg, engine_cls, monkeypatch
    ):
        monkeypatch.delenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", raising=False)
        engine_cls.return_value = _engine(_registry_with(_bundle("ETHUSDT", "1h", "basic")))

        with pytest.raises(ModelNotAvailableError) as exc_info:
            call_strategy_factory(create_ml_adaptive_strategy, symbol="ETHUSDT", timeframe="4h")

        message = str(exc_info.value)
        assert "4h" in message
        assert "ETHUSDT:1h:basic:v1" in message


class TestRunnersThreadTimeframe:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_backtest_cli_loader(self, _cfg, engine_cls):
        from cli.commands.backtest import _load_strategy

        engine_cls.return_value = _engine(_registry_with(_bundle("ETHUSDT", "4h", "basic")))

        strategy = _load_strategy("ml_adaptive", symbol="ETHUSDT", timeframe="4h")

        assert _signal_generator(strategy).model_timeframe == "4h"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_live_runner_loader(self, _cfg, engine_cls):
        from src.engines.live.runner import load_strategy

        engine_cls.return_value = _engine(_registry_with(_bundle("ETHUSDT", "4h", "basic")))

        strategy = load_strategy("hyper_growth", symbol="ETHUSDT", timeframe="4h")

        assert _signal_generator(strategy).model_timeframe == "4h"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_experiment_runner_loader(self, _cfg, engine_cls):
        from src.experiments.runner import ExperimentRunner

        engine_cls.return_value = _engine(_registry_with(_bundle("ETHUSDT", "4h", "basic")))

        strategy = ExperimentRunner()._load_strategy(
            "ml_adaptive", symbol="ETHUSDT", timeframe="4h"
        )

        assert _signal_generator(strategy).model_timeframe == "4h"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_experiment_factory_kwargs_timeframe_wins(self, _cfg, engine_cls):
        from src.experiments.runner import ExperimentRunner

        engine_cls.return_value = _engine(
            _registry_with(_bundle("ETHUSDT", "4h", "basic"), _bundle("ETHUSDT", "1h", "basic"))
        )

        strategy = ExperimentRunner()._load_strategy(
            "ml_adaptive",
            factory_kwargs={"timeframe": "1h"},
            symbol="ETHUSDT",
            timeframe="4h",
        )

        assert _signal_generator(strategy).model_timeframe == "1h"

    def test_strategy_manager_threads_timeframe(self):
        from src.engines.live.strategy_manager import StrategyManager

        seen = {}

        def factory(symbol=None, timeframe="1h"):
            seen.update(symbol=symbol, timeframe=timeframe)
            return SimpleNamespace(name="s")

        manager = StrategyManager(symbol="ETHUSDT", timeframe="4h")
        manager.strategy_registry = {"s": factory}
        manager.load_strategy("s")

        assert seen == {"symbol": "ETHUSDT", "timeframe": "4h"}
