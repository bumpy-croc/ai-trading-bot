"""Tests that ExperimentRunner threads config.symbol into strategy factories.

Regression coverage for GH #997: ``ExperimentRunner._load_strategy`` built
strategies via ``builder(**factory_kwargs)`` and never threaded
``config.symbol`` in, so any ML strategy run through the experiments
framework without an explicit ``symbol`` factory kwarg silently defaulted
its signal generator's symbol to ``MLBasicSignalGenerator.DEFAULT_SYMBOL``
("BTCUSDT") -- scoring whatever symbol's candles the backtest actually ran
on with the wrong model. This mirrors the equivalent regression suites for
the live runner (``tests/unit/engines/live/test_runner_symbol_wiring.py``)
and the backtest CLI (``tests/unit/cli/test_backtest_symbol_wiring.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _mock_prediction_engine():
    engine = MagicMock()
    engine.health_check.return_value = {"status": "healthy"}
    return engine


class TestLoadStrategySymbolThreading:
    """Direct ``_load_strategy`` coverage of the auto-injection logic."""

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_threads_symbol_to_hyper_growth(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()
        runner = ExperimentRunner()

        strategy = runner._load_strategy("hyper_growth", symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_threads_symbol_to_ml_basic(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()
        runner = ExperimentRunner()

        strategy = runner._load_strategy("ml_basic", symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    def test_explicit_factory_kwargs_symbol_wins_over_auto_injected(self):
        """An explicit factory_kwargs["symbol"] must not be silently
        overwritten by the auto-injected config.symbol."""
        runner = ExperimentRunner()

        with patch(
            "src.experiments.runner.create_hyper_growth_strategy", autospec=True
        ) as mock_create:
            runner._load_strategy(
                "hyper_growth",
                factory_kwargs={"symbol": "SOLUSDT"},
                symbol="ETHUSDT",
            )

        _, kwargs = mock_create.call_args
        assert kwargs["symbol"] == "SOLUSDT"

    def test_no_symbol_kwarg_means_no_injection(self):
        """Callers that don't pass symbol (e.g. legacy call sites) keep the
        pre-fix behavior rather than injecting a stray None."""
        runner = ExperimentRunner()

        with patch(
            "src.experiments.runner.create_hyper_growth_strategy", autospec=True
        ) as mock_create:
            runner._load_strategy("hyper_growth")

        mock_create.assert_called_once_with()

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_factory_without_symbol_param_not_regressed(self, _cfg, engine_cls):
        """ml_adaptive's factory (create_ml_adaptive_strategy) has no
        `symbol` parameter at all -- auto-injection must detect this via
        inspect.signature and skip it rather than raising
        TypeError: create_ml_adaptive_strategy() got an unexpected keyword
        argument 'symbol'."""
        engine_cls.return_value = _mock_prediction_engine()
        runner = ExperimentRunner()

        strategy = runner._load_strategy("ml_adaptive", symbol="ETHUSDT")

        assert strategy is not None
        assert not hasattr(strategy.signal_generator, "symbol")


class TestRunThreadsConfigSymbol:
    """End-to-end coverage through the public ``run()`` entry point.

    ``Backtester`` is replaced with a capturing stub so the test exercises
    the real strategy-construction path (including ``_load_strategy``'s
    kwarg merging) without running an actual backtest.
    """

    class _CapturingBacktester:
        captured_strategy = None

        def __init__(self, *, strategy, **_kwargs):
            type(self).captured_strategy = strategy

        def run(self, **_kwargs):
            return {}

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_run_threads_config_symbol_into_ml_strategy_factory(
        self, _cfg, engine_cls, monkeypatch
    ):
        engine_cls.return_value = _mock_prediction_engine()
        monkeypatch.setattr("src.experiments.runner.Backtester", self._CapturingBacktester)

        runner = ExperimentRunner()
        end = datetime.now(UTC)
        start = end - timedelta(hours=2)
        cfg = ExperimentConfig(
            strategy_name="hyper_growth",
            symbol="ETHUSDT",
            timeframe="1h",
            start=start,
            end=end,
            initial_balance=1000.0,
            provider="mock",
            use_cache=False,
        )

        runner.run(cfg)

        captured = self._CapturingBacktester.captured_strategy
        assert captured is not None
        assert captured.signal_generator.symbol == "ETHUSDT"
