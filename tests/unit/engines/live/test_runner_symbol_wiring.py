"""Tests that the live runner threads --symbol into strategy factories.

Regression tests for the P0 wiring bug where load_strategy() called
create_hyper_growth_strategy() with zero args, so live ETHUSDT scored
with the BTCUSDT model.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _mock_prediction_engine():
    engine = MagicMock()
    engine.health_check.return_value = {"status": "healthy"}
    return engine


class TestLoadStrategySymbolThreading:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_load_strategy_threads_symbol_to_hyper_growth(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()
        from src.engines.live.runner import load_strategy

        strategy = load_strategy("hyper_growth", symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_load_strategy_threads_symbol_to_ml_basic(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()
        from src.engines.live.runner import load_strategy

        strategy = load_strategy("ml_basic", symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    def test_load_strategy_without_symbol_keeps_working(self):
        """Factories without a symbol parameter must not receive one."""
        from src.engines.live.runner import load_strategy

        strategy = load_strategy("chaos_test", symbol="ETHUSDT")

        assert strategy is not None


class TestMainSymbolWiring:
    def test_main_passes_cli_symbol_to_load_strategy(self, monkeypatch):
        from src.engines.live import runner

        strategy = MagicMock()
        strategy.name = "HyperGrowth"
        recorded = {}

        def fake_load(name, symbol=None):
            recorded["call"] = (name, symbol)
            return strategy

        engine = MagicMock()
        monkeypatch.setattr(runner, "load_strategy", fake_load)
        monkeypatch.setattr(runner, "validate_configuration", lambda args: True)
        monkeypatch.setattr(runner, "LiveTradingEngine", MagicMock(return_value=engine))
        monkeypatch.setattr(runner, "MockDataProvider", MagicMock())
        monkeypatch.setattr(runner, "LiveEngineSettings", MagicMock())
        monkeypatch.setattr(
            sys,
            "argv",
            ["runner", "hyper_growth", "--symbol", "ETHUSDT", "--paper-trading", "--mock-data"],
        )

        runner.main()

        assert recorded["call"] == ("hyper_growth", "ETHUSDT")
        engine.start.assert_called_once_with("ETHUSDT", "1h", exit_on_crash=True)
