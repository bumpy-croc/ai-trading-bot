"""Single-source long-only deployment config for HyperGrowth/ETHUSDT (GH #1020).

Risk-review condition C1: ``allow_shorts`` is defined in exactly one place
(``src.strategies.deployment_config``) and consumed by BOTH engines'
strategy-construction paths — the live runner's ``load_strategy`` and the
backtest CLI's ``_load_strategy`` both go through
``call_strategy_factory -> create_hyper_growth_strategy``, so a backtest of
HyperGrowth on ETHUSDT resolves the same effective value as live with zero
extra flags. All other strategies/symbols are unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.strategies import call_strategy_factory
from src.strategies.deployment_config import LONG_ONLY_DEPLOYMENTS, resolve_allow_shorts
from src.strategies.hyper_growth import create_hyper_growth_strategy
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _mock_prediction_engine() -> MagicMock:
    engine = MagicMock()
    engine.health_check.return_value = {"status": "healthy"}
    return engine


class TestResolveAllowShorts:
    def test_hyper_growth_ethusdt_is_long_only(self):
        assert resolve_allow_shorts("hyper_growth", "ETHUSDT") is False

    def test_symbol_formats_normalize_to_same_deployment(self):
        """Provider-specific symbol formats must hit the same config entry."""
        assert resolve_allow_shorts("hyper_growth", "ETH-USD") is False
        assert resolve_allow_shorts("hyper_growth", "ethusdt") is False

    def test_other_symbols_allow_shorts(self):
        assert resolve_allow_shorts("hyper_growth", "BTCUSDT") is True

    def test_other_strategies_allow_shorts(self):
        assert resolve_allow_shorts("ml_basic", "ETHUSDT") is True

    def test_no_symbol_allows_shorts(self):
        assert resolve_allow_shorts("hyper_growth", None) is True

    def test_deployment_table_pins_exactly_hyper_growth_ethusdt(self):
        """The named config location holds the one board-ratified deployment."""
        assert LONG_ONLY_DEPLOYMENTS == {"hyper_growth": frozenset({"ETHUSDT"})}


class TestFactoryResolution:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_ethusdt_defaults_long_only(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()

        strategy = create_hyper_growth_strategy(symbol="ETHUSDT")

        assert strategy.signal_generator.allow_shorts is False

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_btcusdt_defaults_shorts_allowed(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()

        strategy = create_hyper_growth_strategy(symbol="BTCUSDT")

        assert strategy.signal_generator.allow_shorts is True

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_no_symbol_defaults_shorts_allowed(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()

        strategy = create_hyper_growth_strategy()

        assert strategy.signal_generator.allow_shorts is True

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_explicit_true_overrides_deployment_default(self, _cfg, engine_cls):
        """Experiments can re-enable shorts explicitly (counterfactual runs)."""
        engine_cls.return_value = _mock_prediction_engine()

        strategy = create_hyper_growth_strategy(symbol="ETHUSDT", allow_shorts=True)

        assert strategy.signal_generator.allow_shorts is True

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_explicit_false_applies_to_any_symbol(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()

        strategy = create_hyper_growth_strategy(symbol="BTCUSDT", allow_shorts=False)

        assert strategy.signal_generator.allow_shorts is False

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_other_strategies_unchanged(self, _cfg, engine_cls):
        """ml_basic on ETHUSDT keeps shorts — the config is per-deployment."""
        engine_cls.return_value = _mock_prediction_engine()

        strategy = create_ml_basic_strategy(symbol="ETHUSDT")

        assert strategy.signal_generator.allow_shorts is True


class TestBothEnginesResolveSameValue:
    """C1 parity: live and backtest construction resolve identical values."""

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_shared_factory_path_resolves_long_only(self, _cfg, engine_cls):
        """call_strategy_factory is the choke point both runners use."""
        engine_cls.return_value = _mock_prediction_engine()

        strategy = call_strategy_factory(create_hyper_growth_strategy, symbol="ETHUSDT")

        assert strategy.signal_generator.allow_shorts is False

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_live_runner_resolves_long_only(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()
        from src.engines.live.runner import load_strategy

        strategy = load_strategy("hyper_growth", symbol="ETHUSDT")

        assert strategy.signal_generator.allow_shorts is False

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_backtest_cli_resolves_long_only(self, _cfg, engine_cls):
        engine_cls.return_value = _mock_prediction_engine()
        from cli.commands.backtest import _load_strategy

        strategy = _load_strategy("hyper_growth", symbol="ETHUSDT")

        assert strategy.signal_generator.allow_shorts is False

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_live_and_backtest_values_match_for_both_symbols(self, _cfg, engine_cls):
        """The parity requirement stated directly: same symbol -> same value."""
        engine_cls.return_value = _mock_prediction_engine()
        from cli.commands.backtest import _load_strategy
        from src.engines.live.runner import load_strategy

        for symbol in ("ETHUSDT", "BTCUSDT"):
            live_value = load_strategy("hyper_growth", symbol=symbol).signal_generator.allow_shorts
            backtest_value = _load_strategy(
                "hyper_growth", symbol=symbol
            ).signal_generator.allow_shorts
            assert live_value == backtest_value
