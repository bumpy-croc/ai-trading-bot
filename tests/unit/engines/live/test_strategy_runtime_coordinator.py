"""Contract tests for StrategyRuntimeCoordinator wiring (#486).

Behavioral coverage of the runtime pipeline lives in test_engine_parity_wiring.py,
test_live_policy_hydration.py, and test_strategy.py (exercised via the engine
wrappers). These tests pin the extraction's own contract: the engine builds the
coordinator bound to itself, the thin wrappers delegate to it, and strategy-state
mutations made through the coordinator land on the engine object.
"""

from unittest.mock import create_autospec, patch

import pytest

from src.data_providers.data_provider import DataProvider
from src.engines.live.strategy_runtime import StrategyRuntimeCoordinator
from src.engines.live.trading_engine import LiveTradingEngine
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast


def make_engine() -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=create_autospec(DataProvider, instance=True),
            initial_balance=1000.0,
        )


class TestWiring:
    def test_engine_builds_coordinator_bound_to_itself(self):
        engine = make_engine()
        assert isinstance(engine.strategy_coordinator, StrategyRuntimeCoordinator)
        assert engine.strategy_coordinator._state is engine

    def test_configure_strategy_mutations_reach_engine(self):
        # configure_strategy writes strategy/_component_strategy/_runtime on the
        # engine through the coordinator; the engine must see them.
        engine = make_engine()
        new_strategy = create_ml_basic_strategy()

        engine._configure_strategy(new_strategy)

        assert engine.strategy is new_strategy
        assert engine._component_strategy is new_strategy
        # A ComponentStrategy is wrapped in a runtime, so the engine is "runtime".
        assert engine._is_runtime_strategy() is True

    def test_strategy_name_delegates(self):
        engine = make_engine()
        assert engine._strategy_name() == engine.strategy_coordinator.strategy_name()
        assert engine._strategy_name() == "MlBasic"


class TestRiskParameterHelpers:
    def test_clone_is_a_deep_copy(self):
        params = RiskParameters()
        clone = StrategyRuntimeCoordinator.clone_risk_parameters(params)
        assert clone is not None
        assert clone is not params
        assert clone == params

    def test_clone_none_returns_none(self):
        assert StrategyRuntimeCoordinator.clone_risk_parameters(None) is None

    def test_merge_non_default_engine_overrides_component(self):
        coord = make_engine().strategy_coordinator
        engine_params = RiskParameters(base_risk_per_trade=0.025)  # non-default, within max
        component_params = RiskParameters(base_risk_per_trade=0.01)

        merged = coord.merge_risk_parameters(engine_params, component_params)

        assert merged is not None
        # Engine value is non-default, so it overrides the component value.
        assert merged.base_risk_per_trade == 0.025

    def test_merge_both_none_returns_none(self):
        coord = make_engine().strategy_coordinator
        assert coord.merge_risk_parameters(None, None) is None
