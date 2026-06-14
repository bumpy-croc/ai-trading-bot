"""Contract tests for StrategyHotSwapCoordinator wiring (#486).

Behavioral coverage of the hot-swap pipeline lives in
test_strategy_hotswap_overrides.py, test_partial_ops_disabled_734.py, and
test_strategy_manager_unit.py (exercised via the engine wrappers). These tests
pin the extraction's own contract: the engine builds the coordinator bound to
itself, the public API + callback wrappers delegate to it, and the pending-update
application drives the engine through the coordinator.
"""

from unittest.mock import MagicMock, create_autospec, patch

import pytest

from src.data_providers.data_provider import DataProvider
from src.engines.live.strategy_hot_swap import StrategyHotSwapCoordinator
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast


def make_engine(**kwargs) -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=create_autospec(DataProvider, instance=True),
            initial_balance=1000.0,
            **kwargs,
        )


class TestWiring:
    def test_engine_builds_coordinator_bound_to_itself(self):
        engine = make_engine()
        assert isinstance(engine.hot_swap_coordinator, StrategyHotSwapCoordinator)
        assert engine.hot_swap_coordinator._state is engine

    def test_public_api_delegates_to_coordinator(self):
        engine = make_engine()
        engine.hot_swap_coordinator = MagicMock()
        engine.hot_swap_coordinator.hot_swap_strategy.return_value = True
        engine.hot_swap_coordinator.update_model.return_value = True

        assert engine.hot_swap_strategy("ml_basic", close_positions=True) is True
        engine.hot_swap_coordinator.hot_swap_strategy.assert_called_once_with(
            "ml_basic", close_positions=True, new_config=None
        )

        assert engine.update_model("/path/to/model") is True
        engine.hot_swap_coordinator.update_model.assert_called_once_with("/path/to/model")

    def test_callbacks_and_pending_update_delegate(self):
        engine = make_engine()
        engine.hot_swap_coordinator = MagicMock()

        engine._handle_strategy_change({"close_positions": False})
        engine.hot_swap_coordinator.handle_strategy_change.assert_called_once_with(
            {"close_positions": False}
        )

        engine._apply_pending_strategy_update()
        engine.hot_swap_coordinator.apply_pending_strategy_update.assert_called_once_with()


class TestNoStrategyManager:
    def test_hot_swap_returns_false_without_manager(self):
        # enable_hot_swapping defaults on, but a non-component strategy leaves
        # strategy_manager None; the public API must fail gracefully.
        engine = make_engine()
        engine.strategy_manager = None
        assert engine.hot_swap_strategy("ml_basic") is False
        assert engine.update_model("/path") is False

    def test_apply_pending_returns_false_without_manager(self):
        engine = make_engine()
        engine.strategy_manager = None
        assert engine._apply_pending_strategy_update() is False
