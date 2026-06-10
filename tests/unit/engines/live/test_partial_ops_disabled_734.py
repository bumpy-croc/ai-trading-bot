"""#734: partial exits/scale-ins are bookkeeping-only in the live engine.

The live engine applies partial operations as pure tracker/DB mutations — no
exchange order is placed — and with mismatched units (fraction-of-original
applied to fraction-of-balance state). Until #734 lands a real implementation,
the engine must hard-disable them behind the default-OFF
``live_partial_operations`` feature flag so it cannot desync tracked size from
real holdings or book phantom PnL.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.engines.live.trading_engine import LiveTradingEngine
from src.position_management.partial_manager import PartialExitPolicy
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast


def make_engine(**kwargs) -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        engine = LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=MagicMock(),
            initial_balance=1000.0,
            **kwargs,
        )
    engine.db_manager = MagicMock()
    return engine


def _policy() -> PartialExitPolicy:
    return PartialExitPolicy(
        exit_targets=[0.03, 0.06],
        exit_sizes=[0.25, 0.25],
        scale_in_thresholds=[0.02],
        scale_in_sizes=[0.25],
        max_scale_ins=1,
    )


class TestPartialOpsDisabledByDefault:
    def test_default_engine_builds_no_partial_manager(self):
        """enable_partial_operations defaults True, but the flag gate wins."""
        engine = make_engine()
        assert engine.partial_manager is None
        assert engine.enable_partial_operations is False
        assert engine._partial_operations_opt_in is False

    def test_exit_handler_receives_no_partial_manager(self):
        engine = make_engine()
        assert engine.live_exit_handler.partial_manager is None

    def test_explicitly_passed_partial_manager_is_ignored(self):
        engine = make_engine(partial_manager=_policy(), enable_partial_operations=True)
        assert engine.partial_manager is None
        assert engine.live_exit_handler.partial_manager is None

    def test_flag_restores_partial_operations_for_development(self):
        with patch("src.engines.live.trading_engine.is_enabled", return_value=True):
            engine = make_engine(enable_partial_operations=True)
        assert engine.partial_manager is not None
        assert engine.enable_partial_operations is True
        assert engine.live_exit_handler.partial_manager is not None


class TestHotSwapCannotReenablePartialOps:
    def test_swap_overrides_with_partial_config_stay_disabled(self):
        engine = make_engine()
        engine._refresh_partial_manager_after_swap(
            {
                "partial_operations": {
                    "exit_targets": [0.03],
                    "exit_sizes": [0.25],
                }
            }
        )
        assert engine.partial_manager is None
        assert engine._partial_operations_opt_in is False
        assert engine.live_exit_handler.partial_manager is None

    def test_swap_rebuilds_policy_when_flag_enabled(self):
        with patch("src.engines.live.trading_engine.is_enabled", return_value=True):
            engine = make_engine(enable_partial_operations=True)
            engine._refresh_partial_manager_after_swap(
                {
                    "partial_operations": {
                        "exit_targets": [0.05],
                        "exit_sizes": [0.5],
                    }
                }
            )
        assert engine.partial_manager is not None
        assert engine.partial_manager.exit_targets == [0.05]
