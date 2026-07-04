"""Backtest scale-ins must respect the max-position cap like live (#835).

PR #835 gave the LIVE scale-in path a max-position headroom clamp
(never-shrink semantics: ``min(size + delta, max(cap, size))``). The
backtest scale-in path previously bounded scale-ins only by the remaining
daily-risk budget and hard-capped the tracker at 1.0, so a backtest could
grow positions past ``--max-position`` while live would clamp — a silent
parity break in reported exposure.

These tests pin the backtest semantics to #835's spec:

1. scale-in under the cap passes through unclamped;
2. scale-in exceeding headroom is clamped to headroom;
3. an at-cap position gets its scale-in zeroed (skipped);
4. an over-cap position is never shrunk by a scale-in.

The live-vs-backtest comparison cases activate automatically once #835's
live-side clamp is present on this branch (skipped until then).
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.models import ActiveTrade
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.models import PositionSide
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.position_management.partial_manager import PartialExitPolicy

pytestmark = pytest.mark.unit

ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)

# Live gains its max-position headroom clamp in #835; until that lands on
# this branch the cross-engine comparisons are skipped (backtest-side
# semantics are still pinned below). Source inspection is used because the
# max_position_size constructor param predates the clamp itself.
LIVE_HAS_MAX_POSITION_CLAMP = "headroom" in inspect.getsource(
    LiveExitHandler.check_partial_operations
)


def _scale_in_policy() -> PartialExitPolicy:
    """Policy requesting a 50%-of-original scale-in at +1%."""
    return PartialExitPolicy(
        exit_targets=[0.99],
        exit_sizes=[0.25],
        scale_in_thresholds=[0.01],
        scale_in_sizes=[0.50],
        max_scale_ins=1,
    )


def _make_backtest_handler(
    *,
    max_position_size: float,
    max_daily_risk: float = 1.0,
) -> tuple[ExitHandler, PositionTracker]:
    tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
    risk_manager = MagicMock()
    risk_manager.params = SimpleNamespace(max_daily_risk=max_daily_risk)
    risk_manager.daily_risk_used = 0.0
    execution_engine = MagicMock()
    execution_engine.calculate_scale_in_costs.return_value = (0.0, 0.0)
    handler = ExitHandler(
        execution_engine=execution_engine,
        position_tracker=tracker,
        risk_manager=risk_manager,
        execution_model=ExecutionModel(default_fill_policy()),
        partial_manager=PartialOperationsManager(policy=_scale_in_policy()),
        max_position_size=max_position_size,
    )
    return handler, tracker


def _open_backtest_long(tracker: PositionTracker, *, size: float) -> ActiveTrade:
    trade = ActiveTrade(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        entry_time=ENTRY_TIME,
        size=size,
        entry_balance=1000.0,
    )
    tracker.open_position(trade)
    return trade


def _run_backtest_scale_in(handler: ExitHandler) -> None:
    df = pd.DataFrame(
        {
            "open": [102.0],
            "high": [102.0],
            "low": [102.0],
            "close": [102.0],
            "volume": [1.0],
        },
        index=pd.DatetimeIndex([datetime(2024, 1, 2, tzinfo=UTC)]),
    )
    handler.check_partial_operations(
        current_price=102.0,
        df=df,
        index=0,
        balance=1000.0,
    )


class TestBacktestMaxPositionCap:
    """Backtest scale-in cap semantics pinned to #835's live spec."""

    def test_scale_in_under_cap_passes_through(self) -> None:
        # 0.10 position, +50% of original = +0.05 → 0.15, cap 0.20: no clamp.
        handler, tracker = _make_backtest_handler(max_position_size=0.20)
        trade = _open_backtest_long(tracker, size=0.10)

        _run_backtest_scale_in(handler)

        assert trade.scale_ins_taken == 1
        assert trade.current_size == pytest.approx(0.15)
        assert trade.size == pytest.approx(0.15)

    def test_scale_in_clamped_to_headroom(self) -> None:
        # 0.10 position, +0.05 requested, cap 0.12 → only 0.02 headroom.
        handler, tracker = _make_backtest_handler(max_position_size=0.12)
        trade = _open_backtest_long(tracker, size=0.10)

        _run_backtest_scale_in(handler)

        assert trade.scale_ins_taken == 1
        assert trade.current_size == pytest.approx(0.12)
        assert trade.size == pytest.approx(0.12)

    def test_scale_in_skipped_at_cap(self) -> None:
        # Position already at the cap: delta zeroed, scale-in skipped.
        handler, tracker = _make_backtest_handler(max_position_size=0.10)
        trade = _open_backtest_long(tracker, size=0.10)

        _run_backtest_scale_in(handler)

        assert trade.scale_ins_taken == 0
        assert trade.current_size == pytest.approx(0.10)
        assert trade.size == pytest.approx(0.10)

    def test_over_cap_position_never_shrunk(self) -> None:
        # An over-cap position (e.g. cap lowered mid-run) must keep its
        # tracked exposure — the clamp only prevents growth.
        handler, tracker = _make_backtest_handler(max_position_size=0.05)
        trade = _open_backtest_long(tracker, size=0.10)

        _run_backtest_scale_in(handler)

        assert trade.scale_ins_taken == 0
        assert trade.current_size == pytest.approx(0.10)
        assert trade.size == pytest.approx(0.10)

    def test_tracker_never_shrink_semantics(self) -> None:
        # Tracker-level: even if a positive delta reaches an over-cap
        # position, size must not shrink below its current value.
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open_backtest_long(tracker, size=0.10)

        tracker.apply_scale_in(0.05, max_position_size=0.05)

        assert trade.current_size == pytest.approx(0.10)
        assert trade.size == pytest.approx(0.10)


@pytest.mark.skipif(
    not LIVE_HAS_MAX_POSITION_CLAMP,
    reason="live-side max-position scale-in clamp lands with #835",
)
class TestMaxPositionCapParity:
    """Live and backtest must clamp scale-ins identically."""

    @pytest.mark.parametrize(
        ("position_size", "cap", "expected_size", "expected_scale_ins"),
        [
            (0.10, 0.20, 0.15, 1),  # under cap: full +0.05
            (0.10, 0.12, 0.12, 1),  # clamped to headroom
            (0.10, 0.10, 0.10, 0),  # at cap: skipped
            (0.10, 0.05, 0.10, 0),  # over cap: never shrunk
        ],
    )
    def test_scale_in_clamp_parity(
        self, position_size, cap, expected_size, expected_scale_ins
    ) -> None:
        from unittest.mock import Mock

        from src.engines.live.execution.position_tracker import (
            LivePosition,
            LivePositionTracker,
        )

        # --- Backtest side ---
        bt_handler, bt_tracker = _make_backtest_handler(max_position_size=cap)
        bt_trade = _open_backtest_long(bt_tracker, size=position_size)
        _run_backtest_scale_in(bt_handler)

        # --- Live side ---
        live_tracker = LivePositionTracker(db_manager=Mock())
        live_risk = MagicMock()
        live_risk.params = SimpleNamespace(max_daily_risk=1.0)
        live_risk.daily_risk_used = 0.0
        live_exec = MagicMock()
        live_exec.fee_rate = 0.0
        live_exec.slippage_rate = 0.0
        live_handler = LiveExitHandler(
            execution_engine=live_exec,
            position_tracker=live_tracker,
            risk_manager=live_risk,
            execution_model=ExecutionModel(default_fill_policy()),
            partial_manager=PartialOperationsManager(policy=_scale_in_policy()),
            max_position_size=cap,
        )
        live_position = LivePosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_time=ENTRY_TIME,
            size=position_size,
            entry_balance=1000.0,
            order_id="order-1",
        )
        live_tracker.track_recovered_position(live_position, db_id=None)
        live_handler.check_partial_operations(
            df=pd.DataFrame(
                {"close": [102.0]},
                index=pd.DatetimeIndex([datetime(2024, 1, 2, tzinfo=UTC)]),
            ),
            current_index=0,
            current_price=102.0,
            current_balance=1000.0,
            candle_time=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert bt_trade.current_size == pytest.approx(expected_size)
        assert bt_trade.scale_ins_taken == expected_scale_ins
        assert live_position.current_size == pytest.approx(bt_trade.current_size)
        assert live_position.scale_ins_taken == bt_trade.scale_ins_taken
