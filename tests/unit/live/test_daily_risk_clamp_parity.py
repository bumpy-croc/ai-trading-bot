"""Live scale-in must clamp by remaining daily-risk budget for parity.

Backtest's exit_handler clamps scale-in delta_size by
``risk_manager.params.max_daily_risk - risk_manager.daily_risk_used``
(src/engines/backtest/execution/exit_handler.py:357-361). Live previously
sized scale-ins from the strategy's `scale_fraction` directly with no
budget check, so a strategy with two scale-in targets could push total
exposure above the configured per-day cap while backtest results would
have it capped — silently understating the live risk profile a backtest
predicted.

The fix mirrors backtest: after ``check_scale_in`` returns
``should_scale=True``, the live engine reads ``max_daily_risk`` and
``daily_risk_used`` off the risk manager and shrinks the scale-in
fraction to ``min(requested, remaining_daily)``. When remaining is
zero or negative, the scale-in is skipped entirely with an info log.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import LivePositionTracker, PositionSide
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy


def _make_handler(
    *, max_daily_risk: float, daily_risk_used: float
) -> tuple[LiveExitHandler, LivePositionTracker, MagicMock, MagicMock]:
    """Build a LiveExitHandler with a stubbed partial manager whose
    ``check_scale_in`` reports a positive scale request, and a risk
    manager configured with the given daily-risk numbers.
    """
    position_tracker = LivePositionTracker(db_manager=Mock())

    risk_manager = MagicMock()
    risk_manager.params = SimpleNamespace(max_daily_risk=max_daily_risk)
    risk_manager.daily_risk_used = daily_risk_used

    partial_manager = MagicMock()
    # Always-true partial-exit gate is suppressed; only scale-ins matter
    # for these tests.
    partial_manager.check_partial_exit.return_value = SimpleNamespace(
        should_exit=False, exit_fraction=0.0, target_index=0
    )
    partial_manager.check_scale_in.return_value = SimpleNamespace(
        should_scale=True, scale_fraction=0.20, target_index=0
    )

    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0

    handler = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=position_tracker,
        risk_manager=risk_manager,
        execution_model=ExecutionModel(default_fill_policy()),
        partial_manager=partial_manager,
    )

    # Spy on the actual scale-in execution.
    handler._execute_scale_in = MagicMock()
    return handler, position_tracker, risk_manager, partial_manager


def _track_position(tracker: LivePositionTracker) -> str:
    from src.engines.live.execution.position_tracker import LivePosition

    position = LivePosition(
        symbol="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        entry_time=datetime(2024, 1, 1, tzinfo=UTC),
        size=0.10,
        original_size=0.10,
        current_size=0.10,
        order_id="order-1",
    )
    tracker.track_recovered_position(position, db_id=None)
    return position.order_id


@pytest.mark.fast
class TestLiveDailyRiskClampParity:
    """Live's scale-in must respect the same daily-risk budget backtest does."""

    def test_clamps_scale_in_to_remaining_daily(self) -> None:
        """Strategy requests 20% scale-in but only 5% of daily risk
        remains — the executed delta must be 5%, not 20%."""
        handler, tracker, _rm, _pm = _make_handler(max_daily_risk=0.30, daily_risk_used=0.25)
        _track_position(tracker)

        df = pd.DataFrame({"close": [100.0]}, index=[datetime(2024, 1, 1, tzinfo=UTC)])
        handler.check_partial_operations(
            df=df,
            current_index=0,
            current_price=100.0,
            current_balance=10_000.0,
            candle_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        )

        handler._execute_scale_in.assert_called_once()
        kwargs = handler._execute_scale_in.call_args.kwargs
        # Requested 0.20, remaining 0.05 → clamp to 0.05.
        assert kwargs["delta_fraction"] == pytest.approx(0.05)
        assert kwargs["fraction_of_original"] == pytest.approx(0.05)

    def test_skips_scale_in_when_budget_exhausted(self) -> None:
        """Daily-risk budget already at the cap — scale-in must NOT execute."""
        handler, tracker, _rm, _pm = _make_handler(max_daily_risk=0.30, daily_risk_used=0.30)
        _track_position(tracker)

        df = pd.DataFrame({"close": [100.0]}, index=[datetime(2024, 1, 1, tzinfo=UTC)])
        handler.check_partial_operations(
            df=df,
            current_index=0,
            current_price=100.0,
            current_balance=10_000.0,
            candle_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        )

        handler._execute_scale_in.assert_not_called()

    def test_passes_full_request_when_budget_has_room(self) -> None:
        """Plenty of remaining budget — scale-in executes at the full
        requested fraction (no clamp)."""
        handler, tracker, _rm, _pm = _make_handler(max_daily_risk=1.0, daily_risk_used=0.10)
        _track_position(tracker)

        df = pd.DataFrame({"close": [100.0]}, index=[datetime(2024, 1, 1, tzinfo=UTC)])
        handler.check_partial_operations(
            df=df,
            current_index=0,
            current_price=100.0,
            current_balance=10_000.0,
            candle_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        )

        handler._execute_scale_in.assert_called_once()
        kwargs = handler._execute_scale_in.call_args.kwargs
        assert kwargs["delta_fraction"] == pytest.approx(0.20)

    def test_no_risk_manager_does_not_crash(self) -> None:
        """Defensive: if risk_manager is missing, scale-in still runs at
        the requested fraction (legacy behaviour)."""
        handler, tracker, _rm, _pm = _make_handler(max_daily_risk=1.0, daily_risk_used=0.0)
        handler.risk_manager = None  # simulate degraded mode
        _track_position(tracker)

        df = pd.DataFrame({"close": [100.0]}, index=[datetime(2024, 1, 1, tzinfo=UTC)])
        handler.check_partial_operations(
            df=df,
            current_index=0,
            current_price=100.0,
            current_balance=10_000.0,
            candle_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        )

        handler._execute_scale_in.assert_called_once()
        kwargs = handler._execute_scale_in.call_args.kwargs
        assert kwargs["delta_fraction"] == pytest.approx(0.20)
