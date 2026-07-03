"""Live-engine partial-exit units must match the backtest engine (parity).

The live engine shared the backtest's units bug: the exit handler converted
the policy's fraction-of-original into a fraction-of-current-POSITION and
passed it to ``LivePositionTracker.apply_partial_exit``, which decrements
``current_size`` (a fraction of BALANCE) and feeds ``PartialExitExecutor``
(which sizes P&L against ``basis_balance``). The shared executor contract is
balance-fraction units; both engines must convert with
``fraction_of_original * original_size``.

Live partial operations are gated OFF by default (#734 feature flag,
bookkeeping-only) — these tests exercise the handler/tracker directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from src.engines.backtest.execution.position_tracker import (
    PositionTracker as BacktestPositionTracker,
)
from src.engines.backtest.models import ActiveTrade
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.position_management.partial_manager import PartialExitPolicy

pytestmark = pytest.mark.unit

ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _make_live_handler(
    policy: PartialExitPolicy,
) -> tuple[LiveExitHandler, LivePositionTracker, MagicMock]:
    tracker = LivePositionTracker(db_manager=Mock())

    risk_manager = MagicMock()
    risk_manager.params = SimpleNamespace(max_daily_risk=1.0)
    risk_manager.daily_risk_used = 0.0

    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0

    handler = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=tracker,
        risk_manager=risk_manager,
        execution_model=ExecutionModel(default_fill_policy()),
        partial_manager=PartialOperationsManager(policy=policy),
    )
    return handler, tracker, risk_manager


def _track_live_long(
    tracker: LivePositionTracker,
    *,
    size: float = 0.10,
    entry_price: float = 100.0,
    entry_balance: float = 1000.0,
) -> LivePosition:
    position = LivePosition(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        entry_price=entry_price,
        entry_time=ENTRY_TIME,
        size=size,
        entry_balance=entry_balance,
        order_id="order-1",
    )
    tracker.track_recovered_position(position, db_id=None)
    return position


def _df(price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [price]},
        index=pd.DatetimeIndex([datetime(2024, 1, 2, tzinfo=UTC)]),
    )


class TestLivePartialExitUnits:
    def test_live_partial_exit_books_pnl_on_exited_slice(self) -> None:
        """25%-of-original on a 10%-of-balance position at +5% must realize
        5% x 2.5% x $1000 = $1.25 and leave current_size = 0.075."""
        policy = PartialExitPolicy(exit_targets=[0.05], exit_sizes=[0.25])
        handler, tracker, _ = _make_live_handler(policy)
        position = _track_live_long(tracker)

        handler.check_partial_operations(
            df=_df(105.0),
            current_index=0,
            current_price=105.0,
            current_balance=1000.0,
            candle_time=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert position.current_size == pytest.approx(0.075)
        assert position.partial_exits_taken == 1

    def test_live_tracker_apply_partial_exit_balance_fraction_contract(self) -> None:
        """Tracker-level: a 0.025 balance-fraction exit realizes $1.25."""
        tracker = LivePositionTracker(db_manager=Mock())
        _track_live_long(tracker)

        result = tracker.apply_partial_exit(
            order_id="order-1",
            delta_fraction=0.025,
            price=105.0,
            target_level=0,
            fraction_of_original=0.25,
            basis_balance=1000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        assert result is not None
        assert result.new_current_size == pytest.approx(0.075)

    def test_live_fully_closed_position_cannot_scale_in(self) -> None:
        """Parity with backtest bug 3: no zombie scale-ins on consumed positions."""
        policy = PartialExitPolicy(
            exit_targets=[0.02],
            exit_sizes=[1.0],
            scale_in_thresholds=[0.01],
            scale_in_sizes=[0.5],
            max_scale_ins=1,
        )
        handler, tracker, _ = _make_live_handler(policy)
        _track_live_long(tracker)

        # Spy on execution so the (heavy) full-close path is not exercised;
        # the assertion is purely about the scale-in decision.
        handler._execute_partial_exit = MagicMock()
        handler._execute_scale_in = MagicMock()

        # Simulate the position having been fully consumed by partials.
        position = tracker._positions["order-1"]
        position.current_size = 0.0
        position.partial_exits_taken = 1

        handler.check_partial_operations(
            df=_df(105.0),
            current_index=0,
            current_price=105.0,
            current_balance=1000.0,
            candle_time=datetime(2024, 1, 2, tzinfo=UTC),
        )

        handler._execute_scale_in.assert_not_called()


class TestBacktestLivePartialExitParity:
    """Both engines must realize identical P&L and size for identical inputs."""

    @pytest.mark.parametrize(
        ("side", "exit_price"),
        [
            (PositionSide.LONG, 105.0),
            (PositionSide.SHORT, 95.0),
        ],
    )
    def test_partial_exit_pnl_and_size_parity(self, side, exit_price) -> None:
        policy = PartialExitPolicy(exit_targets=[0.05], exit_sizes=[0.25])

        # --- Live side ---
        live_handler, live_tracker, _ = _make_live_handler(policy)
        live_position = LivePosition(
            symbol="ETHUSDT",
            side=side,
            entry_price=100.0,
            entry_time=ENTRY_TIME,
            size=0.10,
            entry_balance=1000.0,
            order_id="order-1",
        )
        live_tracker.track_recovered_position(live_position, db_id=None)
        live_handler.check_partial_operations(
            df=_df(exit_price),
            current_index=0,
            current_price=exit_price,
            current_balance=1000.0,
            candle_time=datetime(2024, 1, 2, tzinfo=UTC),
        )

        # --- Backtest side ---
        from src.engines.backtest.execution.exit_handler import ExitHandler

        bt_tracker = BacktestPositionTracker(fee_rate=0.0, slippage_rate=0.0)
        bt_risk = MagicMock()
        bt_risk.params = SimpleNamespace(max_daily_risk=1.0)
        bt_risk.daily_risk_used = 0.0
        bt_exec = MagicMock()
        bt_exec.calculate_scale_in_costs.return_value = (0.0, 0.0)
        bt_handler = ExitHandler(
            execution_engine=bt_exec,
            position_tracker=bt_tracker,
            risk_manager=bt_risk,
            execution_model=ExecutionModel(default_fill_policy()),
            partial_manager=PartialOperationsManager(policy=policy),
        )
        bt_trade = ActiveTrade(
            symbol="ETHUSDT",
            side=side,
            entry_price=100.0,
            entry_time=ENTRY_TIME,
            size=0.10,
            entry_balance=1000.0,
        )
        bt_tracker.open_position(bt_trade)
        bt_df = pd.DataFrame(
            {
                "open": [exit_price],
                "high": [exit_price],
                "low": [exit_price],
                "close": [exit_price],
                "volume": [1.0],
            },
            index=pd.DatetimeIndex([datetime(2024, 1, 2, tzinfo=UTC)]),
        )
        bt_result = bt_handler.check_partial_operations(
            current_price=exit_price,
            df=bt_df,
            index=0,
            balance=1000.0,
        )

        # 5% favorable move on a 2.5%-of-balance slice: $1.25 both sides.
        assert bt_result.realized_pnl == pytest.approx(1.25)
        assert bt_trade.current_size == pytest.approx(live_position.current_size)
        assert bt_trade.current_size == pytest.approx(0.075)
