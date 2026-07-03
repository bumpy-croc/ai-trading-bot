"""FEATURE_ENTRY_PAUSE: suppress new live entries while everything else runs.

When the flag is truthy the live engine skips opening new positions (long,
short, and the legacy duck-typed short path) so a human can flatten risk
ahead of macro events (FOMC/CPI) with a single env var. Exits, stop-loss
management, reconciliation and monitoring are unaffected — the flag is only
consulted in the entry coordinator.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest

from src.config.constants import ENTRY_PAUSE_WARNING_INTERVAL_SECONDS
from src.engines.live.execution.entry_coordinator import (
    LiveEntryCoordinator,
    LiveEntryEngineState,
)
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.shared.models import PositionSide
from src.strategies.components import SignalDirection

pytestmark = pytest.mark.fast


@pytest.fixture
def entry_pause_on(monkeypatch):
    monkeypatch.setenv("FEATURE_ENTRY_PAUSE", "true")


@pytest.fixture
def entry_pause_off(monkeypatch):
    monkeypatch.delenv("FEATURE_ENTRY_PAUSE", raising=False)


def _make_component_state(*, notional: float = 150.0) -> MagicMock:
    """Backref for the direct ComponentStrategy path of check_entry_conditions."""
    from src.strategies.components import Strategy as ComponentStrategy

    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = False
    state._is_runtime_strategy.return_value = False
    state.current_balance = 1000.0
    state.max_position_size = 1.0
    state.trading_session_id = None
    state.db_manager = None

    strategy = MagicMock(spec=ComponentStrategy)
    decision = MagicMock()
    decision.position_size = notional
    decision.signal.direction = SignalDirection.BUY
    decision.signal.strength = 0.9
    decision.signal.confidence = 0.8
    strategy.process_candle.return_value = decision
    strategy.get_stop_loss_price.return_value = 45000.0
    state.strategy = strategy

    state._extract_indicators.return_value = {}
    state._extract_sentiment_data.return_value = {}
    state._extract_ml_predictions.return_value = {}
    state._build_component_positions.return_value = []
    state._get_correlation_context.return_value = None
    state._resolve_take_profit_pct.return_value = 0.3
    state._apply_dynamic_risk_adjustment.side_effect = lambda original_size, current_time: (
        original_size
    )
    return state


def _run_entry_check(state: MagicMock) -> MagicMock:
    import pandas as pd

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    df = pd.DataFrame({"close": [50000.0]})
    coordinator.check_entry_conditions(
        df=df,
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return coordinator


# ---------------------------------------------------------------------------
# Flag ON: entries suppressed on every path
# ---------------------------------------------------------------------------


def test_pause_skips_entry_check_before_strategy_evaluation(entry_pause_on):
    state = _make_component_state()

    coordinator = _run_entry_check(state)

    coordinator.execute_entry.assert_not_called()
    state.strategy.process_candle.assert_not_called()


def test_pause_blocks_execute_entry_locked_defense_in_depth(entry_pause_on):
    """Even a direct entry-execution call is refused while paused."""
    state = create_autospec(LiveEntryEngineState, instance=True)
    state.live_entry_handler = MagicMock()
    state.live_position_tracker = MagicMock()
    state.live_position_tracker.has_position_for_symbol.return_value = False
    state.live_position_tracker.position_count = 0
    state.risk_manager = MagicMock()
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    state.max_position_size = 1.0

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry_locked(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        price=50000.0,
        stop_loss=49000.0,
        take_profit=51000.0,
        signal_strength=0.8,
        signal_confidence=0.7,
    )

    state.live_entry_handler.execute_entry.assert_not_called()


def test_pause_blocks_legacy_short_entry(entry_pause_on):
    state = create_autospec(LiveEntryEngineState, instance=True)
    state._is_runtime_strategy.return_value = False
    strategy = MagicMock()
    strategy.check_short_entry_conditions.return_value = True
    strategy.get_risk_overrides.return_value = {"position_sizer": "x"}
    state.strategy = strategy

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    coordinator.process_legacy_short_entry(
        df=MagicMock(),
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )

    coordinator.execute_entry.assert_not_called()
    strategy.check_short_entry_conditions.assert_not_called()


def test_pause_warning_is_rate_limited(entry_pause_on, caplog):
    state = _make_component_state()
    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    import pandas as pd

    df = pd.DataFrame({"close": [50000.0]})

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            coordinator.check_entry_conditions(
                df=df,
                current_index=0,
                symbol="BTCUSDT",
                current_price=50000.0,
                current_time=datetime(2024, 1, 1, tzinfo=UTC),
            )

    pause_warnings = [r for r in caplog.records if "FEATURE_ENTRY_PAUSE" in r.message]
    assert len(pause_warnings) == 1

    # After the rate-limit window elapses, the warning fires again.
    coordinator._entry_pause._last_warning = (
        time.monotonic() - ENTRY_PAUSE_WARNING_INTERVAL_SECONDS - 1
    )
    with caplog.at_level(logging.WARNING):
        coordinator.check_entry_conditions(
            df=df,
            current_index=0,
            symbol="BTCUSDT",
            current_price=50000.0,
            current_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
    pause_warnings = [r for r in caplog.records if "FEATURE_ENTRY_PAUSE" in r.message]
    assert len(pause_warnings) == 2


# ---------------------------------------------------------------------------
# Flag ON: scale-ins (exposure increases) are suppressed too
# ---------------------------------------------------------------------------


def _make_partial_ops_handler(*, should_exit: bool = False, should_scale: bool = True):
    """LiveExitHandler wired for check_partial_operations with one position."""
    position = MagicMock()
    position.symbol = "BTCUSDT"
    position.order_id = "order-1"
    position.entry_time = datetime(2024, 1, 1, tzinfo=UTC)
    position.current_size = 0.08
    position.original_size = 0.08
    position.size = 0.08

    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0
    position_tracker = MagicMock()
    position_tracker.positions = {"order-1": position}
    position_tracker.apply_partial_exit.return_value = SimpleNamespace(
        realized_pnl=1.0, new_current_size=0.04, partial_exits_taken=1
    )
    partial_manager = MagicMock()
    partial_manager.check_partial_exit.side_effect = [
        SimpleNamespace(should_exit=should_exit, exit_fraction=0.5, target_index=0),
        SimpleNamespace(should_exit=False, exit_fraction=None, target_index=None),
    ]
    partial_manager.check_scale_in.return_value = SimpleNamespace(
        should_scale=should_scale, scale_fraction=0.05, target_index=0
    )
    handler = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=position_tracker,
        execution_model=MagicMock(),
        risk_manager=None,
        partial_manager=partial_manager,
        max_position_size=0.5,
    )
    return handler, position_tracker


def test_pause_suppresses_scale_in(entry_pause_on):
    """Scale-ins increase exposure, so the pause flag blocks them."""
    handler, tracker = _make_partial_ops_handler(should_scale=True)

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_not_called()


def test_pause_keeps_partial_exits_running(entry_pause_on):
    """Partial exits reduce risk and must keep firing while paused."""
    handler, tracker = _make_partial_ops_handler(should_exit=True, should_scale=True)

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_partial_exit.assert_called_once()
    tracker.apply_scale_in.assert_not_called()


def test_flag_off_scale_in_proceeds(entry_pause_off):
    handler, tracker = _make_partial_ops_handler(should_scale=True)

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_called_once()


# ---------------------------------------------------------------------------
# Flag ON: exit path is unaffected
# ---------------------------------------------------------------------------


def test_pause_does_not_affect_exit_path(entry_pause_on):
    """Exits execute normally while entries are paused (flatten-risk use case)."""
    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0
    execution_engine.execute_exit.return_value = SimpleNamespace(
        success=True,
        executed_price=51000.0,
        exit_fee=0.0,
        slippage_cost=0.0,
        error=None,
    )

    position = MagicMock()
    position.symbol = "BTCUSDT"
    position.side = PositionSide.LONG
    position.order_id = "order-1"
    position.current_size = 0.1
    position.size = 0.1
    position.original_size = 0.1
    position.entry_price = 50000.0
    position.entry_balance = 1000.0
    position.stop_loss = 49000.0
    position.take_profit = None
    position.quantity = 0.002

    decision = SimpleNamespace(should_fill=True, fill_price=51000.0, liquidity=None, reason="")
    execution_model = MagicMock()
    execution_model.decide_fill.return_value = decision

    handler = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=MagicMock(),
        execution_model=execution_model,
    )

    result = handler.execute_exit(
        position=position,
        exit_reason="Strategy exit",
        current_price=51000.0,
        limit_price=None,
        current_balance=1000.0,
    )

    assert result.success is True
    execution_engine.execute_exit.assert_called_once()


# ---------------------------------------------------------------------------
# Flag OFF: behavior unchanged
# ---------------------------------------------------------------------------


def test_flag_off_entries_proceed_normally(entry_pause_off):
    state = _make_component_state(notional=150.0)

    coordinator = _run_entry_check(state)

    coordinator.execute_entry.assert_called_once()
    assert coordinator.execute_entry.call_args.kwargs["size"] == pytest.approx(0.15)
