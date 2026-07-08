"""Manual system halt: entry/scale-in suppression + engine wiring (#922).

While the DB-backed ``system_halt`` flag is mirrored active, the live engine
must not INCREASE exposure — new entries (component, chokepoint, legacy short)
and scale-ins are skipped with FEATURE_ENTRY_PAUSE semantics, while exits,
partial exits, stop-loss management and reconciliation continue untouched.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pandas as pd
import pytest

from src.database.manager import SystemHaltStatus
from src.engines.live.execution.entry_coordinator import (
    LiveEntryCoordinator,
    LiveEntryEngineState,
)
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.system_halt import SystemHaltState
from src.engines.shared.models import PositionSide
from src.strategies.components import SignalDirection

pytestmark = [pytest.mark.unit, pytest.mark.fast]


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


def _run_entry_check(state: MagicMock, halt: SystemHaltState) -> LiveEntryCoordinator:
    coordinator = LiveEntryCoordinator(engine_state=state, system_halt=halt)
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
# Halt ACTIVE: entries suppressed on every path
# ---------------------------------------------------------------------------


def test_halt_skips_entry_check_before_strategy_evaluation():
    state = _make_component_state()
    halt = SystemHaltState(active=True, reason="kill-switch drill", established=True)

    coordinator = _run_entry_check(state, halt)

    coordinator.execute_entry.assert_not_called()
    state.strategy.process_candle.assert_not_called()


def test_halt_blocks_execute_entry_locked_defense_in_depth():
    """Even a direct entry-execution call is refused while halted."""
    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = False
    state.live_entry_handler = MagicMock()
    state.live_position_tracker = MagicMock()
    state.live_position_tracker.has_position_for_symbol.return_value = False
    state.live_position_tracker.position_count = 0
    state.risk_manager = MagicMock()
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    state.max_position_size = 1.0
    halt = SystemHaltState(active=True, reason="drill", established=True)

    coordinator = LiveEntryCoordinator(engine_state=state, system_halt=halt)
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


def test_halt_blocks_legacy_short_entry():
    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = False
    state._is_runtime_strategy.return_value = False
    strategy = MagicMock()
    strategy.check_short_entry_conditions.return_value = True
    strategy.get_risk_overrides.return_value = {"position_sizer": "x"}
    state.strategy = strategy
    halt = SystemHaltState(active=True, reason="drill", established=True)

    coordinator = LiveEntryCoordinator(engine_state=state, system_halt=halt)
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


def test_halt_warning_mentions_manual_halt_and_reason(caplog):
    state = _make_component_state()
    halt = SystemHaltState(active=True, reason="FOMC de-risk", established=True)

    with caplog.at_level(logging.WARNING):
        _run_entry_check(state, halt)

    halt_warnings = [r for r in caplog.records if "MANUAL SYSTEM HALT" in r.message]
    assert len(halt_warnings) == 1
    assert "FOMC de-risk" in halt_warnings[0].message


# ---------------------------------------------------------------------------
# Halt ACTIVE: scale-ins suppressed, partial exits keep running
# ---------------------------------------------------------------------------


def _make_partial_ops_handler(
    halt: SystemHaltState | None,
    *,
    should_exit: bool = False,
    should_scale: bool = True,
):
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
        system_halt=halt,
    )
    return handler, position_tracker


def test_halt_suppresses_scale_in():
    halt = SystemHaltState(active=True, reason="drill", established=True)
    handler, tracker = _make_partial_ops_handler(halt, should_scale=True)

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_not_called()


def test_halt_keeps_partial_exits_running():
    halt = SystemHaltState(active=True, reason="drill", established=True)
    handler, tracker = _make_partial_ops_handler(halt, should_exit=True, should_scale=True)

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_partial_exit.assert_called_once()
    tracker.apply_scale_in.assert_not_called()


# ---------------------------------------------------------------------------
# Halt INACTIVE: behavior unchanged
# ---------------------------------------------------------------------------


def test_inactive_halt_entries_proceed(monkeypatch):
    monkeypatch.delenv("FEATURE_ENTRY_PAUSE", raising=False)
    state = _make_component_state(notional=150.0)
    halt = SystemHaltState(established=True)

    coordinator = _run_entry_check(state, halt)

    coordinator.execute_entry.assert_called_once()


def test_inactive_halt_scale_in_proceeds(monkeypatch):
    monkeypatch.delenv("FEATURE_ENTRY_PAUSE", raising=False)
    handler, tracker = _make_partial_ops_handler(
        SystemHaltState(established=True), should_scale=True
    )

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_called_once()


# ---------------------------------------------------------------------------
# Engine wiring + in-loop behavior
# ---------------------------------------------------------------------------


def _make_engine(get_system_halt=None, **engine_kwargs):
    """Real LiveTradingEngine with mocked externals (DB, provider, strategy).

    ``get_system_halt`` configures the DB mock BEFORE construction so boot-time
    priming reads see it (a SystemHaltStatus return value or an Exception
    side effect).
    """
    with patch("src.engines.live.trading_engine.DatabaseManager") as manager_cls:
        from src.engines.live.trading_engine import LiveTradingEngine

        if isinstance(get_system_halt, Exception):
            manager_cls.return_value.get_system_halt.side_effect = get_system_halt
        elif get_system_halt is not None:
            manager_cls.return_value.get_system_halt.return_value = get_system_halt

        mock_strategy = MagicMock()
        mock_strategy.get_risk_overrides.return_value = {}
        mock_strategy.__class__.__name__ = "MockStrategy"
        mock_strategy.config = {}

        mock_dp = MagicMock()

        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_dp,
            enable_live_trading=False,
            initial_balance=1000.0,
            enable_dynamic_risk=False,
            enable_hot_swapping=False,
            **engine_kwargs,
        )
        return engine


def _run_one_loop_iteration(engine, *, entry_probe, exit_probe):
    """Drive one real `_trading_loop` iteration with the data pipeline mocked."""
    rows = 5
    idx = pd.date_range(end=datetime.now(UTC), periods=rows, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [50000.0] * rows,
            "high": [51000.0] * rows,
            "low": [49000.0] * rows,
            "close": [50500.0] * rows,
            "volume": [100.0] * rows,
        },
        index=idx,
    )

    engine._ensure_ws_health_monitor_alive = MagicMock()
    engine._drain_pending_fill_exits = MagicMock()
    engine._get_latest_data = MagicMock(return_value=df)
    engine._is_data_fresh = MagicMock(return_value=True)
    engine._prepare_strategy_dataframe = MagicMock(side_effect=lambda d: d)
    engine._is_context_ready = MagicMock(return_value=(True, "ready"))
    engine._runtime_process_decision = MagicMock(return_value=None)
    engine._check_entry_conditions = MagicMock(side_effect=entry_probe)
    engine._check_exit_conditions = MagicMock(side_effect=exit_probe)
    engine._update_performance_metrics = MagicMock()
    engine._check_max_drawdown = MagicMock()
    engine._log_periodic_account_state = MagicMock()
    engine._log_status = MagicMock()
    engine._calculate_adaptive_interval = MagicMock(return_value=0)
    engine._sleep_with_interrupt = MagicMock()
    engine.stop = MagicMock()

    engine.is_running = True
    engine._trading_loop("BTCUSDT", "1h", max_steps=1)


def test_engine_wires_shared_halt_state():
    """One SystemHaltState instance: enforcer writes it, both gates read it."""
    engine = _make_engine()

    assert engine.entry_coordinator._entry_pause._halt_state is engine._system_halt
    assert engine.live_exit_handler._entry_pause._halt_state is engine._system_halt
    assert engine._system_halt_enforcer._halt is engine._system_halt


def test_trading_loop_honors_halt_within_one_iteration():
    """A halt flag already in the DB blocks entries on the very first iteration.

    The enforcer poll runs at the top of the loop iteration, so the halt is
    mirrored BEFORE entry evaluation of that same iteration; exits still run.
    """
    engine = _make_engine()
    engine.db_manager.get_system_halt.return_value = SystemHaltStatus(
        active=True, reason="drill", source="cli:test", updated_at=None
    )

    halt_seen_by_entry_check: list[bool] = []
    halt_seen_by_exit_check: list[bool] = []
    _run_one_loop_iteration(
        engine,
        entry_probe=lambda *a, **k: halt_seen_by_entry_check.append(engine._system_halt.active),
        exit_probe=lambda *a, **k: halt_seen_by_exit_check.append(engine._system_halt.active),
    )

    # Halt mirrored from the DB before entry evaluation of the SAME iteration.
    assert halt_seen_by_entry_check == [True]
    # Exit management still ran while halted.
    assert halt_seen_by_exit_check == [True]
    # The legacy short path went through the real gate and evaluated nothing.
    engine.strategy.check_short_entry_conditions.assert_not_called()


# ---------------------------------------------------------------------------
# Fail-closed startup (#929 review P1): never-successfully-polled = halted
# ---------------------------------------------------------------------------


def test_unestablished_state_blocks_entries_fail_closed(caplog):
    """A never-successfully-polled halt state gates entries (fail closed)."""
    state = _make_component_state()
    halt = SystemHaltState()  # unestablished default

    with caplog.at_level(logging.WARNING):
        coordinator = _run_entry_check(state, halt)

    coordinator.execute_entry.assert_not_called()
    state.strategy.process_candle.assert_not_called()
    unverified = [r for r in caplog.records if "UNVERIFIED" in r.message]
    assert len(unverified) == 1


def test_unestablished_state_blocks_scale_in():
    handler, tracker = _make_partial_ops_handler(SystemHaltState(), should_scale=True)

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_not_called()


def test_boot_with_dead_db_fails_closed_until_first_successful_poll():
    """Boot while the DB is unreachable (whether or not an active halt row
    exists): entries stay blocked; exits keep running."""
    engine = _make_engine(get_system_halt=RuntimeError("db unreachable at boot"))

    assert engine._system_halt.established is False

    exit_ran: list[bool] = []
    _run_one_loop_iteration(
        engine,
        entry_probe=lambda *a, **k: None,
        exit_probe=lambda *a, **k: exit_ran.append(True),
    )

    # Still unestablished (poll failing) -> the real gates refuse new risk.
    assert engine._system_halt.established is False
    assert engine.entry_coordinator._entry_pause.paused("probe") is True
    engine.strategy.check_short_entry_conditions.assert_not_called()
    # Exits are never gated, even fail-closed.
    assert exit_ran == [True]


def test_healthy_boot_establishes_state_and_trades_immediately(monkeypatch):
    """DB up + no halt row at boot: no fail-closed window, entries evaluated."""
    monkeypatch.delenv("FEATURE_ENTRY_PAUSE", raising=False)
    engine = _make_engine(
        get_system_halt=SystemHaltStatus(active=False, reason=None, source=None, updated_at=None)
    )

    # Priming read at construction established the state before the loop.
    assert engine._system_halt.established is True
    assert engine._system_halt.active is False

    entries_evaluated: list[bool] = []
    engine.strategy.check_short_entry_conditions.return_value = False
    _run_one_loop_iteration(
        engine,
        entry_probe=lambda *a, **k: entries_evaluated.append(True),
        exit_probe=lambda *a, **k: None,
    )

    assert entries_evaluated == [True]
    # The real legacy-short gate let evaluation through (halt inactive).
    engine.strategy.check_short_entry_conditions.assert_called_once()


def test_injected_exit_handler_rebound_to_shared_halt_state():
    """A DI-injected exit handler must not bypass the manual halt (#929 review)."""
    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0
    injected = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=MagicMock(),
        execution_model=MagicMock(),
    )

    engine = _make_engine(exit_handler=injected)

    assert engine.live_exit_handler is injected
    assert injected._entry_pause._halt_state is engine._system_halt
