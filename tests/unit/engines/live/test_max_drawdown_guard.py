"""Live enforcement of the portfolio max-drawdown hard cap.

`RiskManager.check_drawdown()` had zero callers, so nothing halted the live
engine when account drawdown reached ``max_drawdown_pct`` (0.20 in
risk-limits.json / constants). The MaxDrawdownGuard + MaxDrawdownEnforcer pair
computes rolling drawdown from the session peak balance on every trading-loop
iteration and trips the EXISTING close-only mode at the cap: entries stop,
exits/stop-losses keep running, and the trip is latched until an operator
intervenes (restart with FEATURE_MAX_DRAWDOWN_RESET_PEAK to re-baseline).

Peak equity source: same numbers account_history.drawdown is derived from
(balance vs session peak balance), recomputed from account_history on boot so
restarts don't reset the baseline.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, create_autospec

import pytest

from src.config.constants import (
    DEFAULT_MAX_DRAWDOWN,
    DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT,
    DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS,
    DRAWDOWN_WARNING_AT_PCT_OF_LIMIT,
)
from src.database.models import EventType
from src.engines.live.monitoring.drawdown_guard import (
    DrawdownTier,
    MaxDrawdownEnforcer,
    MaxDrawdownGuard,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Guard: tier + trip arithmetic
# ---------------------------------------------------------------------------


def _seeded_guard(peak: float = 100.0, **kwargs) -> MaxDrawdownGuard:
    guard = MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN, **kwargs)
    guard.seed_peak(peak)
    return guard


def test_trips_at_exactly_the_cap():
    guard = _seeded_guard(peak=100.0)

    assessment = guard.observe(80.0)  # exactly 20% drawdown

    assert assessment.tier is DrawdownTier.BREACH
    assert assessment.tripped is True
    assert guard.tripped is True


def test_no_trip_just_below_the_cap():
    guard = _seeded_guard(peak=100.0)

    assessment = guard.observe(80.1)  # 19.9% drawdown

    assert assessment.tier is DrawdownTier.CRITICAL
    assert assessment.tripped is False
    assert guard.tripped is False


def test_warning_tier_at_half_of_limit():
    guard = _seeded_guard(peak=100.0)

    # exactly limit * warning fraction = 10% drawdown
    assessment = guard.observe(
        100.0 * (1 - DEFAULT_MAX_DRAWDOWN * DRAWDOWN_WARNING_AT_PCT_OF_LIMIT)
    )

    assert assessment.tier is DrawdownTier.WARNING
    assert assessment.tripped is False


def test_critical_tier_at_eighty_pct_of_limit():
    guard = _seeded_guard(peak=100.0)

    # exactly limit * critical fraction = 16% drawdown; float products like
    # 0.20 * 0.80 land 1 ulp off, so an exact reading must still register.
    assessment = guard.observe(
        100.0 * (1 - DEFAULT_MAX_DRAWDOWN * DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT)
    )

    assert assessment.tier is DrawdownTier.CRITICAL
    assert assessment.tripped is False


def test_no_tier_below_warning_threshold():
    guard = _seeded_guard(peak=100.0)

    assessment = guard.observe(95.0)  # 5% drawdown

    assert assessment.tier is DrawdownTier.NONE


def test_trip_latches_even_if_balance_recovers():
    guard = _seeded_guard(peak=100.0)
    guard.observe(80.0)  # trip

    assessment = guard.observe(95.0)  # recovered to 5% drawdown

    assert assessment.tripped is True
    assert assessment.tier is DrawdownTier.BREACH


def test_peak_ratchets_up_with_new_highs():
    guard = _seeded_guard(peak=100.0)

    guard.observe(120.0)
    assert guard.peak_balance == pytest.approx(120.0)

    # 20% off the NEW peak trips, even though it is only 4% off the old one
    assessment = guard.observe(96.0)
    assert assessment.tripped is True


def test_seed_peak_takes_max_of_valid_candidates():
    guard = MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN)

    guard.seed_peak(80.0, 100.0, None, float("nan"), -5.0, 90.0)

    assert guard.peak_balance == pytest.approx(100.0)


def test_non_finite_balance_sample_is_ignored():
    guard = _seeded_guard(peak=100.0)

    assessment = guard.observe(float("nan"))

    assert assessment.tripped is False
    assert guard.peak_balance == pytest.approx(100.0)

    # A garbage sample must not have corrupted subsequent arithmetic.
    assert guard.observe(80.0).tripped is True


def test_reset_peak_flag_rebaselines_to_current_balance(monkeypatch):
    monkeypatch.setenv("FEATURE_MAX_DRAWDOWN_RESET_PEAK", "true")
    guard = MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN)

    guard.seed_peak(80.0, 100.0)  # historical peak 100 ignored under override

    assert guard.peak_balance == pytest.approx(80.0)
    assert guard.observe(80.0).tripped is False
    # Still armed from the new baseline: a further 20% drawdown trips.
    assert guard.observe(64.0).tripped is True


def test_tier_logs_are_rate_limited(caplog):
    guard = _seeded_guard(peak=100.0)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            guard.observe(90.0)  # warning tier every iteration

    warnings = [r for r in caplog.records if "drawdown" in r.message.lower()]
    assert len(warnings) == 1

    # After the rate-limit window elapses the tier logs again.
    guard._last_tier_log = time.monotonic() - DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS - 1
    with caplog.at_level(logging.WARNING):
        guard.observe(90.0)
    warnings = [r for r in caplog.records if "drawdown" in r.message.lower()]
    assert len(warnings) == 2


def test_tier_escalation_logs_immediately_despite_rate_limit(caplog):
    guard = _seeded_guard(peak=100.0)

    with caplog.at_level(logging.WARNING):
        guard.observe(90.0)  # WARNING tier
        guard.observe(83.0)  # CRITICAL tier — escalation logs immediately

    criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(criticals) == 1


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        MaxDrawdownGuard(0.0)
    with pytest.raises(ValueError):
        MaxDrawdownGuard(1.5)


# ---------------------------------------------------------------------------
# Enforcer: engine integration (close-only trip, events, idempotence)
# ---------------------------------------------------------------------------


def _make_state(
    *,
    balance: float = 80.0,
    db_peak: float | None = 100.0,
    tracker_peak: float = 80.0,
    session_id: int | None = 7,
) -> MagicMock:
    from src.engines.live.monitoring.drawdown_guard import DrawdownEngineState

    state = create_autospec(DrawdownEngineState, instance=True)
    state.current_balance = balance
    state.trading_session_id = session_id
    state._recovered_inactive_session_id = None
    state._close_only_mode = False
    state.db_manager = MagicMock()
    state.db_manager.get_session_peak_balance.return_value = db_peak
    state.performance_tracker = MagicMock()
    metrics = MagicMock()
    metrics.peak_balance = tracker_peak
    state.performance_tracker.get_metrics.return_value = metrics

    def _enter():
        state._close_only_mode = True

    state._enter_close_only_mode.side_effect = _enter
    return state


def _make_enforcer(state: MagicMock) -> MaxDrawdownEnforcer:
    return MaxDrawdownEnforcer(engine_state=state, guard=MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN))


def test_breach_enters_close_only_and_pages_operator():
    state = _make_state(balance=80.0, db_peak=100.0)
    enforcer = _make_enforcer(state)

    enforcer.check()

    state._enter_close_only_mode.assert_called_once()
    state._record_event.assert_called_once()
    args, kwargs = state._record_event.call_args
    assert args[0] is EventType.ALERT
    assert kwargs["severity"] == "critical"
    assert kwargs["component"] == "risk"
    assert kwargs["error_code"] == "MAX_DRAWDOWN_BREACH"
    assert kwargs["alert"] is True


def test_no_trip_below_cap():
    state = _make_state(balance=80.1, db_peak=100.0)
    enforcer = _make_enforcer(state)

    enforcer.check()

    state._enter_close_only_mode.assert_not_called()
    state._record_event.assert_not_called()


def test_trip_is_idempotent_across_loop_iterations():
    state = _make_state(balance=80.0, db_peak=100.0)
    enforcer = _make_enforcer(state)

    for _ in range(5):
        enforcer.check()

    state._enter_close_only_mode.assert_called_once()
    state._record_event.assert_called_once()


def test_retrips_if_resumed_without_operator_override():
    """resume_trading() alone cannot clear the halt while still in breach."""
    state = _make_state(balance=80.0, db_peak=100.0)
    enforcer = _make_enforcer(state)
    enforcer.check()

    state._close_only_mode = False  # human called resume_trading()
    enforcer.check()

    assert state._enter_close_only_mode.call_count == 2
    assert state._record_event.call_count == 2


def test_restart_recompute_re_trips_from_persisted_peak():
    """A restart mid-breach recomputes the peak from account_history and
    re-trips naturally — the halt survives restarts without persisted flags."""
    state = _make_state(balance=78.0, db_peak=100.0, tracker_peak=78.0)
    enforcer = _make_enforcer(state)  # fresh process: guard has no memory

    enforcer.check()

    state.db_manager.get_session_peak_balance.assert_called_once_with(7, fallback_session_id=None)
    state._enter_close_only_mode.assert_called_once()


def test_restart_with_reset_peak_override_stays_trading(monkeypatch):
    """Documented clear procedure: restart with FEATURE_MAX_DRAWDOWN_RESET_PEAK
    re-baselines the peak to the current balance instead of halting."""
    monkeypatch.setenv("FEATURE_MAX_DRAWDOWN_RESET_PEAK", "true")
    state = _make_state(balance=78.0, db_peak=100.0, tracker_peak=78.0)
    enforcer = _make_enforcer(state)

    enforcer.check()

    state._enter_close_only_mode.assert_not_called()
    assert enforcer.guard.peak_balance == pytest.approx(78.0)


def test_breach_already_in_close_only_still_records_cause():
    """If close-only was entered for another reason (e.g. DB outage) the
    drawdown breach must still be recorded as a cause, exactly once."""
    state = _make_state(balance=80.0, db_peak=100.0)
    state._close_only_mode = True
    enforcer = _make_enforcer(state)

    enforcer.check()
    enforcer.check()

    state._record_event.assert_called_once()


def test_db_seed_failure_fails_soft_to_in_memory_peak():
    state = _make_state(balance=90.0, db_peak=None, tracker_peak=90.0)
    state.db_manager.get_session_peak_balance.side_effect = RuntimeError("db down")

    enforcer = _make_enforcer(state)
    enforcer.check()  # must not raise; seeds from tracker/balance → 0% drawdown

    state._enter_close_only_mode.assert_not_called()


def test_check_never_raises_into_the_trading_loop():
    state = _make_state()
    state.performance_tracker.get_metrics.side_effect = RuntimeError("boom")
    state.current_balance = "not-a-number"  # type: ignore[assignment]

    enforcer = _make_enforcer(state)
    enforcer.check()  # swallowed + logged


def test_no_db_session_seeds_from_in_memory_only():
    state = _make_state(balance=100.0, session_id=None, tracker_peak=100.0)
    enforcer = _make_enforcer(state)

    enforcer.check()

    state.db_manager.get_session_peak_balance.assert_not_called()
    state._enter_close_only_mode.assert_not_called()


# ---------------------------------------------------------------------------
# While tripped: entries blocked, exits/stop-loss calls pass through
# ---------------------------------------------------------------------------


def test_close_only_mode_blocks_entry_evaluation():
    """The enforcer reuses the existing close-only gate at the entry seam."""
    from datetime import UTC, datetime

    import pandas as pd

    from src.engines.live.execution.entry_coordinator import (
        LiveEntryCoordinator,
        LiveEntryEngineState,
    )

    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = True  # what the drawdown trip sets
    strategy = MagicMock()
    state.strategy = strategy

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    coordinator.check_entry_conditions(
        df=pd.DataFrame({"close": [50000.0]}),
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )

    coordinator.execute_entry.assert_not_called()
    strategy.process_candle.assert_not_called()


def test_exits_still_execute_while_tripped():
    """Close-only must never block risk-reducing exits or stop-loss handling."""
    from types import SimpleNamespace

    from src.engines.live.execution.exit_handler import LiveExitHandler
    from src.engines.shared.models import PositionSide

    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0
    execution_engine.execute_exit.return_value = SimpleNamespace(
        success=True, executed_price=51000.0, exit_fee=0.0, slippage_cost=0.0, error=None
    )
    execution_model = MagicMock()
    execution_model.decide_fill.return_value = SimpleNamespace(
        should_fill=True, fill_price=51000.0, liquidity=None, reason=""
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

    handler = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=MagicMock(),
        execution_model=execution_model,
    )

    # The exit path has no dependency on close-only state or the guard at all:
    # this exit executes exactly as it would in normal mode.
    result = handler.execute_exit(
        position=position,
        exit_reason="Stop loss",
        current_price=51000.0,
        limit_price=None,
        current_balance=800.0,  # tripped account, 20% down
    )

    assert result.success is True
    execution_engine.execute_exit.assert_called_once()
