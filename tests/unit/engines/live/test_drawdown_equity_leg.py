"""Mark-to-market (equity) leg of the live max-drawdown hard cap (#1267).

The guard used to compare only REALIZED balance with its peak, so an open
position could run deep underwater while the cap stayed silent. It now trips on
``max(balance leg, equity leg)``: the balance leg is the original condition
(so the guard is never looser than before) and the equity leg adds unrealized
P&L. The equity leg is skipped, never guessed, when no fresh mark is available.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, create_autospec

import pytest

from src.config.constants import DEFAULT_MAX_DRAWDOWN
from src.database.models import EventType
from src.engines.live.execution.position_tracker import MarkReading
from src.engines.live.monitoring import drawdown_guard
from src.engines.live.monitoring.drawdown_guard import (
    MARK_BLIND_ESCALATE_SECONDS,
    DrawdownEngineState,
    DrawdownTier,
    MaxDrawdownEnforcer,
    MaxDrawdownGuard,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Guard arithmetic
# ---------------------------------------------------------------------------


def _guard(peak: float = 100.0, equity_peak: float | None = None) -> MaxDrawdownGuard:
    guard = MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN)
    guard.seed_peak(peak, equity_peak=equity_peak)
    return guard


def test_unrealized_loss_alone_trips_the_cap_while_balance_is_unchanged():
    guard = _guard(peak=100.0)

    assessment = guard.observe(100.0, equity=79.0)  # -$21 unrealized, balance flat

    assert assessment.balance_drawdown == pytest.approx(0.0)
    assert assessment.equity_drawdown == pytest.approx(0.21)
    assert assessment.tier is DrawdownTier.BREACH
    assert guard.tripped is True


def test_balance_only_guard_would_have_stayed_silent():
    guard = _guard(peak=100.0)

    assessment = guard.observe(100.0)  # equity unavailable: legacy behaviour

    assert assessment.tripped is False
    assert assessment.equity is None


def test_equity_leg_drives_the_tiers_below_the_cap():
    guard = _guard(peak=100.0)

    assert guard.observe(100.0, equity=89.0).tier is DrawdownTier.WARNING
    assert guard.observe(100.0, equity=83.0).tier is DrawdownTier.CRITICAL
    assert guard.tripped is False


def test_never_looser_than_balance_only_when_open_profit_masks_a_realized_loss():
    """Realized balance is down 20% but an open winner lifts equity to -15%:
    pure equity would stay quiet; the balance leg must still trip."""
    guard = _guard(peak=100.0)

    assessment = guard.observe(80.0, equity=85.0)

    assert assessment.tripped is True
    assert assessment.balance_drawdown == pytest.approx(0.20)
    assert assessment.equity_drawdown == pytest.approx(0.15)
    assert assessment.drawdown == pytest.approx(0.20)


def test_legs_are_identical_with_no_open_position():
    guard = _guard(peak=100.0)

    assessment = guard.observe(90.0, equity=90.0)

    assert assessment.balance_drawdown == pytest.approx(assessment.equity_drawdown)


def test_equity_trip_latches_after_recovery():
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=78.0)

    assessment = guard.observe(100.0, equity=100.0)

    assert assessment.tripped is True
    assert assessment.tier is DrawdownTier.BREACH


def test_equity_peak_is_floored_at_the_balance_peak():
    guard = _guard(peak=100.0, equity_peak=90.0)  # stale/low persisted equity peak

    assert guard.equity_peak == pytest.approx(100.0)


def test_equity_peak_seeds_from_the_persisted_session_equity_max():
    guard = _guard(peak=100.0, equity_peak=110.0)

    # Balance is flat at the realized peak, but equity was $110 earlier.
    assessment = guard.observe(100.0, equity=88.0)  # 20% off the $110 equity peak

    assert assessment.tripped is True
    assert assessment.peak_balance == pytest.approx(100.0)
    assert assessment.equity_peak == pytest.approx(110.0)


def test_equity_peak_needs_two_distinct_marks_before_it_ratchets():
    guard = _guard(peak=100.0)

    guard.observe(100.0, equity=130.0, mark_sequence=1)
    assert guard.equity_peak == pytest.approx(100.0)  # one reading is not confirmation

    guard.observe(100.0, equity=130.0, mark_sequence=1)  # same mark re-read
    assert guard.equity_peak == pytest.approx(100.0)

    guard.observe(100.0, equity=125.0, mark_sequence=2)  # confirmed at the LOWER value
    assert guard.equity_peak == pytest.approx(125.0)


def test_single_bad_tick_cannot_inflate_the_equity_peak_into_a_phantom_trip():
    guard = _guard(peak=100.0)

    guard.observe(100.0, equity=100.0, mark_sequence=1)
    guard.observe(100.0, equity=150.0, mark_sequence=2)  # bad tick
    assessment = guard.observe(100.0, equity=100.0, mark_sequence=3)  # real price back

    assert guard.equity_peak == pytest.approx(100.0)
    assert assessment.tripped is False


def test_giving_back_confirmed_open_profit_counts_as_drawdown():
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=110.0, mark_sequence=1)
    guard.observe(100.0, equity=110.0, mark_sequence=2)  # confirmed $110 equity peak

    assessment = guard.observe(100.0, equity=88.0, mark_sequence=3)  # 20% off $110

    assert assessment.tripped is True


def test_missing_mark_resets_peak_confirmation():
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=130.0, mark_sequence=1)
    guard.observe(100.0, equity=None)  # blind sample

    guard.observe(100.0, equity=130.0, mark_sequence=2)

    assert guard.equity_peak == pytest.approx(100.0)  # needs a fresh pair again


def test_non_finite_equity_is_treated_as_unavailable():
    guard = _guard(peak=100.0)

    assessment = guard.observe(100.0, equity=float("nan"))

    assert assessment.equity is None
    assert assessment.tripped is False


def test_reset_peak_flag_rebaselines_both_legs(monkeypatch):
    monkeypatch.setenv("FEATURE_MAX_DRAWDOWN_RESET_PEAK", "true")
    guard = MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN)

    guard.seed_peak(80.0, 100.0, equity_peak=110.0)

    assert guard.peak_balance == pytest.approx(80.0)
    assert guard.equity_peak == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Enforcer: wiring, seeding, restart, stale marks
# ---------------------------------------------------------------------------


class _Marks:
    """Controllable stand-in for LivePositionTracker.mark_reading."""

    def __init__(self, unrealized: float = 0.0, fresh: bool = True) -> None:
        self.unrealized = unrealized
        self.fresh = fresh
        self.sequence = 0
        self.raises: Exception | None = None

    def __call__(self) -> MarkReading:
        if self.raises is not None:
            raise self.raises
        return MarkReading(self.unrealized, self.fresh, self.sequence)

    def tick(self, unrealized: float | None = None, fresh: bool | None = None) -> None:
        self.sequence += 1
        if unrealized is not None:
            self.unrealized = unrealized
        if fresh is not None:
            self.fresh = fresh


def _state(
    *,
    balance: float = 100.0,
    db_peak: float | None = 100.0,
    db_equity_peak: float | None = 100.0,
) -> MagicMock:
    state = create_autospec(DrawdownEngineState, instance=True)
    state.current_balance = balance
    state.trading_session_id = 7
    state._recovered_inactive_session_id = None
    state._history_seed_session_id = None
    state._history_seed_lookup_failed = False
    state._close_only_mode = False
    state.db_manager = MagicMock()
    state.db_manager.get_session_peak_balance.return_value = db_peak
    state.db_manager.get_session_peak_equity.return_value = db_equity_peak

    def _enter(reason=None):
        state._close_only_mode = True
        state._close_only_reason = reason

    state._enter_close_only_mode.side_effect = _enter
    return state


def _enforcer(state: MagicMock, marks: _Marks | None) -> MaxDrawdownEnforcer:
    return MaxDrawdownEnforcer(
        engine_state=state,
        guard=MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN),
        mark_provider=marks,
    )


def test_open_position_with_large_unrealized_loss_trips_close_only():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=-21.0)
    enforcer = _enforcer(state, marks)

    enforcer.check()

    state._enter_close_only_mode.assert_called_once()
    alert = state._record_event.call_args
    assert alert.args[0] is EventType.ALERT
    assert alert.kwargs["error_code"] == "MAX_DRAWDOWN_BREACH"
    assert "equity leg 21.00%" in alert.args[1]
    assert "unrealized $-21.00" in alert.args[1]


def test_pre_order_chokepoint_trips_on_the_same_equity_reading():
    state = _state(balance=100.0)
    enforcer = _enforcer(state, _Marks(unrealized=-25.0))

    enforcer.check_before_new_risk()

    assert state._close_only_mode is True


def test_unrealized_loss_below_the_cap_does_not_trip():
    state = _state(balance=100.0)
    enforcer = _enforcer(state, _Marks(unrealized=-19.0))

    enforcer.check()

    state._enter_close_only_mode.assert_not_called()


def test_without_a_mark_provider_the_guard_is_balance_only():
    state = _state(balance=100.0)
    enforcer = _enforcer(state, None)

    enforcer.check()

    state._enter_close_only_mode.assert_not_called()


def test_restart_mid_drawdown_retrips_on_first_iteration_from_persisted_equity_peak():
    # Fresh process (no in-memory peak). Balance still at its realized peak; the
    # session's equity peaked at $110 and the open position is now $23 under water.
    state = _state(balance=100.0, db_peak=100.0, db_equity_peak=110.0)
    enforcer = _enforcer(state, _Marks(unrealized=-12.0))  # equity 88 = 20% off 110

    enforcer.check()

    state.db_manager.get_session_peak_equity.assert_called_once_with(
        session_id=7, fallback_session_id=None
    )
    assert enforcer.guard.equity_peak == pytest.approx(110.0)
    state._enter_close_only_mode.assert_called_once()


def test_restart_with_reset_flag_rebaselines_equity_peak_too(monkeypatch):
    monkeypatch.setenv("FEATURE_MAX_DRAWDOWN_RESET_PEAK", "true")
    state = _state(balance=100.0, db_equity_peak=110.0)
    enforcer = _enforcer(state, _Marks(unrealized=-12.0))

    enforcer.check()

    assert enforcer.guard.equity_peak == pytest.approx(100.0)
    state._enter_close_only_mode.assert_not_called()


def test_equity_peak_read_failure_keeps_balance_leg_and_retries_the_equity_peak():
    state = _state(balance=100.0, db_peak=100.0)
    state.db_manager.get_session_peak_equity.side_effect = [RuntimeError("db hiccup"), 110.0]
    enforcer = _enforcer(state, _Marks(unrealized=0.0))

    enforcer.check()  # armed on the balance peak; equity peak read failed
    assert enforcer.guard.seeded is True
    assert enforcer.guard.equity_peak == pytest.approx(100.0)

    enforcer.check()  # upgrade retry succeeds
    assert enforcer.guard.equity_peak == pytest.approx(110.0)
    state._enter_close_only_mode.assert_not_called()


def test_stale_mark_degrades_to_balance_leg_with_a_warning(caplog):
    state = _state(balance=100.0)
    marks = _Marks(unrealized=-30.0, fresh=False)  # untrustworthy: must NOT trip or be used
    enforcer = _enforcer(state, marks)

    with caplog.at_level(logging.WARNING):
        enforcer.check()
        enforcer.check()

    state._enter_close_only_mode.assert_not_called()
    blind = [r for r in caplog.records if "equity leg blind" in r.message]
    assert len(blind) == 1  # logged on the transition, not every iteration


def test_balance_leg_still_trips_while_the_equity_leg_is_blind():
    state = _state(balance=78.0, db_peak=100.0, db_equity_peak=100.0)
    enforcer = _enforcer(state, _Marks(fresh=False))

    enforcer.check()

    state._enter_close_only_mode.assert_called_once()


def test_provider_exception_degrades_instead_of_crashing_the_check():
    state = _state(balance=78.0, db_peak=100.0)
    marks = _Marks()
    marks.raises = RuntimeError("tracker exploded")
    enforcer = _enforcer(state, marks)

    enforcer.check()

    state._enter_close_only_mode.assert_called_once()  # balance leg unaffected


def test_fresh_marks_resume_the_equity_leg_after_a_blind_spell():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=-30.0, fresh=False)
    enforcer = _enforcer(state, marks)
    enforcer.check()
    state._enter_close_only_mode.assert_not_called()

    marks.tick(fresh=True)
    enforcer.check()

    state._enter_close_only_mode.assert_called_once()


def test_persistent_blindness_pages_the_operator_once(monkeypatch):
    state = _state(balance=100.0)
    marks = _Marks(unrealized=-5.0, fresh=False)
    enforcer = _enforcer(state, marks)
    clock = [1000.0]
    monkeypatch.setattr(drawdown_guard.time, "monotonic", lambda: clock[0])

    enforcer.check()
    clock[0] += MARK_BLIND_ESCALATE_SECONDS - 1
    enforcer.check()
    state._record_event.assert_not_called()

    clock[0] += 2
    enforcer.check()
    enforcer.check()

    state._record_event.assert_called_once()
    kwargs = state._record_event.call_args.kwargs
    assert kwargs["error_code"] == "DRAWDOWN_EQUITY_MARK_UNAVAILABLE"
    assert kwargs["alert"] is True
    state._enter_close_only_mode.assert_not_called()  # a blind spell alone never latches


def test_bad_tick_does_not_inflate_the_enforcer_equity_peak():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=0.0)
    enforcer = _enforcer(state, marks)
    enforcer.check()

    marks.tick(unrealized=50.0)  # bad tick
    enforcer.check()
    marks.tick(unrealized=0.0)
    enforcer.check()

    assert enforcer.guard.equity_peak == pytest.approx(100.0)
    state._enter_close_only_mode.assert_not_called()


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_enforcer_reads_marks_from_the_live_position_tracker():
    """Real engine, real tracker: an open position priced deep underwater with an
    unchanged realized balance latches close-only."""
    from datetime import UTC, datetime
    from unittest.mock import patch

    from src.engines.live.execution.position_tracker import LivePosition, PositionSide
    from src.engines.live.trading_engine import LiveTradingEngine

    db = MagicMock()
    db.get_active_session_id.return_value = 20
    db.get_current_balance.return_value = 100.0
    db.get_session_peak_balance.return_value = 100.0
    db.get_session_peak_equity.return_value = 100.0

    with (
        patch("src.engines.live.trading_engine.DatabaseManager", return_value=db),
        patch("src.engines.live.trading_engine.get_config", return_value={}),
        patch(
            "src.engines.live.trading_engine._create_exchange_provider",
            return_value=(MagicMock(), "mock"),
        ),
    ):
        engine = LiveTradingEngine(
            strategy=MagicMock(),
            data_provider=MagicMock(),
            initial_balance=100.0,
            enable_live_trading=True,
            resume_from_last_balance=True,
        )

    engine.trading_session_id = 20  # normally resolved in start()
    engine.live_position_tracker.open_position(
        LivePosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            size=1.0,  # fully invested: a 25% adverse move is a 25% equity loss
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            entry_balance=100.0,
            order_id="wiring-1",
        )
    )
    engine.live_position_tracker.update_pnl(75.0, fallback_balance=100.0)
    assert engine.current_balance == pytest.approx(100.0)

    engine._check_max_drawdown()

    assert engine._close_only_mode is True
