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
    warning = _guard(peak=100.0)
    critical = _guard(peak=100.0)

    assert warning.observe(100.0, equity=89.0).tier is DrawdownTier.WARNING
    assert critical.observe(100.0, equity=83.0).tier is DrawdownTier.CRITICAL
    assert warning.tripped is False and critical.tripped is False


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
    guard.observe(100.0, equity=88.0, mark_sequence=3)  # 20% off $110, first low mark

    assessment = guard.observe(100.0, equity=88.0, mark_sequence=4)  # confirmed

    assert assessment.tripped is True


def test_single_low_mark_does_not_latch_the_halt():
    """One bad kline close must not force an operator restart."""
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=100.0, mark_sequence=1)

    spike = guard.observe(100.0, equity=79.0, mark_sequence=2)
    recovered = guard.observe(100.0, equity=100.0, mark_sequence=3)

    assert spike.tripped is False
    assert recovered.tripped is False


def test_equity_breach_latches_once_a_second_distinct_mark_agrees():
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=100.0, mark_sequence=1)
    guard.observe(100.0, equity=79.0, mark_sequence=2)

    assessment = guard.observe(100.0, equity=79.0, mark_sequence=3)

    assert assessment.tripped is True


def test_rereading_one_low_mark_is_not_a_second_confirmation():
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=100.0, mark_sequence=1)

    for _ in range(3):
        assessment = guard.observe(100.0, equity=79.0, mark_sequence=2)

    assert assessment.tripped is False


def test_first_mark_after_boot_stands_alone_so_restart_mid_breach_retrips():
    guard = _guard(peak=100.0)

    assert guard.observe(100.0, equity=79.0, mark_sequence=7).tripped is True


def test_flat_reading_needs_no_confirmation():
    """After a stop-out to flat, equity is realized balance: nothing to second-guess."""
    guard = _guard(peak=100.0, equity_peak=110.0)
    guard.observe(100.0, equity=100.0, mark_sequence=1)

    assessment = guard.observe(87.0, equity=87.0, mark_sequence=2, flat=True)

    assert assessment.tripped is True  # 20.9% off the $110 equity peak


def test_balance_leg_stays_immediate_without_confirmation():
    guard = _guard(peak=100.0)
    guard.observe(100.0, equity=100.0, mark_sequence=1)

    assert guard.observe(79.0, equity=79.0, mark_sequence=2).tripped is True  # realized loss


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


def test_reset_peak_cannot_clear_a_halt_while_the_position_is_still_that_deep(monkeypatch):
    """The override re-baselines to the current BALANCE; an open position still
    >= cap underwater re-trips the equity leg on the first mark (documented: the
    override works once the position is closed or recovers)."""
    monkeypatch.setenv("FEATURE_MAX_DRAWDOWN_RESET_PEAK", "true")
    guard = MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN)
    guard.seed_peak(100.0, equity_peak=110.0)

    assert guard.observe(100.0, equity=79.0, mark_sequence=1).tripped is True


# ---------------------------------------------------------------------------
# Enforcer: wiring, seeding, restart, stale marks
# ---------------------------------------------------------------------------


class _Marks:
    """Controllable stand-in for LivePositionTracker.mark_reading."""

    def __init__(self, unrealized: float = 0.0, fresh: bool = True, flat: bool = False) -> None:
        self.unrealized = unrealized
        self.fresh = fresh
        self.flat = flat
        self.sequence = 0
        self.raises: Exception | None = None

    def __call__(self) -> MarkReading:
        if self.raises is not None:
            raise self.raises
        return MarkReading(self.unrealized, self.fresh, self.sequence, self.flat)

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


# ---------------------------------------------------------------------------
# Durable equity peak, data-gap enforcement
# ---------------------------------------------------------------------------


def _enforcer_with_writer(state, marks, writer) -> MaxDrawdownEnforcer:
    return MaxDrawdownEnforcer(
        engine_state=state,
        guard=MaxDrawdownGuard(DEFAULT_MAX_DRAWDOWN),
        mark_provider=marks,
        snapshot_writer=writer,
    )


def test_confirmed_new_equity_peak_is_persisted_so_a_restart_can_reseed_it():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=0.0)
    writer = MagicMock()
    enforcer = _enforcer_with_writer(state, marks, writer)
    enforcer.check()  # arms; baseline $100 is already durable
    writer.assert_not_called()

    marks.tick(unrealized=6.0)
    enforcer.check()
    writer.assert_not_called()  # one mark is not a confirmed peak
    marks.tick(unrealized=6.0)  # second distinct mark confirms $106
    enforcer.check()

    assert enforcer.guard.equity_peak == pytest.approx(106.0)
    writer.assert_called_once()


def test_small_peak_increase_does_not_write_a_snapshot():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=0.0)
    writer = MagicMock()
    enforcer = _enforcer_with_writer(state, marks, writer)
    enforcer.check()

    marks.tick(unrealized=0.1)
    enforcer.check()
    marks.tick(unrealized=0.1)
    enforcer.check()

    assert enforcer.guard.equity_peak == pytest.approx(100.1)
    writer.assert_not_called()


def test_snapshot_writer_failure_never_breaks_the_check():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=0.0)
    writer = MagicMock(side_effect=RuntimeError("db down"))
    enforcer = _enforcer_with_writer(state, marks, writer)
    enforcer.check()
    marks.tick(unrealized=8.0)
    enforcer.check()
    marks.tick(unrealized=8.0)

    enforcer.check()

    assert writer.call_count >= 1
    state._enter_close_only_mode.assert_not_called()


def test_data_gap_iterations_still_run_the_drawdown_check():
    """A market-data gap `continue`s before the regular check; the cap (and its
    blind-mark escalation) must keep running through it."""
    from tests.unit.engines.live.test_system_halt_gate import _make_engine

    for df_result, fresh in ((None, True), (MagicMock(empty=False), False)):
        engine = _make_engine()
        engine._drawdown_enforcer = MagicMock()
        engine._ensure_ws_health_monitor_alive = MagicMock()
        engine._drain_pending_fill_exits = MagicMock()
        engine._get_latest_data = MagicMock(return_value=df_result)
        engine._is_data_fresh = MagicMock(return_value=fresh)
        engine._calculate_adaptive_interval = MagicMock(return_value=0)
        engine._sleep_with_interrupt = MagicMock()
        engine.stop = MagicMock()
        engine.is_running = True

        engine._trading_loop("BTCUSDT", "1h", max_steps=1)

        engine._drawdown_enforcer.check_during_data_gap.assert_called_once()


def test_data_gap_check_runs_the_balance_leg_only():
    state = _state(balance=78.0, db_peak=100.0)
    marks = MagicMock()
    enforcer = _enforcer(state, marks)

    enforcer.check_during_data_gap()

    marks.assert_not_called()  # marks are not refreshed on gap iterations
    state._enter_close_only_mode.assert_called_once()


def test_data_gap_check_does_not_consume_the_seed_budget():
    state = _state(balance=100.0)
    state.db_manager.get_session_peak_balance.side_effect = RuntimeError("db down")
    enforcer = _enforcer(state, _Marks())

    for _ in range(drawdown_guard.MAX_SEED_ATTEMPTS + 5):
        enforcer.check_during_data_gap()

    assert enforcer.guard.seeded is False  # never self-anchored on the boot balance


def test_flat_account_equity_breach_trips_without_a_second_mark():
    state = _state(balance=87.0, db_peak=100.0, db_equity_peak=110.0)
    enforcer = _enforcer(state, _Marks(unrealized=0.0, flat=True))

    enforcer.check()

    state._enter_close_only_mode.assert_called_once()


def test_peak_snapshot_is_skipped_when_current_equity_is_not_at_the_confirmed_peak():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=0.0)
    writer = MagicMock()
    enforcer = _enforcer_with_writer(state, marks, writer)
    enforcer.check()
    marks.tick(unrealized=6.0)
    enforcer.check()

    marks.tick(unrealized=6.0)  # confirms $106...
    enforcer.guard.observe(100.0, 106.0, marks.sequence)
    enforcer._last_equity = 500.0  # ...but the live mark on the persisting check is a bad tick
    enforcer._persist_equity_peak()

    writer.assert_not_called()


def test_peak_snapshot_is_not_written_from_the_pre_order_gate():
    state = _state(balance=100.0)
    marks = _Marks(unrealized=0.0)
    writer = MagicMock()
    enforcer = _enforcer_with_writer(state, marks, writer)
    enforcer.check()

    marks.tick(unrealized=6.0)
    enforcer.check_before_new_risk()
    marks.tick(unrealized=6.0)
    enforcer.check_before_new_risk()

    assert enforcer.guard.equity_peak == pytest.approx(106.0)
    writer.assert_not_called()
