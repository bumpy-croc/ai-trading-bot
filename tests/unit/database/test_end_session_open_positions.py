"""An ended trading session must not leave OPEN position rows behind silently (#1044).

Uses a real in-memory SQLite database so the end-of-session transaction, the audit
rows and the emitted events are checked end to end.
"""

import pytest

from src.database.manager import DatabaseManager
from src.database.models import (
    AccountBalance,
    EventType,
    Position,
    PositionStatus,
    ReconciliationAuditEvent,
    SystemEvent,
    Trade,
    TradeSource,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _new_session(db: DatabaseManager, mode: TradeSource = TradeSource.PAPER) -> int:
    return db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=mode,
        initial_balance=1000.0,
    )


def _open_position(db: DatabaseManager, session_id: int, order_id: str = "order-1") -> int:
    return db.log_position(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        size=0.1,
        strategy_name="TestStrategy",
        entry_order_id=order_id,
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
    )


def _status(db: DatabaseManager, position_id: int) -> PositionStatus:
    with db.get_session() as session:
        position = session.query(Position).filter(Position.id == position_id).one()
        return position.status


def _events(db: DatabaseManager, error_code: str | None = None) -> list[SystemEvent]:
    with db.get_session() as session:
        query = session.query(SystemEvent)
        if error_code is not None:
            query = query.filter(SystemEvent.error_code == error_code)
        rows = query.all()
        session.expunge_all()
        return rows


def _audit_rows(db: DatabaseManager) -> list[ReconciliationAuditEvent]:
    with db.get_session() as session:
        rows = session.query(ReconciliationAuditEvent).all()
        session.expunge_all()
        return rows


def _count(db: DatabaseManager, model: type) -> int:
    with db.get_session() as session:
        return session.query(model).count()


class TestEndSessionHandlesOpenPositions:
    def test_paper_session_without_carry_forward_closes_positions_balance_neutrally(self, db):
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)
        balances_before = _count(db, AccountBalance)

        db.end_trading_session(session_id, final_balance=1000.0)

        assert _status(db, position_id) == PositionStatus.CLOSED
        assert _count(db, AccountBalance) == balances_before
        assert _count(db, Trade) == 0
        audit = _audit_rows(db)
        assert len(audit) == 1
        assert audit[0].entity_id == position_id
        assert audit[0].session_id == session_id
        assert audit[0].old_value == "OPEN"
        assert audit[0].new_value == "CLOSED"
        assert "#1044" in audit[0].reason

    def test_paper_close_emits_warning_event_listing_positions(self, db):
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        db.end_trading_session(session_id, final_balance=1000.0)

        events = _events(db, "SESSION_ENDED_OPEN_POSITIONS_CLOSED")
        assert len(events) == 1
        assert events[0].event_type == EventType.WARNING
        assert events[0].severity == "warning"
        assert events[0].session_id == session_id
        assert [p["position_id"] for p in events[0].details["positions"]] == [position_id]

    def test_paper_session_with_carry_forward_keeps_positions_open(self, db):
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        db.end_trading_session(session_id, final_balance=1000.0, carry_forward_open_positions=True)

        assert _status(db, position_id) == PositionStatus.OPEN
        assert _audit_rows(db) == []
        assert _events(db, "SESSION_ENDED_OPEN_POSITIONS_CLOSED") == []

    def test_carry_forward_records_preserved_rows_on_engine_stop_event(self, db):
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        db.end_trading_session(session_id, final_balance=1000.0, carry_forward_open_positions=True)

        stop_events = [e for e in _events(db) if e.event_type == EventType.ENGINE_STOP]
        assert len(stop_events) == 1
        open_positions = stop_events[0].details["open_positions"]
        assert open_positions["disposition"] == "preserved"
        assert [p["position_id"] for p in open_positions["positions"]] == [position_id]

    def test_preserved_positions_are_still_carried_forward_on_restart(self, db):
        old_session = _new_session(db)
        position_id = _open_position(db, old_session)
        db.end_trading_session(old_session, final_balance=1000.0, carry_forward_open_positions=True)
        new_session = _new_session(db)

        moved = db.reassign_open_positions_to_session(old_session, new_session)

        assert moved == [position_id]
        assert db.get_open_positions_in_ended_sessions() == []

    def test_live_session_never_closes_rows_and_raises_critical_alert(self, db):
        session_id = _new_session(db, TradeSource.LIVE)
        position_id = _open_position(db, session_id)

        db.end_trading_session(session_id, final_balance=1000.0)

        assert _status(db, position_id) == PositionStatus.OPEN
        assert _audit_rows(db) == []
        events = _events(db, "LIVE_SESSION_ENDED_WITH_OPEN_POSITIONS")
        assert len(events) == 1
        assert events[0].event_type == EventType.ALERT
        assert events[0].severity == "critical"
        assert [p["position_id"] for p in events[0].details["positions"]] == [position_id]

    def test_live_session_alerts_even_when_caller_declares_carry_forward(self, db):
        session_id = _new_session(db, TradeSource.LIVE)
        _open_position(db, session_id)

        db.end_trading_session(session_id, final_balance=1000.0, carry_forward_open_positions=True)

        assert len(_events(db, "LIVE_SESSION_ENDED_WITH_OPEN_POSITIONS")) == 1

    def test_session_without_open_positions_emits_no_position_events(self, db):
        session_id = _new_session(db)

        db.end_trading_session(session_id, final_balance=1000.0)

        assert _events(db, "SESSION_ENDED_OPEN_POSITIONS_CLOSED") == []
        stop_events = [e for e in _events(db) if e.event_type == EventType.ENGINE_STOP]
        assert "open_positions" not in stop_events[0].details

    def test_already_closed_positions_and_other_sessions_are_untouched(self, db):
        session_id = _new_session(db)
        other_session = _new_session(db)
        closed_id = _open_position(db, session_id, "closed-1")
        other_open_id = _open_position(db, other_session, "other-1")
        db.close_position(closed_id)

        db.end_trading_session(session_id, final_balance=1000.0)

        assert _audit_rows(db) == []
        assert _status(db, other_open_id) == PositionStatus.OPEN


class TestOpenPositionsInEndedSessionsCheck:
    def test_lists_open_rows_of_ended_sessions_only(self, db):
        ended = _new_session(db)
        active = _new_session(db)
        stale_id = _open_position(db, ended, "stale-1")
        _open_position(db, active, "live-1")
        db.end_trading_session(ended, carry_forward_open_positions=True)

        rows = db.get_open_positions_in_ended_sessions()

        assert [r["position_id"] for r in rows] == [stale_id]
        assert rows[0]["session_id"] == ended
        assert rows[0]["session_mode"] == "PAPER"
        assert rows[0]["symbol"] == "BTCUSDT"

    def test_session_filter_scopes_the_result(self, db):
        first = _new_session(db)
        second = _new_session(db)
        _open_position(db, first, "a")
        second_id = _open_position(db, second, "b")
        db.end_trading_session(first, carry_forward_open_positions=True)
        db.end_trading_session(second, carry_forward_open_positions=True)

        rows = db.get_open_positions_in_ended_sessions(session_id=second)

        assert [r["position_id"] for r in rows] == [second_id]

    def test_empty_when_nothing_is_stale(self, db):
        session_id = _new_session(db)
        _open_position(db, session_id)

        assert db.get_open_positions_in_ended_sessions() == []


class TestCloseOpenPositionsInEndedSessions:
    def test_closes_paper_rows_with_audit_and_no_ledger_writes(self, db):
        ended = _new_session(db)
        position_id = _open_position(db, ended)
        db.end_trading_session(ended, carry_forward_open_positions=True)
        balances_before = _count(db, AccountBalance)

        closed = db.close_open_positions_in_ended_sessions([position_id], reason="cleanup note")

        assert closed == [position_id]
        assert _status(db, position_id) == PositionStatus.CLOSED
        assert _count(db, AccountBalance) == balances_before
        assert _count(db, Trade) == 0
        audit = _audit_rows(db)
        assert [a.reason for a in audit] == ["cleanup note"]

    def test_skips_live_session_rows(self, db):
        ended = _new_session(db, TradeSource.LIVE)
        position_id = _open_position(db, ended)
        db.end_trading_session(ended)

        closed = db.close_open_positions_in_ended_sessions([position_id], reason="x")

        assert closed == []
        assert _status(db, position_id) == PositionStatus.OPEN

    def test_skips_rows_of_active_sessions(self, db):
        active = _new_session(db)
        position_id = _open_position(db, active)

        closed = db.close_open_positions_in_ended_sessions([position_id], reason="x")

        assert closed == []
        assert _status(db, position_id) == PositionStatus.OPEN

    def test_skips_already_closed_rows(self, db):
        ended = _new_session(db)
        position_id = _open_position(db, ended)
        db.end_trading_session(ended, carry_forward_open_positions=True)
        db.close_position(position_id)

        assert db.close_open_positions_in_ended_sessions([position_id], reason="x") == []
        assert _audit_rows(db) == []

    def test_empty_id_list_is_a_noop(self, db):
        assert db.close_open_positions_in_ended_sessions([], reason="x") == []
