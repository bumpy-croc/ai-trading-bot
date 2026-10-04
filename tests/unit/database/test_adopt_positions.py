"""DatabaseManager support for exchange-verified adoption (#683)."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Position, PositionStatus, ReconciliationAuditEvent, TradeSource

pytestmark = pytest.mark.fast

STRATEGY = "MlBasic"
SYMBOL = "ETHUSDT"


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _session(db: DatabaseManager, mode: TradeSource = TradeSource.LIVE) -> int:
    return db.create_trading_session(
        strategy_name=STRATEGY,
        symbol=SYMBOL,
        timeframe="1h",
        mode=mode,
        initial_balance=1000.0,
    )


_entry_counter = itertools.count()


def _position(
    db: DatabaseManager, session_id: int, *, strategy: str = STRATEGY, symbol: str = SYMBOL
) -> int:
    return db.log_position(
        symbol=symbol,
        side="long",
        entry_price=2000.0,
        size=0.1,
        strategy_name=strategy,
        entry_order_id=f"entry-{next(_entry_counter)}",
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
    )


def _find(db: DatabaseManager, mode: TradeSource = TradeSource.LIVE) -> list[dict]:
    return db.get_open_positions_across_sessions(symbol=SYMBOL, strategy_name=STRATEGY, mode=mode)


class TestGetOpenPositionsAcrossSessions:
    def test_returns_open_rows_of_every_session_with_their_session_id(self, db):
        first, second = _session(db), _session(db)
        a, b = _position(db, first), _position(db, second)

        rows = {r["id"]: r for r in _find(db)}

        assert set(rows) == {a, b}
        assert rows[a]["session_id"] == first
        assert rows[b]["session_id"] == second

    def test_excludes_closed_rows(self, db):
        session = _session(db)
        closed = _position(db, session)
        db.close_position(closed)

        assert _find(db) == []

    def test_excludes_other_symbol_strategy_and_mode(self, db):
        live, paper = _session(db), _session(db, TradeSource.PAPER)
        wanted = _position(db, live)
        _position(db, live, strategy="Other")
        _position(db, live, symbol="BTCUSDT")
        _position(db, paper)

        assert [r["id"] for r in _find(db)] == [wanted]

    def test_excludes_rows_that_already_have_a_terminal_trade(self, db):
        session = _session(db)
        stale = _position(db, session)
        live = _position(db, session)
        now = datetime.now(UTC)
        db.log_trade(
            symbol=SYMBOL,
            side="long",
            entry_price=2000.0,
            exit_price=2000.0,
            size=0.1,
            entry_time=now,
            exit_time=now,
            pnl=0.0,
            exit_reason="stop_loss",
            strategy_name=STRATEGY,
            source=TradeSource.LIVE,
            session_id=session,
            position_id=stale,
            exit_order_id="exit-1",
        )

        assert [r["id"] for r in _find(db)] == [live]


class TestAdoptPositionsIntoSession:
    def test_moves_rows_and_audits_each_under_the_new_session(self, db):
        old, new = _session(db), _session(db)
        position_id = _position(db, old)

        adopted = db.adopt_positions_into_session([position_id], new)

        assert adopted == [position_id]
        with db.get_session() as s:
            assert s.query(Position).one().session_id == new
            audits = s.query(ReconciliationAuditEvent).filter_by(field="session_id").all()
        assert [(a.session_id, a.old_value, a.new_value) for a in audits] == [
            (new, str(old), str(new))
        ]

    def test_row_already_in_the_session_is_not_moved_or_audited(self, db):
        session = _session(db)
        position_id = _position(db, session)

        adopted = db.adopt_positions_into_session([position_id], session)

        assert adopted == [position_id]
        with db.get_session() as s:
            assert s.query(ReconciliationAuditEvent).count() == 0

    def test_current_size_override_is_written_with_the_move(self, db):
        old, new = _session(db), _session(db)
        position_id = _position(db, old)

        db.adopt_positions_into_session([position_id], new, {position_id: 0.04})

        with db.get_session() as s:
            row = s.query(Position).one()
            assert row.session_id == new
            assert float(row.current_size) == pytest.approx(0.04)
            assert float(row.original_size) == pytest.approx(0.1)
            resize = s.query(ReconciliationAuditEvent).filter_by(field="current_size").one()
        assert float(resize.new_value) == pytest.approx(0.04)

    def test_closed_row_is_left_alone(self, db):
        old, new = _session(db), _session(db)
        position_id = _position(db, old)
        db.close_position(position_id)

        assert db.adopt_positions_into_session([position_id], new) == []
        with db.get_session() as s:
            row = s.query(Position).one()
            assert row.session_id == old
            assert row.status == PositionStatus.CLOSED

    def test_empty_list_is_a_noop(self, db):
        assert db.adopt_positions_into_session([], _session(db)) == []
