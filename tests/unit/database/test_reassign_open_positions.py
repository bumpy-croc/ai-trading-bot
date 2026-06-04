"""Tests for ``DatabaseManager.reassign_open_positions_to_session`` (#668).

Regression coverage for the capital-critical clean-restart bug where a graceful
restart created a NEW trading session but left OPEN positions bound to the OLD
(now-inactive) session. ``get_active_positions(new_session_id)`` filters by
session, so the OPEN position was orphaned (still OPEN in the DB, invisible to
the live tracker). This reassigns genuinely-OPEN positions — and their linked
orders — onto the new session so recovery can carry them forward.

Uses a real in-memory SQLite database (the same lightweight pattern as
``test_position_close_atomic.py``) so the transaction semantics and the
``uq_order_internal_session`` collision handling are exercised end to end.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.database.manager import DatabaseManager
from src.database.models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    ReconciliationAuditEvent,
    TradeSource,
)

pytestmark = pytest.mark.unit


def _make_db() -> DatabaseManager:
    """Fresh isolated in-memory DB with schema created."""
    return DatabaseManager("sqlite:///:memory:")


def _new_session(
    db: DatabaseManager, symbol: str = "BTCUSDT", strategy: str = "TestStrategy"
) -> int:
    return db.create_trading_session(
        strategy_name=strategy,
        symbol=symbol,
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )


def _open_position(
    db: DatabaseManager,
    session_id: int,
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "TestStrategy",
    entry_order_id: str = "order-1",
    stop_loss_order_id: str | None = None,
) -> int:
    """Log a fresh OPEN position (with a linked ENTRY order) and return its id."""
    return db.log_position(
        symbol=symbol,
        side="long",
        entry_price=100.0,
        size=0.1,
        strategy_name=strategy,
        entry_order_id=entry_order_id,
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
        stop_loss_order_id=stop_loss_order_id,
    )


def _close_position(db: DatabaseManager, position_id: int) -> None:
    with db.get_session() as session:
        pos = session.query(Position).filter(Position.id == position_id).first()
        pos.status = PositionStatus.CLOSED
        session.commit()


def _position_session(db: DatabaseManager, position_id: int) -> int:
    with db.get_session() as session:
        return session.query(Position).filter(Position.id == position_id).first().session_id


def _order_session_for_position(db: DatabaseManager, position_id: int) -> int:
    with db.get_session() as session:
        return (
            session.query(Order)
            .filter(Order.position_id == position_id)
            .order_by(Order.id.asc())
            .first()
            .session_id
        )


class TestReassignOpenPositions:
    """Only genuinely-OPEN, matching positions move; closed/other rows stay put."""

    def test_moves_only_open_position_for_matching_session(self):
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        position_id = _open_position(db, old, entry_order_id="exch-1")

        moved = db.reassign_open_positions_to_session(old, new)

        assert moved == [position_id]
        assert _position_session(db, position_id) == new
        # And it now surfaces under the new session's active set.
        assert any(p["id"] == position_id for p in db.get_active_positions(new))
        # ...and no longer under the old one.
        assert all(p["id"] != position_id for p in db.get_active_positions(old))

    def test_old_session_none_adopts_all_orphaned_positions(self):
        """old_session_id=None adopts EVERY orphaned OPEN position for the
        symbol/strategy (any session != new) — including one stranded under an
        OLDER session than the most-recent (an intervening flat restart left it
        two sessions back: the live-prod-orphan case, #668)."""
        db = _make_db()
        oldest = _new_session(db)  # holds the real orphan
        intervening = _new_session(db)  # a later, flat session
        new = _new_session(db)  # the current active session

        orphan_oldest = _open_position(db, oldest, entry_order_id="exch-oldest")
        orphan_mid = _open_position(db, intervening, entry_order_id="exch-mid")
        closed = _open_position(db, oldest, entry_order_id="exch-closed")
        _close_position(db, closed)
        already_new = _open_position(db, new, entry_order_id="exch-new")
        other_symbol = _open_position(db, oldest, symbol="ETHUSDT", entry_order_id="exch-eth")

        moved = db.reassign_open_positions_to_session(
            old_session_id=None,
            new_session_id=new,
            symbol="BTCUSDT",
            strategy_name="TestStrategy",
        )

        assert set(moved) == {orphan_oldest, orphan_mid}  # both orphans, across two sessions
        assert _position_session(db, orphan_oldest) == new
        assert _position_session(db, orphan_mid) == new
        assert _position_session(db, closed) == oldest  # CLOSED not resurrected
        assert _position_session(db, already_new) == new  # already-current untouched
        assert _position_session(db, other_symbol) == oldest  # co-tenant symbol not pulled in

    def test_old_session_none_adopts_null_session_orphan(self):
        """old=None must also adopt an OPEN position whose session_id is NULL
        (the column is nullable). `session_id != new` ALONE drops NULL rows
        (NULL <> x → UNKNOWN → excluded), so the explicit `IS NULL` in the
        filter is load-bearing — this guards against a future "simplification"
        that would silently re-orphan a NULL-session position (#668)."""
        db = _make_db()
        new = _new_session(db)
        orphan = _open_position(db, new, entry_order_id="exch-null")
        # Strand it: NULL session_id (no owning session).
        with db.get_session() as session:
            session.query(Position).filter(Position.id == orphan).first().session_id = None
            session.commit()
        assert _position_session(db, orphan) is None

        moved = db.reassign_open_positions_to_session(
            old_session_id=None,
            new_session_id=new,
            symbol="BTCUSDT",
            strategy_name="TestStrategy",
        )

        assert moved == [orphan]
        assert _position_session(db, orphan) == new

    def test_closed_position_is_not_carried_forward(self):
        """A position CLOSED under the old session (e.g. by #671's atomic close)
        must NOT be resurrected onto the new session."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        open_id = _open_position(db, old, entry_order_id="exch-open")
        closed_id = _open_position(db, old, entry_order_id="exch-closed")
        _close_position(db, closed_id)

        moved = db.reassign_open_positions_to_session(old, new)

        assert moved == [open_id]
        assert _position_session(db, closed_id) == old  # stayed behind
        assert _position_session(db, open_id) == new

    def test_other_symbol_position_is_left_behind(self):
        """With a symbol filter, a different symbol's OPEN position must not move."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        btc_id = _open_position(db, old, symbol="BTCUSDT", entry_order_id="exch-btc")
        eth_id = _open_position(db, old, symbol="ETHUSDT", entry_order_id="exch-eth")

        moved = db.reassign_open_positions_to_session(old, new, symbol="BTCUSDT")

        assert moved == [btc_id]
        assert _position_session(db, eth_id) == old

    def test_other_strategy_position_is_left_behind(self):
        """With a strategy filter, a different strategy's OPEN position must not move."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        a_id = _open_position(db, old, strategy="StratA", entry_order_id="exch-a")
        b_id = _open_position(db, old, strategy="StratB", entry_order_id="exch-b")

        moved = db.reassign_open_positions_to_session(old, new, strategy_name="StratA")

        assert moved == [a_id]
        assert _position_session(db, b_id) == old

    def test_linked_order_is_repointed_to_new_session(self):
        """The position's linked ENTRY order moves to the new session too."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        position_id = _open_position(db, old, entry_order_id="exch-1")

        # Pre-condition: the ENTRY order lives under the old session.
        assert _order_session_for_position(db, position_id) == old

        db.reassign_open_positions_to_session(old, new)

        assert _order_session_for_position(db, position_id) == new

    def test_audit_event_recorded_under_new_session(self):
        """Each move records an immutable audit row under the new session."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        position_id = _open_position(db, old, entry_order_id="exch-1")

        db.reassign_open_positions_to_session(old, new)

        with db.get_session() as session:
            event = (
                session.query(ReconciliationAuditEvent)
                .filter(
                    ReconciliationAuditEvent.entity_id == position_id,
                    ReconciliationAuditEvent.field == "session_id",
                )
                .first()
            )
        assert event is not None
        assert event.entity_type == "position"
        assert event.session_id == new
        assert event.old_value == str(old)
        assert event.new_value == str(new)
        assert event.severity == "HIGH"

    def test_idempotent_second_call_moves_nothing(self):
        """A second call finds nothing under the old session — no double-adopt."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        position_id = _open_position(db, old, entry_order_id="exch-1")

        first = db.reassign_open_positions_to_session(old, new)
        second = db.reassign_open_positions_to_session(old, new)

        assert first == [position_id]
        assert second == []

    def test_same_session_is_noop(self):
        """Passing old == new is a guarded no-op (cannot carry onto itself)."""
        db = _make_db()
        session_id = _new_session(db)
        _open_position(db, session_id, entry_order_id="exch-1")

        assert db.reassign_open_positions_to_session(session_id, session_id) == []

    def test_no_matching_positions_returns_empty(self):
        """Empty old session ⇒ empty result, no error."""
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)

        assert db.reassign_open_positions_to_session(old, new) == []


class TestReassignUniqueConstraintCollision:
    """``uq_order_internal_session`` (internal_order_id + session_id) is respected:
    a colliding order is left under the old session; the position still moves and
    stays linked via ``Order.position_id``."""

    def test_colliding_order_left_behind_position_still_moves(self):
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        position_id = _open_position(db, old, entry_order_id=None)

        # Give the position's linked order a known internal_order_id.
        with db.get_session() as session:
            order = session.query(Order).filter(Order.position_id == position_id).first()
            order.internal_order_id = "shared-iid"
            session.commit()
            old_order_id = order.id

        # Pre-existing order under the NEW session with the SAME internal_order_id.
        with db.get_session() as session:
            clash = Order(
                position_id=None,
                order_type=OrderType.ENTRY,
                status=OrderStatus.FILLED,
                internal_order_id="shared-iid",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=Decimal("1.0"),
                strategy_name="TestStrategy",
                session_id=new,
            )
            session.add(clash)
            session.commit()

        moved = db.reassign_open_positions_to_session(old, new)

        # Position moves; the colliding order stays under the old session.
        assert moved == [position_id]
        assert _position_session(db, position_id) == new
        with db.get_session() as session:
            order = session.query(Order).filter(Order.id == old_order_id).first()
            assert order.session_id == old  # not re-pointed (would violate uq)
            assert order.position_id == position_id  # linkage intact

        # The position still surfaces under the new session (join is on position_id).
        assert any(p["id"] == position_id for p in db.get_active_positions(new))


class TestReassignPreservesPartialExitState:
    """Sizing fields are untouched by the move — a partially-exited position keeps
    its current_size/original_size so risk accounting stays correct."""

    def test_partial_exit_sizes_preserved_across_move(self):
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        full_id = _open_position(db, old, entry_order_id="exch-full")
        partial_id = _open_position(db, old, entry_order_id="exch-partial")

        # Simulate a partial exit on the second position: half remains.
        with db.get_session() as session:
            partial = session.query(Position).filter(Position.id == partial_id).first()
            partial.original_size = Decimal("0.1")
            partial.current_size = Decimal("0.05")
            partial.partial_exits_taken = 1
            session.commit()

        moved = db.reassign_open_positions_to_session(old, new)

        # Both OPEN positions move.
        assert sorted(moved) == sorted([full_id, partial_id])
        assert _position_session(db, full_id) == new
        assert _position_session(db, partial_id) == new

        # The partially-exited position keeps its sizing fields unchanged.
        with db.get_session() as session:
            partial = session.query(Position).filter(Position.id == partial_id).first()
            assert float(partial.original_size) == pytest.approx(0.1)
            assert float(partial.current_size) == pytest.approx(0.05)
            assert partial.partial_exits_taken == 1


class TestReassignAtomicCloseInterplay:
    """End-to-end interplay with #671's atomic close: a position closed via
    log_trade(position_id=...) under the old session is NOT carried forward."""

    def test_position_closed_via_log_trade_not_carried_forward(self):
        db = _make_db()
        old = _new_session(db)
        new = _new_session(db)
        kept_open_id = _open_position(db, old, entry_order_id="exch-keep")
        closed_id = _open_position(db, old, entry_order_id="exch-close")

        # Close one position atomically via log_trade (#671): status flips to CLOSED
        # in the same transaction as the Trade insert.
        db.log_trade(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            exit_price=110.0,
            size=0.1,
            entry_time=datetime.now(UTC) - timedelta(hours=1),
            exit_time=datetime.now(UTC),
            pnl=10.0,
            exit_reason="take_profit",
            strategy_name="TestStrategy",
            source=TradeSource.PAPER,
            session_id=old,
            position_id=closed_id,
        )

        moved = db.reassign_open_positions_to_session(old, new)

        # Only the still-open position carries forward; the closed one stays put.
        assert moved == [kept_open_id]
        assert _position_session(db, closed_id) == old
        assert all(p["id"] != closed_id for p in db.get_active_positions(new))
