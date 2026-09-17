"""Atomic position-close tests for ``DatabaseManager.log_trade`` (#657).

Regression coverage for the capital-critical bug where a normal live/paper exit
wrote the closed Trade row but never flipped ``positions.status`` to CLOSED. The
stale-OPEN row was then reloaded on restart and re-closed, producing phantom
duplicate trades.

These use a real in-memory SQLite database (the same lightweight pattern as
``test_mfe_mae_database.py``) so the transaction semantics — and the atomic
both-or-neither guarantee — are exercised end to end rather than mocked.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from src.database.manager import DatabaseManager
from src.database.models import Position, PositionStatus, Trade, TradeSource

pytestmark = pytest.mark.unit


def _make_db() -> DatabaseManager:
    """Fresh isolated in-memory DB with schema created."""
    return DatabaseManager("sqlite:///:memory:")


def _new_session(db: DatabaseManager) -> int:
    return db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )


def _open_position(db: DatabaseManager, session_id: int, order_id: str = "order-1") -> int:
    """Log a fresh OPEN position and return its DB id."""
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


def _position_status(db: DatabaseManager, position_id: int) -> PositionStatus:
    with db.get_session() as session:
        pos = session.query(Position).filter(Position.id == position_id).first()
        assert pos is not None
        return pos.status


def _count_trades(db: DatabaseManager, session_id: int) -> int:
    with db.get_session() as session:
        return session.query(Trade).filter(Trade.session_id == session_id).count()


class TestAtomicPositionClose:
    """log_trade(position_id=...) flips the position CLOSED in one transaction."""

    def test_exit_flips_status_and_links_position_id(self):
        """A normal (paper) exit closes the position AND links trades.position_id."""
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        assert _position_status(db, position_id) == PositionStatus.OPEN

        trade_id = db.log_trade(
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
            session_id=session_id,
            position_id=position_id,
        )

        # Return type unchanged: still the trade id int.
        assert isinstance(trade_id, int)

        # Position flipped to CLOSED in the same call.
        assert _position_status(db, position_id) == PositionStatus.CLOSED

        # Trade row links back to the position and carries final exit state.
        with db.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            assert trade is not None
            assert trade.position_id == position_id
            pos = session.query(Position).filter(Position.id == position_id).first()
            assert float(pos.current_price) == pytest.approx(110.0)
            assert float(pos.unrealized_pnl) == pytest.approx(10.0)

    def test_closed_position_not_returned_by_get_active_positions(self):
        """After an atomic close, the position drops out of the active set so it
        cannot be reloaded by recovery."""
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        assert any(p["id"] == position_id for p in db.get_active_positions(session_id))

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
            session_id=session_id,
            position_id=position_id,
        )

        active_ids = [p["id"] for p in db.get_active_positions(session_id)]
        assert position_id not in active_ids


class TestLogTradeBackwardCompatibility:
    """log_trade without position_id must behave exactly as before."""

    def test_without_position_id_inserts_trade_and_leaves_status(self):
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        trade_id = db.log_trade(
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
            session_id=session_id,
            # No position_id passed.
        )

        # Trade still inserted, return type unchanged.
        assert isinstance(trade_id, int)
        assert _count_trades(db, session_id) == 1

        # Position status untouched (still OPEN) and trade has no FK link.
        assert _position_status(db, position_id) == PositionStatus.OPEN
        with db.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            assert trade.position_id is None

    def test_unknown_position_id_still_inserts_trade(self):
        """A position_id that no longer exists must not crash or lose the trade —
        the trade is the durable audit record."""
        db = _make_db()
        session_id = _new_session(db)

        trade_id = db.log_trade(
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
            session_id=session_id,
            position_id=999_999,  # does not exist
        )

        assert isinstance(trade_id, int)
        assert _count_trades(db, session_id) == 1


class TestAtomicRollback:
    """If the transaction fails, the Trade insert and the status flip both roll
    back — neither an orphaned trade nor a half-closed position survives."""

    def test_commit_failure_rolls_back_both_trade_and_status(self):
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        # First trade claims a specific exit_order_id within the session.
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
            session_id=session_id,
            exit_order_id="dup-exit-1",
        )
        trades_before = _count_trades(db, session_id)

        # Second trade reuses the SAME exit_order_id + session, violating
        # uq_trade_order_session at commit — AFTER the position flip line has run
        # inside the transaction. The rollback must revert BOTH the trade insert
        # and the in-session status flip.
        with pytest.raises(IntegrityError):  # uq_trade_order_session violation
            db.log_trade(
                symbol="BTCUSDT",
                side="long",
                entry_price=100.0,
                exit_price=120.0,
                size=0.1,
                entry_time=datetime.now(UTC) - timedelta(hours=1),
                exit_time=datetime.now(UTC),
                pnl=20.0,
                exit_reason="take_profit",
                strategy_name="TestStrategy",
                source=TradeSource.PAPER,
                session_id=session_id,
                exit_order_id="dup-exit-1",  # duplicate -> commit fails
                position_id=position_id,
            )

        # Neither side persisted: no extra trade row, position still OPEN.
        assert _count_trades(db, session_id) == trades_before
        assert _position_status(db, position_id) == PositionStatus.OPEN


class TestStartupHeal:
    """heal_positions_with_terminal_trades closes stale-OPEN rows that already
    have a terminal Trade — the paper-safe startup self-heal."""

    def test_heal_closes_stale_open_with_terminal_trade(self):
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        # Simulate the historical bug: a terminal trade exists for the position
        # but its status was never flipped (insert trade WITHOUT position_id, then
        # manually link the FK so the heal can match it — mimicking a legacy row
        # whose status flip was skipped).
        trade_id = db.log_trade(
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
            session_id=session_id,
        )
        with db.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            trade.position_id = position_id
            session.commit()

        # Pre-condition: position is wrongly OPEN.
        assert _position_status(db, position_id) == PositionStatus.OPEN

        healed = db.heal_positions_with_terminal_trades(session_id)
        assert healed == 1
        assert _position_status(db, position_id) == PositionStatus.CLOSED

        # Idempotent: a second run finds nothing left to close.
        assert db.heal_positions_with_terminal_trades(session_id) == 0

    def test_heal_leaves_genuinely_open_positions_untouched(self):
        """A position with NO terminal trade must stay OPEN."""
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        healed = db.heal_positions_with_terminal_trades(session_id)

        assert healed == 0
        assert _position_status(db, position_id) == PositionStatus.OPEN

    def test_recovery_does_not_reload_position_with_terminal_trade(self):
        """End-to-end: a position closed atomically (status CLOSED + linked trade)
        is not returned by get_active_positions, so recovery cannot reload it and
        produce a phantom duplicate trade."""
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)

        db.log_trade(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            exit_price=110.0,
            size=0.1,
            entry_time=datetime.now(UTC) - timedelta(hours=1),
            exit_time=datetime.now(UTC),
            pnl=10.0,
            exit_reason="stop_loss",
            strategy_name="TestStrategy",
            source=TradeSource.PAPER,
            session_id=session_id,
            position_id=position_id,
        )

        # Recovery reads get_active_positions(status==OPEN); the closed position
        # must be absent so no second (phantom) close can occur.
        recovered = db.get_active_positions(session_id)
        assert all(p["id"] != position_id for p in recovered)
        # And only the single, legitimate trade exists.
        assert _count_trades(db, session_id) == 1


def _balance(db: DatabaseManager, session_id: int) -> float:
    return db.get_current_balance(session_id)


class TestLogTradeBalanceDelta:
    """log_trade(balance_delta=...) commits the balance change, the Trade insert,
    and the Position CLOSED flip in ONE transaction (#736, #735).

    A separate balance commit followed by a separate trade/position commit is
    exactly the split that let a crash between them double-apply or drop the
    same PnL on recovery. These tests pin the all-or-nothing contract directly
    against a real (in-memory) SQLAlchemy transaction, not a mock, so a
    regression in the atomicity actually fails the test.
    """

    def test_balance_moves_atomically_with_trade_and_close(self):
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)
        db.update_balance(1000.0, "seed", "test", session_id)

        trade_id = db.log_trade(
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
            session_id=session_id,
            position_id=position_id,
            balance_delta=-5.0,  # e.g. net of fees, independent of gross pnl
        )

        assert isinstance(trade_id, int)
        assert _position_status(db, position_id) == PositionStatus.CLOSED
        assert _balance(db, session_id) == pytest.approx(995.0)

    def test_balance_delta_requires_position_id(self):
        db = _make_db()
        session_id = _new_session(db)

        with pytest.raises(ValueError, match="position_id"):
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
                session_id=session_id,
                balance_delta=-5.0,  # no position_id
            )

    def test_negative_balance_rolls_back_trade_and_position_too(self):
        """The discriminating case: a delta that would take the balance negative
        must abort the WHOLE write — no trade row, no status flip, no balance
        change — not just skip the balance leg while still logging the trade
        (the exact inverse-direction #736 symptom: 'trade recorded + position
        closed + balance never adjusted')."""
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)
        db.update_balance(10.0, "seed", "test", session_id)
        trades_before = _count_trades(db, session_id)

        with pytest.raises(ValueError, match="negative balance"):
            db.log_trade(
                symbol="BTCUSDT",
                side="long",
                entry_price=100.0,
                exit_price=50.0,
                size=0.1,
                entry_time=datetime.now(UTC) - timedelta(hours=1),
                exit_time=datetime.now(UTC),
                pnl=-500.0,
                exit_reason="stop_loss",
                strategy_name="TestStrategy",
                source=TradeSource.PAPER,
                session_id=session_id,
                position_id=position_id,
                balance_delta=-500.0,  # 10.0 - 500.0 < 0
            )

        # Nothing committed: no trade, position still OPEN, balance untouched.
        assert _count_trades(db, session_id) == trades_before
        assert _position_status(db, position_id) == PositionStatus.OPEN
        assert _balance(db, session_id) == pytest.approx(10.0)

    def test_dedup_failure_rolls_back_balance_too(self):
        """Extends TestAtomicRollback's existing dedup-collision coverage: the
        SAME uq_trade_order_session violation must ALSO roll back a balance
        write that was folded into the same transaction via balance_delta —
        not just the trade/status flip."""
        db = _make_db()
        session_id = _new_session(db)
        position_id = _open_position(db, session_id)
        db.update_balance(1000.0, "seed", "test", session_id)

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
            session_id=session_id,
            exit_order_id="dup-exit-bd",
        )
        balance_before = _balance(db, session_id)
        trades_before = _count_trades(db, session_id)

        with pytest.raises(IntegrityError):
            db.log_trade(
                symbol="BTCUSDT",
                side="long",
                entry_price=100.0,
                exit_price=120.0,
                size=0.1,
                entry_time=datetime.now(UTC) - timedelta(hours=1),
                exit_time=datetime.now(UTC),
                pnl=20.0,
                exit_reason="take_profit",
                strategy_name="TestStrategy",
                source=TradeSource.PAPER,
                session_id=session_id,
                exit_order_id="dup-exit-bd",  # duplicate -> commit fails
                position_id=position_id,
                balance_delta=20.0,
            )

        # Balance untouched too — not just the trade/position.
        assert _balance(db, session_id) == pytest.approx(balance_before)
        assert _count_trades(db, session_id) == trades_before
        assert _position_status(db, position_id) == PositionStatus.OPEN
