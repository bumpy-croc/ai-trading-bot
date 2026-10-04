"""Crash-window regression tests for the reconciler's offline-exit closes (#1223).

``_close_position_from_filled_sl`` and ``_reconcile_filled_exit`` used to close the
position row in a separate upfront transaction before the atomic balance+trade write.
A crash between the two left the position CLOSED with no Trade and no balance change —
a PnL that was dropped and never recovered, because a CLOSED position is never
re-picked-up. The close now commits inside the same transaction as the Trade insert and
the balance delta, so these tests run against a real in-memory database and fail the
commit to prove the all-or-nothing behavior.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.database.manager import DatabaseManager
from src.database.models import Position, PositionStatus, Trade, TradeSource
from src.engines.live.execution.position_tracker import LivePosition
from src.engines.live.reconciliation import PositionReconciler
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast

SEED_BALANCE = 1000.0
ENTRY_PRICE = 100.0
QUANTITY = 2.0
SIZE = 0.2
SL_PRICE = 90.0


class _Env(SimpleNamespace):
    """Bundle of the real DB, the reconciler, the tracker and the tracked position."""


@pytest.fixture
def env() -> _Env:
    db = DatabaseManager("sqlite:///:memory:")
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.LIVE,
        initial_balance=SEED_BALANCE,
    )
    db.update_balance(SEED_BALANCE, "seed", "test", session_id)
    db_position_id = db.log_position(
        symbol="BTCUSDT",
        side="long",
        entry_price=ENTRY_PRICE,
        size=SIZE,
        strategy_name="TestStrategy",
        entry_order_id="entry-1",
        quantity=QUANTITY,
        entry_balance=SEED_BALANCE,
        session_id=session_id,
    )
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=SIZE,
        entry_price=ENTRY_PRICE,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        order_id="entry-1",
        original_size=SIZE,
        current_size=SIZE,
        quantity=QUANTITY,
    )
    position.db_position_id = db_position_id
    positions = {position.order_id: position}
    tracker = SimpleNamespace(
        _positions=positions,
        _positions_lock=threading.Lock(),
        remove_position=Mock(side_effect=lambda oid: positions.pop(oid, None)),
    )
    reconciler = PositionReconciler(
        exchange_interface=Mock(),
        position_tracker=tracker,
        db_manager=db,
        session_id=session_id,
        fee_rate=0.0,
    )
    return _Env(
        db=db,
        session_id=session_id,
        position=position,
        position_id=db_position_id,
        tracker=tracker,
        reconciler=reconciler,
    )


def _sl_order(order_id: str = "sl-1") -> SimpleNamespace:
    return SimpleNamespace(
        average_price=SL_PRICE, order_id=order_id, commission=0.0, commission_asset="USDT"
    )


def _position_status(db: DatabaseManager, position_id: int) -> PositionStatus:
    with db.get_session() as session:
        row = session.query(Position).filter(Position.id == position_id).one()
        return row.status


def _trade_count(db: DatabaseManager, session_id: int) -> int:
    with db.get_session() as session:
        return session.query(Trade).filter(Trade.session_id == session_id).count()


@pytest.fixture(params=["commit_fails", "trade_write_fails"])
def fail_next_commit(request, monkeypatch, env):
    """Arm a one-shot failure of the close's DB write.

    ``commit_fails``: the next ``Session.commit`` raises (connection lost mid-transaction).
    ``trade_write_fails``: the next ``log_trade`` call raises. In the old two-step design the
    position row had already been closed by then, so this is the dropped-PnL crash window.
    """
    armed = {"on": False}

    if request.param == "commit_fails":
        real_commit = Session.commit

        def _commit(self):
            if armed["on"]:
                armed["on"] = False
                self.rollback()
                raise OperationalError("COMMIT", {}, Exception("connection lost"))
            return real_commit(self)

        monkeypatch.setattr(Session, "commit", _commit)
    else:
        real_log_trade = env.db.log_trade

        def _log_trade(*args, **kwargs):
            if armed["on"]:
                armed["on"] = False
                raise RuntimeError("process died before the trade write")
            return real_log_trade(*args, **kwargs)

        monkeypatch.setattr(env.db, "log_trade", _log_trade)

    def arm() -> None:
        armed["on"] = True

    return arm


def test_offline_sl_close_commits_position_trade_and_balance_together(env):
    env.reconciler._close_position_from_filled_sl(env.position, _sl_order())

    assert _position_status(env.db, env.position_id) == PositionStatus.CLOSED
    assert _trade_count(env.db, env.session_id) == 1
    assert env.db.get_current_balance(env.session_id) == pytest.approx(
        SEED_BALANCE + (SL_PRICE - ENTRY_PRICE) * QUANTITY
    )
    assert env.position.order_id not in env.tracker._positions


def test_offline_sl_crash_mid_close_leaves_position_open_and_retry_realizes_pnl_once(
    env, fail_next_commit
):
    """The crash lands between 'flip the position' and 'record the PnL' in the old design.
    Now nothing commits, the position stays OPEN and tracked, and the retry books the PnL
    exactly once."""
    fail_next_commit()
    env.reconciler._close_position_from_filled_sl(env.position, _sl_order())

    assert _position_status(env.db, env.position_id) == PositionStatus.OPEN
    assert _trade_count(env.db, env.session_id) == 0
    assert env.db.get_current_balance(env.session_id) == pytest.approx(SEED_BALANCE)
    assert env.position.order_id in env.tracker._positions
    assert env.position.exchange_close_pending is True

    env.reconciler._close_position_from_filled_sl(env.position, _sl_order())

    expected_pnl = (SL_PRICE - ENTRY_PRICE) * QUANTITY
    assert _position_status(env.db, env.position_id) == PositionStatus.CLOSED
    assert _trade_count(env.db, env.session_id) == 1
    assert env.db.get_current_balance(env.session_id) == pytest.approx(SEED_BALANCE + expected_pnl)


def test_filled_exit_crash_mid_close_raises_leaves_position_open_and_retry_realizes_once(
    env, fail_next_commit
):
    order_data = {"position_id": env.position_id, "client_order_id": "exit-1"}

    fail_next_commit()
    with pytest.raises(RuntimeError, match="could not be persisted"):
        env.reconciler._reconcile_filled_exit(
            order_data, fill_price=SL_PRICE, exit_fee=0.0, exit_order_id="exch-exit-1"
        )

    assert _position_status(env.db, env.position_id) == PositionStatus.OPEN
    assert _trade_count(env.db, env.session_id) == 0
    assert env.db.get_current_balance(env.session_id) == pytest.approx(SEED_BALANCE)
    assert env.position.order_id in env.tracker._positions

    env.reconciler._reconcile_filled_exit(
        order_data, fill_price=SL_PRICE, exit_fee=0.0, exit_order_id="exch-exit-1"
    )

    assert _position_status(env.db, env.position_id) == PositionStatus.CLOSED
    assert _trade_count(env.db, env.session_id) == 1
    assert env.db.get_current_balance(env.session_id) == pytest.approx(
        SEED_BALANCE + (SL_PRICE - ENTRY_PRICE) * QUANTITY
    )
    assert env.position.order_id not in env.tracker._positions


def test_dedup_collision_closes_position_without_reapplying_pnl(env):
    """A trade with the same exit order id already exists (an earlier pass) but the position
    row is still OPEN: the collision must still close the row, and must not re-book the PnL."""
    env.db.log_trade(
        symbol="BTCUSDT",
        side="long",
        entry_price=ENTRY_PRICE,
        exit_price=SL_PRICE,
        size=SIZE,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        exit_time=datetime.now(UTC),
        pnl=-20.0,
        exit_reason="stop_loss",
        strategy_name="TestStrategy",
        source=TradeSource.LIVE,
        session_id=env.session_id,
        exit_order_id="sl-dup",
    )

    env.reconciler._close_position_from_filled_sl(env.position, _sl_order("sl-dup"))

    assert _position_status(env.db, env.position_id) == PositionStatus.CLOSED
    assert _trade_count(env.db, env.session_id) == 1
    assert env.db.get_current_balance(env.session_id) == pytest.approx(SEED_BALANCE)
    assert env.position.order_id not in env.tracker._positions


def test_unpriced_exit_still_closes_the_row(env):
    """No usable exit price means no PnL can be realized, but the exchange position is gone,
    so the DB row must still be closed."""
    closed = env.reconciler._realize_pnl_on_close(
        env.position, None, "stop_loss_filled_offline", log_trade=True, exit_order_id="x"
    )

    assert closed is True
    assert _position_status(env.db, env.position_id) == PositionStatus.CLOSED
    assert _trade_count(env.db, env.session_id) == 0
    assert env.db.get_current_balance(env.session_id) == pytest.approx(SEED_BALANCE)


def test_realize_pnl_on_close_reports_failure_when_write_does_not_commit(env, fail_next_commit):
    fail_next_commit()

    closed = env.reconciler._realize_pnl_on_close(
        env.position,
        SL_PRICE,
        "stop_loss_filled_offline",
        log_trade=True,
        exit_order_id="sl-fail",
    )

    assert closed is False
    assert _position_status(env.db, env.position_id) == PositionStatus.OPEN


def test_ledger_rejection_does_not_block_the_offline_close(env):
    """A realized loss larger than the ledger balance is refused by the ledger, but the stop
    already filled on the exchange: the position must close and the anomaly is audited."""
    env.db.update_balance(5.0, "shrunk", "test", env.session_id)

    env.reconciler._close_position_from_filled_sl(env.position, _sl_order())

    assert _position_status(env.db, env.position_id) == PositionStatus.CLOSED
    assert _trade_count(env.db, env.session_id) == 1
    assert env.db.get_current_balance(env.session_id) == pytest.approx(5.0)
    assert len(env.db.get_audit_events(env.session_id, severity="CRITICAL")) == 1
    assert env.position.order_id not in env.tracker._positions


def test_filled_exit_for_untracked_missing_row_does_not_wedge_the_journal(env):
    """An exit whose position is neither tracked nor in the DB has nothing to retry against, so
    it must not raise (which would leave the order journal SUBMITTED forever)."""
    env.reconciler._reconcile_filled_exit(
        {"position_id": 99999, "client_order_id": "exit-orphan"},
        fill_price=SL_PRICE,
        exit_fee=0.0,
        exit_order_id="exch-exit-orphan",
    )

    assert _trade_count(env.db, env.session_id) == 0
    assert env.position.order_id in env.tracker._positions
