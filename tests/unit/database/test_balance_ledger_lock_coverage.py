"""SQLite-safe unit coverage for the account_balances ledger lock (#735/#736, #1224 review).

``tests/unit/database/test_balance_ledger_concurrency.py`` proves the lock is
*effective* under real concurrent Postgres writers, but it is marked
``pytest.mark.slow`` and skips without a reachable Postgres — the unit CI job
runs with no Postgres service and the integration job excludes ``slow`` too, so
that file never actually runs in CI. Deleting ``DatabaseManager._lock_balance_ledger``
entirely would still pass CI today. These tests close that gap cheaply: they
don't prove the lock serializes real concurrent writers (that needs Postgres),
but they do prove nobody can silently drop the lock call from any of the four
ledger-writing paths without a fast, always-run unit test failing.

Also covers the ``atomic_balance_correction`` ``caller_snapshot`` fix (#1224
review finding C): the correction must preserve a concurrent delta writer's
contribution rather than silently reproducing a plain absolute overwrite.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import TradeSource

pytestmark = pytest.mark.unit


def _make_db() -> DatabaseManager:
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


def test_account_balance_has_session_id_index_matching_get_current_balance_ordering():
    """#1224 review finding A: get_current_balance orders by ``id`` DESC (not
    ``last_updated``) to break same-microsecond ties unambiguously (#735). The
    pre-existing ``idx_balance_session_updated`` index is on
    ``(session_id, last_updated)`` and does not serve that query — without a
    matching index it falls back to sorting every row for the session, and this
    query now also runs inside ``DatabaseManager._lock_balance_ledger``'s
    advisory lock on every balance write. Assert the composite index that
    actually matches the query's ordering exists (in addition to, not instead
    of, the old one — other queries still want last_updated ordering)."""
    from src.database.models import AccountBalance

    indexes = {
        idx.name: tuple(col.name for col in idx.columns) for idx in AccountBalance.__table__.indexes
    }
    assert indexes.get("idx_balance_session_id") == ("session_id", "id")
    assert indexes.get("idx_balance_session_updated") == ("session_id", "last_updated")


class TestBalanceLedgerLockCoverage:
    """Every writer of the append-only account_balances ledger must acquire
    ``_lock_balance_ledger`` exactly once before applying its change."""

    def test_atomic_balance_update_acquires_lock_once(self):
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)

        with patch.object(
            DatabaseManager, "_lock_balance_ledger", autospec=True
        ) as mock_lock:
            with db.atomic_balance_update(
                balance_change=-5.0, reason="entry_fee", updated_by="test", session_id=session_id
            ):
                pass

        mock_lock.assert_called_once()

    def test_atomic_position_reconciliation_acquires_lock_once(self):
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)
        position_id = _open_position(db, session_id)

        trade_data = {
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.1,
            "entry_price": 100.0,
            "entry_time": datetime.now(UTC) - timedelta(hours=1),
            "exit_time": datetime.now(UTC),
            "strategy_name": "TestStrategy",
            "source": TradeSource.PAPER,
        }

        with patch.object(
            DatabaseManager, "_lock_balance_ledger", autospec=True
        ) as mock_lock:
            with db.atomic_position_reconciliation(
                position_db_id=position_id,
                realized_pnl=-5.0,
                exit_price=95.0,
                exit_reason="stop_loss_offline",
                trade_data=trade_data,
                session_id=session_id,
            ):
                pass

        mock_lock.assert_called_once()

    def test_atomic_balance_correction_acquires_lock_once(self):
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)

        with patch.object(
            DatabaseManager, "_lock_balance_ledger", autospec=True
        ) as mock_lock:
            with db.atomic_balance_correction(
                950.0,
                "exchange_sync_correction",
                "system",
                session_id,
                caller_snapshot=1000.0,
            ):
                pass

        mock_lock.assert_called_once()

    def test_log_trade_with_balance_delta_acquires_lock_once(self):
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)
        position_id = _open_position(db, session_id)

        with patch.object(
            DatabaseManager, "_lock_balance_ledger", autospec=True
        ) as mock_lock:
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
                balance_delta=-5.0,
            )

        mock_lock.assert_called_once()

    def test_log_trade_without_balance_delta_does_not_acquire_lock(self):
        """Control case: a caller that owns its own balance update (no
        balance_delta) must not touch the ledger lock from inside log_trade."""
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)
        position_id = _open_position(db, session_id)

        with patch.object(
            DatabaseManager, "_lock_balance_ledger", autospec=True
        ) as mock_lock:
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

        mock_lock.assert_not_called()


class TestAtomicBalanceCorrectionCallerSnapshot:
    """#1224 review finding C: atomic_balance_correction must preserve a
    concurrent delta writer's contribution rather than silently reproducing a
    plain absolute overwrite.

    SQLite can't exercise the real concurrent-thread race (that needs Postgres
    advisory locks — see test_balance_ledger_concurrency.py), but the
    caller_snapshot arithmetic itself is pure and single-threaded: this
    simulates "a delta writer commits between the caller's pre-lock read and
    the correction" by simply performing that write, in order, before invoking
    the correction with the now-stale snapshot.
    """

    def test_preserves_concurrent_delta_committed_after_caller_snapshot(self):
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)

        # Caller reads the balance BEFORE the concurrent writer commits.
        caller_snapshot = db.get_current_balance(session_id)
        assert caller_snapshot == pytest.approx(1000.0)

        # A concurrent delta writer (e.g. a trade closing) commits in between.
        with db.atomic_balance_update(
            balance_change=50.0, reason="realized_pnl", updated_by="test", session_id=session_id
        ):
            pass
        assert db.get_current_balance(session_id) == pytest.approx(1050.0)

        # The correction targets 1010.0 based on the STALE pre-delta snapshot,
        # but must still land on target + concurrent_delta = 1060.0, not clobber
        # the concurrent writer's +50 by overwriting to exactly 1010.0.
        with db.atomic_balance_correction(
            1010.0,
            "exchange_sync_correction",
            "system",
            session_id,
            caller_snapshot=caller_snapshot,
        ):
            pass

        assert db.get_current_balance(session_id) == pytest.approx(1060.0)

    def test_reduces_to_plain_overwrite_with_no_concurrent_writer(self):
        """Control case: when caller_snapshot matches the lock-fresh read (the
        common case, no concurrent writer), the correction is arithmetically
        identical to the pre-fix absolute-overwrite behavior."""
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)

        caller_snapshot = db.get_current_balance(session_id)

        with db.atomic_balance_correction(
            975.0,
            "exchange_sync_correction",
            "system",
            session_id,
            caller_snapshot=caller_snapshot,
        ):
            pass

        assert db.get_current_balance(session_id) == pytest.approx(975.0)

    def test_omitted_caller_snapshot_falls_back_to_lock_fresh_read(self):
        """Backward-compatible default: no caller_snapshot means the delta is
        computed against the lock-fresh read, exactly like a plain absolute
        overwrite (no concurrent-delta protection)."""
        db = _make_db()
        session_id = _new_session(db)
        db.update_balance(1000.0, "seed", "test", session_id)

        with db.atomic_balance_correction(
            900.0, "exchange_sync_correction", "system", session_id
        ):
            pass

        assert db.get_current_balance(session_id) == pytest.approx(900.0)
