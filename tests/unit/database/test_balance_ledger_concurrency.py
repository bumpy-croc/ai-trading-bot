"""Real-Postgres concurrent-writer regression test for the account_balances
ledger (#735).

This needs genuine transaction isolation / row-locking semantics that SQLite
does not provide (concurrent writers under READ COMMITTED), so it talks to a
real Postgres instance directly via ``DatabaseManager(database_url=...)``
rather than through the ``DATABASE_URL``-driven session fixture in
``tests/conftest.py`` — that fixture runs a destructive
``Base.metadata.drop_all`` + ``create_all`` against the DATABASE_URL the
moment ANY test in the run is marked ``integration``, which would be
unacceptable here: this repo's dev Postgres (``docker compose up -d
postgres``) is a single shared container that other concurrent agent sessions
may be actively using. This file is deliberately NOT marked
``pytest.mark.integration`` for that reason, connects to its own explicit URL,
only ever CREATEs tables (``checkfirst=True``, never drops), and skips itself
at runtime if Postgres isn't reachable.

Run directly: DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot \
    pytest tests/unit/database/test_balance_ledger_concurrency.py -v
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa

from src.database.manager import DatabaseManager
from src.database.models import Base, TradeSource

_env_url = os.environ.get("DATABASE_URL", "")
_PG_URL = (
    _env_url
    if _env_url.startswith("postgresql")
    else "postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot"
)


def _postgres_available(url: str) -> bool:
    try:
        engine = sa.create_engine(url, pool_pre_ping=True)
        with engine.connect():
            pass
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.database,
    pytest.mark.skipif(
        not _postgres_available(_PG_URL),
        reason="Real Postgres not reachable at DATABASE_URL/local default — this test needs "
        "true MVCC/locking semantics SQLite cannot provide.",
    ),
]


@pytest.fixture(scope="module")
def db() -> DatabaseManager:
    manager = DatabaseManager(database_url=_PG_URL)
    # Non-destructive: creates any missing tables, never drops existing ones —
    # safe to run against the shared dev database other agents may be using.
    Base.metadata.create_all(manager.engine, checkfirst=True)
    return manager


@pytest.fixture
def session_id(db: DatabaseManager) -> int:
    sid = db.create_trading_session(
        strategy_name="test_balance_ledger_concurrency",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )
    db.update_balance(1000.0, "seed", "test", sid)
    return sid


class TestConcurrentAtomicBalanceUpdate:
    """#735a: SELECT...FOR UPDATE on the "latest" row of an append-only table
    does not serialize concurrent writers — two writers can each lock a
    (soon-to-be-stale) row before either commits, and the loser overwrites the
    winner's delta with a stale read. ``_lock_balance_ledger``'s Postgres
    advisory lock closes this by serializing the read+write, not just the row.
    """

    def test_concurrent_writers_lose_no_deltas(self, db: DatabaseManager, session_id: int):
        """N threads apply their own +delta via atomic_balance_update, lined up
        on a Barrier to force real lock contention at (as close to) the same
        instant. The final balance must equal the seed plus every delta —
        losing even one proves the race is open.

        To see this test discriminate: temporarily remove the
        ``self._lock_balance_ledger(session, session_id)`` call from
        ``DatabaseManager.atomic_balance_update`` (src/database/manager.py) and
        re-run — the ORIGINAL ``for_update=True`` row lock alone does not
        serialize this append-only table's writers, so with enough concurrent
        threads this assertion flakes/fails with a balance short of the
        expected total.
        """
        n_threads = 10
        delta = 7.0
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=5)
                with db.atomic_balance_update(
                    balance_change=delta,
                    reason=f"concurrency_test_{i}",
                    updated_by="test",
                    session_id=session_id,
                ):
                    pass
            except Exception as e:  # pragma: no cover - surfaced via the assertion below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Unexpected errors from concurrent writers: {errors}"
        final_balance = db.get_current_balance(session_id)
        assert final_balance == pytest.approx(1000.0 + n_threads * delta)


class TestConcurrentMixedWriters:
    """The exact scenario from the #735 report: the main trading loop's entry
    fee (atomic_balance_update) races the reconciler's exit PnL realization
    (log_trade's balance_delta) on the SAME session. Both deltas must land."""

    def test_atomic_balance_update_and_log_trade_do_not_clobber_each_other(
        self, db: DatabaseManager, session_id: int
    ):
        position_id = db.log_position(
            symbol="ETHUSDT",
            side="long",
            entry_price=2000.0,
            size=0.1,
            strategy_name="test",
            entry_order_id=f"concurrency_entry_{uuid.uuid4().hex}",
            quantity=1.0,
            session_id=session_id,
        )
        starting_balance = db.get_current_balance(session_id)
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def entry_fee_writer() -> None:
            try:
                barrier.wait(timeout=5)
                with db.atomic_balance_update(
                    balance_change=-3.0,
                    reason="entry_fee",
                    updated_by="test",
                    session_id=session_id,
                ):
                    pass
            except Exception as e:
                errors.append(e)

        def exit_pnl_writer() -> None:
            try:
                barrier.wait(timeout=5)
                db.log_trade(
                    symbol="ETHUSDT",
                    side="long",
                    entry_price=2000.0,
                    exit_price=2050.0,
                    size=0.1,
                    entry_time=datetime.now(UTC),
                    exit_time=datetime.now(UTC),
                    pnl=50.0,
                    exit_reason="stop_loss",
                    strategy_name="test",
                    source=TradeSource.PAPER,
                    session_id=session_id,
                    position_id=position_id,
                    balance_delta=50.0,
                )
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=entry_fee_writer)
        t_b = threading.Thread(target=exit_pnl_writer)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        assert errors == [], f"Unexpected errors from concurrent writers: {errors}"
        assert db.get_current_balance(session_id) == pytest.approx(starting_balance - 3.0 + 50.0)


class TestAtomicBalanceCorrection:
    """#735b: account_sync's absolute-value corrections must not clobber a
    concurrent delta writer. A correction converges to its own target by
    construction (new_balance = fresh_current + (target - fresh_current) ==
    target) regardless of timing, so the interesting property isn't "the sum
    of both" (that only holds for two pure deltas) — it's that
    atomic_balance_correction shares the SAME advisory lock as
    atomic_balance_update, so the two writers fully serialize instead of one's
    unlocked insert landing in the middle of the other's locked
    read-then-insert and erasing its contribution outright.

    With serialization, the only two possible outcomes are: the delta commits
    first (correction's fresh read sees it, final == target) or the
    correction commits first (delta then adds on top, final == target +
    delta). A broken (unlocked) correction can also land on a THIRD value —
    the delta alone with the correction's own contribution lost — which is
    what this test rules out.
    """

    def test_correction_and_concurrent_delta_never_produce_a_third_value(
        self, db: DatabaseManager, session_id: int
    ):
        starting_balance = db.get_current_balance(session_id)
        delta = 25.0
        target = starting_balance + 10.0
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def delta_writer() -> None:
            try:
                barrier.wait(timeout=5)
                with db.atomic_balance_update(
                    balance_change=delta,
                    reason="realized_pnl_concurrent",
                    updated_by="test",
                    session_id=session_id,
                ):
                    pass
            except Exception as e:
                errors.append(e)

        def correction_writer() -> None:
            try:
                barrier.wait(timeout=5)
                with db.atomic_balance_correction(
                    target,
                    reason="exchange_sync_correction",
                    updated_by="test",
                    session_id=session_id,
                ):
                    pass
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=delta_writer)
        t_b = threading.Thread(target=correction_writer)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        assert errors == [], f"Unexpected errors from concurrent writers: {errors}"
        final_balance = db.get_current_balance(session_id)
        assert final_balance == pytest.approx(target) or final_balance == pytest.approx(
            target + delta
        ), (
            f"final balance {final_balance} is neither target ({target}) nor "
            f"target+delta ({target + delta}) — a writer's contribution was lost"
        )
