"""Tests for DatabaseManager.get_first_snapshot_of_day (regression for #766).

The live event logger's day-start balance recovery called this method while it
did not exist, so recovery silently never worked. These tests run against a
real in-memory SQLite database to exercise the actual query semantics.
"""

from datetime import UTC, date, datetime

import pytest

from src.database.manager import DatabaseManager
from src.database.models import AccountHistory

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _insert_snapshot(db: DatabaseManager, session_id: int, ts: datetime, balance: float) -> None:
    with db.get_session() as session:
        session.add(
            AccountHistory(
                timestamp=ts,
                balance=balance,
                equity=balance,
                total_pnl=0.0,
                daily_pnl=0.0,
                drawdown=0.0,
                open_positions=0,
                total_exposure=0.0,
                margin_used=0.0,
                margin_available=balance,
                session_id=session_id,
            )
        )
        session.commit()


class TestGetFirstSnapshotOfDay:
    def test_returns_earliest_snapshot_of_target_day(self, db):
        """The day's earliest row wins regardless of insert order."""
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 15, 0, tzinfo=UTC), 1100.0)
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 0, 5, tzinfo=UTC), 1000.0)
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 8, 30, tzinfo=UTC), 1050.0)

        snapshot = db.get_first_snapshot_of_day(session_id=1, target_date=date(2024, 1, 2))

        assert snapshot is not None
        assert float(snapshot.balance) == 1000.0

    def test_day_window_is_utc_calendar_day(self, db):
        """Rows from the previous/next UTC day are excluded."""
        _insert_snapshot(db, 1, datetime(2024, 1, 1, 23, 59, tzinfo=UTC), 900.0)
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 0, 0, tzinfo=UTC), 1000.0)
        _insert_snapshot(db, 1, datetime(2024, 1, 3, 0, 0, tzinfo=UTC), 1200.0)

        snapshot = db.get_first_snapshot_of_day(session_id=1, target_date=date(2024, 1, 2))

        assert snapshot is not None
        assert float(snapshot.balance) == 1000.0

    def test_filters_by_session(self, db):
        """Another session's snapshots are invisible."""
        _insert_snapshot(db, 2, datetime(2024, 1, 2, 0, 5, tzinfo=UTC), 5000.0)
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 9, 0, tzinfo=UTC), 1000.0)

        snapshot = db.get_first_snapshot_of_day(session_id=1, target_date=date(2024, 1, 2))

        assert snapshot is not None
        assert float(snapshot.balance) == 1000.0

    def test_returns_none_when_day_empty(self, db):
        """No snapshot rows on the target day returns None."""
        _insert_snapshot(db, 1, datetime(2024, 1, 1, 12, 0, tzinfo=UTC), 900.0)

        assert db.get_first_snapshot_of_day(session_id=1, target_date=date(2024, 1, 2)) is None

    def test_returns_none_without_session(self, db):
        """No session id (explicit or current) returns None."""
        assert db.get_first_snapshot_of_day(session_id=None, target_date=date(2024, 1, 2)) is None

    def test_defaults_to_current_utc_day(self, db):
        """target_date defaults to today's UTC date (clock frozen: a midnight
        tick between insert and query would otherwise split the day)."""
        from unittest.mock import patch

        now = datetime.now(UTC)
        _insert_snapshot(db, 1, now, 1234.0)

        with patch("src.database.manager.datetime", wraps=datetime) as mock_dt:
            mock_dt.now.return_value = now
            snapshot = db.get_first_snapshot_of_day(session_id=1)

        assert snapshot is not None
        assert float(snapshot.balance) == 1234.0

    def test_attributes_readable_after_session_closes(self, db):
        """The returned row is detached and remains readable."""
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 0, 5, tzinfo=UTC), 1000.0)

        snapshot = db.get_first_snapshot_of_day(session_id=1, target_date=date(2024, 1, 2))

        assert snapshot is not None
        assert snapshot.session_id == 1
        assert float(snapshot.equity) == 1000.0

    def test_fallback_session_covers_clean_restart(self, db):
        """A clean restart creates a NEW session; the day's earlier snapshots
        live under the prior session and must still be found via
        fallback_session_id (#766 review)."""
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 0, 5, tzinfo=UTC), 1000.0)
        _insert_snapshot(db, 2, datetime(2024, 1, 2, 12, 0, tzinfo=UTC), 1100.0)

        snapshot = db.get_first_snapshot_of_day(
            session_id=2, target_date=date(2024, 1, 2), fallback_session_id=1
        )

        assert snapshot is not None
        assert float(snapshot.balance) == 1000.0

    def test_fallback_session_ignored_when_none(self, db):
        """Without a fallback the query stays strictly session scoped."""
        _insert_snapshot(db, 1, datetime(2024, 1, 2, 0, 5, tzinfo=UTC), 1000.0)

        snapshot = db.get_first_snapshot_of_day(
            session_id=2, target_date=date(2024, 1, 2), fallback_session_id=None
        )

        assert snapshot is None
