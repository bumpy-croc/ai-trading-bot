"""DatabaseManager ``system_halt`` control flag: durable manual kill-switch state (#922).

The manual kill-switch (`atb live-control halt`) persists a single
``system_control_flags`` row named ``system_halt`` that the live trading loop
polls each iteration. These tests run against a real in-memory SQLite database
to exercise the actual upsert/read semantics.
"""

from __future__ import annotations

import pytest

from src.database.manager import DatabaseManager
from src.database.models import SystemControlFlag

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


class TestSystemHaltFlag:
    def test_defaults_to_inactive_when_row_missing(self, db):
        """A fresh database reads as not-halted (fail-open for normal boot)."""
        status = db.get_system_halt()

        assert status.active is False
        assert status.reason is None
        assert status.source is None
        assert status.updated_at is None

    def test_set_halt_persists_reason_and_source(self, db):
        """Halting durably records who pulled the switch and why."""
        db.set_system_halt(True, reason="kill-switch drill", source="cli:alex")

        status = db.get_system_halt()

        assert status.active is True
        assert status.reason == "kill-switch drill"
        assert status.source == "cli:alex"
        assert status.updated_at is not None

    def test_resume_clears_halt(self, db):
        """An explicit resume flips the flag off and records the resume reason."""
        db.set_system_halt(True, reason="drill", source="cli:alex")

        db.set_system_halt(False, reason="drill complete", source="cli:alex")
        status = db.get_system_halt()

        assert status.active is False
        assert status.reason == "drill complete"

    def test_set_returns_persisted_snapshot(self, db):
        """The setter returns the state it just wrote (for CLI printouts)."""
        snapshot = db.set_system_halt(True, reason="macro event", source="cli:ops")

        assert snapshot.active is True
        assert snapshot.reason == "macro event"
        assert snapshot.source == "cli:ops"
        assert snapshot.updated_at is not None

    def test_upsert_reuses_single_row(self, db):
        """Repeated halt/resume cycles update one row, not append rows."""
        db.set_system_halt(True, reason="a", source="s")
        db.set_system_halt(False, reason="b", source="s")
        db.set_system_halt(True, reason="c", source="s")

        with db.get_session() as session:
            assert session.query(SystemControlFlag).count() == 1

        status = db.get_system_halt()
        assert status.active is True
        assert status.reason == "c"
