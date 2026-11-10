"""`DatabaseManager.log_audit_event` is best-effort and its failures are observable (#689)."""

from unittest.mock import Mock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_AUDIT_ARGS = {
    "session_id": 7,
    "entity_type": "balance",
    "entity_id": None,
    "field": "total_balance",
    "old_value": "100.00",
    "new_value": "90.00",
    "reason": "Periodic balance correction",
    "severity": "CRITICAL",
}


class TestLogAuditEvent:
    def test_returns_the_new_row_id_on_success(self, mock_postgresql_db):
        row = Mock()
        row.id = 55

        with patch("database.manager.ReconciliationAuditEvent", return_value=row) as row_cls:
            audit_id = mock_postgresql_db.log_audit_event(**_AUDIT_ARGS)

        assert audit_id == 55
        assert row_cls.call_args.kwargs["severity"] == "CRITICAL"
        mock_postgresql_db._mock_session.commit.assert_called()

    def test_failed_write_does_not_raise_and_returns_none(self, mock_postgresql_db):
        with patch.object(
            mock_postgresql_db, "get_session_with_timeout", side_effect=RuntimeError("db down")
        ):
            audit_id = mock_postgresql_db.log_audit_event(**_AUDIT_ARGS)

        assert audit_id is None

    def test_commit_failure_does_not_raise(self, mock_postgresql_db):
        mock_postgresql_db._mock_session.commit.side_effect = RuntimeError("commit failed")

        with patch("database.manager.ReconciliationAuditEvent"):
            audit_id = mock_postgresql_db.log_audit_event(**_AUDIT_ARGS)

        assert audit_id is None

    def test_failed_write_is_reported_to_the_registered_sink(self, mock_postgresql_db):
        sink = Mock()
        mock_postgresql_db.set_audit_failure_sink(sink)
        error = RuntimeError("value too long for type character varying(255)")

        with patch.object(mock_postgresql_db, "get_session_with_timeout", side_effect=error):
            mock_postgresql_db.log_audit_event(**_AUDIT_ARGS)

        sink.assert_called_once_with(
            "CRITICAL", "balance.total_balance: Periodic balance correction", error
        )

    def test_successful_write_does_not_touch_the_sink(self, mock_postgresql_db):
        sink = Mock()
        mock_postgresql_db.set_audit_failure_sink(sink)

        with patch("database.manager.ReconciliationAuditEvent", return_value=Mock(id=1)):
            mock_postgresql_db.log_audit_event(**_AUDIT_ARGS)

        sink.assert_not_called()

    def test_failure_without_a_sink_is_still_swallowed(self, mock_postgresql_db):
        mock_postgresql_db.set_audit_failure_sink(None)

        with patch.object(
            mock_postgresql_db, "get_session_with_timeout", side_effect=RuntimeError("db down")
        ):
            assert mock_postgresql_db.log_audit_event(**_AUDIT_ARGS) is None

    def test_a_raising_sink_cannot_break_the_caller(self, mock_postgresql_db):
        mock_postgresql_db.set_audit_failure_sink(Mock(side_effect=RuntimeError("sink broke")))

        with patch.object(
            mock_postgresql_db, "get_session_with_timeout", side_effect=RuntimeError("db down")
        ):
            assert mock_postgresql_db.log_audit_event(**_AUDIT_ARGS) is None
