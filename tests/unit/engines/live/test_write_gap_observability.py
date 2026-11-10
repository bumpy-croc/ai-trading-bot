"""Engine-side observability for failed audit writes and database outages (#669, #689).

Both gaps share one shape: the failure happens exactly when the database is
unhealthy, so the signal has to be a rate-limited page that does not depend on
the thing that failed, plus a durable record once it recovers.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.database.models import EventType
from src.engines.live import trading_engine as engine_module
from src.engines.live.trading_engine import (
    AUDIT_FAILURE_EVENT_INTERVAL_SECONDS,
    DEFAULT_DB_OUTAGE_ALERT_SECONDS,
)
from tests.unit.engines.live.test_system_halt_gate import _make_engine, _run_one_loop_iteration

pytestmark = pytest.mark.fast


@pytest.fixture
def engine():
    """Real engine whose events are captured and whose daemon threads run inline."""
    eng = _make_engine()
    eng._record_event = MagicMock()
    with patch.object(
        engine_module,
        "create_live_thread",
        side_effect=lambda target, name=None: SimpleNamespace(start=target),
    ):
        yield eng


def _events(engine) -> list[dict]:
    return [{"args": c.args, **c.kwargs} for c in engine._record_event.call_args_list]


class TestAuditFailureSink:
    def test_engine_registers_itself_as_the_database_sink(self):
        eng = _make_engine()

        eng.db_manager.set_audit_failure_sink.assert_called_once_with(eng._on_audit_write_failed)

    def test_failed_audit_write_is_recorded_as_an_error_event(self, engine):
        engine._on_audit_write_failed(
            "HIGH", "position.status: Stop-loss filled", RuntimeError("db hiccup")
        )

        (event,) = _events(engine)
        assert event["args"][0] == EventType.ERROR
        assert event["component"] == "reconciler"
        assert event["error_code"] == "AUDIT_WRITE_FAILED"
        assert event["severity"] == "error"
        assert "[HIGH] position.status: Stop-loss filled" in event["args"][1]
        assert "RuntimeError: db hiccup" in event["args"][1]

    def test_a_lost_critical_row_pages_an_operator(self, engine):
        engine._on_audit_write_failed("CRITICAL", "balance.total_balance: heal", RuntimeError("x"))

        assert _events(engine)[0]["alert"] is True

    @pytest.mark.parametrize("severity", ["HIGH", "MEDIUM", "LOW"])
    def test_a_lost_lower_severity_row_does_not_page(self, engine, severity):
        engine._on_audit_write_failed(severity, "position.status: x", RuntimeError("x"))

        assert _events(engine)[0]["alert"] is False

    def test_repeat_failures_inside_the_window_are_folded_into_one_event(self, engine):
        for _ in range(5):
            engine._on_audit_write_failed("CRITICAL", "balance.total_balance: x", RuntimeError("x"))

        assert len(_events(engine)) == 1

    def test_next_event_after_the_window_reports_the_folded_count(self, engine):
        for _ in range(4):
            engine._on_audit_write_failed("HIGH", "position.status: x", RuntimeError("x"))
        engine._audit_failure_last_emit = time.monotonic() - AUDIT_FAILURE_EVENT_INTERVAL_SECONDS

        engine._on_audit_write_failed("HIGH", "position.status: x", RuntimeError("x"))

        events = _events(engine)
        assert len(events) == 2
        assert "(+3 more failures since the last report)" in events[1]["args"][1]

    def test_a_thread_start_failure_never_propagates(self):
        eng = _make_engine()
        eng._record_event = MagicMock()
        with patch.object(
            engine_module, "create_live_thread", side_effect=RuntimeError("no threads")
        ):
            eng._on_audit_write_failed("CRITICAL", "balance.total_balance: x", RuntimeError("x"))

        eng._record_event.assert_not_called()


class TestDatabaseOutageEvents:
    def test_outage_below_the_threshold_does_not_page(self, engine):
        engine._announce_db_outage(DEFAULT_DB_OUTAGE_ALERT_SECONDS - 1)

        engine._record_event.assert_not_called()

    def test_outage_past_the_threshold_pages_once(self, engine):
        engine._announce_db_outage(DEFAULT_DB_OUTAGE_ALERT_SECONDS)
        engine._announce_db_outage(DEFAULT_DB_OUTAGE_ALERT_SECONDS + 60)
        engine._announce_db_outage(DEFAULT_DB_OUTAGE_ALERT_SECONDS + 120)

        (event,) = _events(engine)
        assert event["args"][0] == EventType.ALERT
        assert event["severity"] == "critical"
        assert event["component"] == "database"
        assert event["error_code"] == "DB_UNREACHABLE"
        assert event["alert"] is True
        assert "unreachable for 300s" in event["args"][1]

    def test_recovery_without_an_outage_writes_nothing(self, engine):
        engine.db_unreachable_since = None

        engine._record_db_recovery()

        engine._record_event.assert_not_called()

    def test_recovery_records_the_outage_duration(self, engine):
        engine.db_unreachable_since = time.monotonic() - 125

        engine._record_db_recovery()

        (event,) = _events(engine)
        assert event["args"][0] == EventType.WARNING
        assert event["component"] == "database"
        assert event["error_code"] == "DB_OUTAGE_RECOVERED"
        assert event["severity"] == "warning"
        assert "after a 125s outage" in event["args"][1]
        assert event.get("alert", False) is False

    def test_a_later_outage_pages_again_after_recovery(self, engine):
        engine._announce_db_outage(DEFAULT_DB_OUTAGE_ALERT_SECONDS)
        engine.db_unreachable_since = time.monotonic() - DEFAULT_DB_OUTAGE_ALERT_SECONDS
        engine._record_db_recovery()
        engine._record_event.reset_mock()

        engine._announce_db_outage(DEFAULT_DB_OUTAGE_ALERT_SECONDS)

        assert len(_events(engine)) == 1

    def test_loop_pages_a_sustained_outage_from_the_error_handler(self, engine):
        engine.is_running = True
        engine._sleep_with_interrupt = MagicMock()
        engine._get_latest_data = MagicMock(
            side_effect=_transient_db_error(),
        )
        engine.db_unreachable_since = time.monotonic() - DEFAULT_DB_OUTAGE_ALERT_SECONDS - 5

        engine._trading_loop("BTCUSDT", "1h", max_steps=2)

        codes = [e["error_code"] for e in _events(engine)]
        assert codes.count("DB_UNREACHABLE") == 1

    def test_loop_records_recovery_on_the_first_successful_iteration(self, engine):
        engine.db_unreachable_since = time.monotonic() - 90

        _run_one_loop_iteration(engine, entry_probe=None, exit_probe=None)

        codes = [e["error_code"] for e in _events(engine)]
        assert "DB_OUTAGE_RECOVERED" in codes
        assert engine.db_unreachable_since is None


def _transient_db_error() -> Exception:
    from sqlalchemy.exc import OperationalError

    return OperationalError("SELECT 1", {}, Exception("could not connect to server"))
