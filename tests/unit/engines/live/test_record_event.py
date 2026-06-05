"""Unit tests for LiveTradingEngine._record_event structured observability.

The emitter only depends on three attributes (``db_manager``,
``trading_session_id`` and ``_send_alert``), so the engine is instantiated via
``__new__`` to keep these tests fast, deterministic and free of the heavy
strategy/data-provider wiring required by the real constructor.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.database.models import EventType
from src.engines.live.trading_engine import LiveTradingEngine

pytestmark = pytest.mark.unit


def _make_engine() -> LiveTradingEngine:
    """Build a bare engine with just the collaborators _record_event touches."""
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.db_manager = MagicMock()
    engine.trading_session_id = 42
    engine._send_alert = MagicMock()
    return engine


class TestRecordEvent:
    def test_captures_stack_trace_when_exc_supplied(self):
        """With an active exception, the formatted traceback reaches log_event."""
        engine = _make_engine()

        try:
            raise ValueError("boom")
        except ValueError as exc:
            engine._record_event(
                EventType.ERROR,
                "Reconciler fell back",
                severity="error",
                component="reconciler",
                error_code="RECONCILER_FALLBACK",
                exc=exc,
            )

        engine.db_manager.log_event.assert_called_once()
        _, kwargs = engine.db_manager.log_event.call_args
        assert kwargs["event_type"] == EventType.ERROR
        assert kwargs["component"] == "reconciler"
        assert kwargs["error_code"] == "RECONCILER_FALLBACK"
        assert kwargs["severity"] == "error"
        assert kwargs["session_id"] == 42
        assert "ValueError: boom" in kwargs["stack_trace"]
        assert "Traceback" in kwargs["stack_trace"]
        # No alert requested -> not sent, columns reflect that.
        assert kwargs["alert_sent"] is False
        assert kwargs["alert_method"] is None
        engine._send_alert.assert_not_called()

    def test_no_stack_trace_without_exc(self):
        """Without exc, stack_trace is None (never the 'NoneType: None' sentinel)."""
        engine = _make_engine()

        engine._record_event(
            EventType.WARNING,
            "Balance overwritten",
            severity="warning",
            component="balance",
            error_code="BALANCE_OVERWRITE",
        )

        _, kwargs = engine.db_manager.log_event.call_args
        assert kwargs["stack_trace"] is None

    def test_alert_true_sends_alert_and_sets_flags(self):
        """alert=True dispatches the webhook and records alert_sent/alert_method."""
        engine = _make_engine()

        engine._record_event(
            EventType.ALERT,
            "Close-only mode activated",
            severity="critical",
            component="risk",
            error_code="CLOSE_ONLY",
            alert=True,
        )

        engine._send_alert.assert_called_once_with("Close-only mode activated")
        _, kwargs = engine.db_manager.log_event.call_args
        assert kwargs["alert_sent"] is True
        assert kwargs["alert_method"] == "webhook"
        assert kwargs["severity"] == "critical"

    def test_swallows_log_event_failure(self):
        """A db_manager.log_event failure must never propagate to the caller."""
        engine = _make_engine()
        engine.db_manager.log_event.side_effect = RuntimeError("db down")

        # Must not raise.
        engine._record_event(
            EventType.ERROR,
            "Reconciler start failed",
            severity="error",
            component="reconciler",
            error_code="RECONCILER_START_FAILED",
        )

        engine.db_manager.log_event.assert_called_once()

    def test_swallows_send_alert_failure(self):
        """An alert dispatch failure is also swallowed (still never raises)."""
        engine = _make_engine()
        engine._send_alert.side_effect = RuntimeError("webhook down")

        engine._record_event(
            EventType.ALERT,
            "Emergency close",
            severity="critical",
            component="execution",
            error_code="EMERGENCY_CLOSE",
            alert=True,
        )

        engine._send_alert.assert_called_once()
