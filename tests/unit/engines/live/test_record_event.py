"""Unit tests for LiveTradingEngine._record_event structured observability.

The emitter only depends on three attributes (``db_manager``,
``trading_session_id`` and ``_send_alert``), so the engine is instantiated via
``__new__`` to keep these tests fast, deterministic and free of the heavy
strategy/data-provider wiring required by the real constructor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.database.models import EventType
from src.engines.live.trading_engine import LiveTradingEngine

pytestmark = [pytest.mark.unit, pytest.mark.fast]


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
        engine._send_alert = MagicMock(return_value=True)  # webhook delivered

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

    def test_alert_not_dispatched_records_alert_sent_false(self):
        """When _send_alert returns False (no webhook configured / POST failed),
        alert_sent is recorded as False and alert_method as None — operators must
        not be misled into believing a critical event paged them when it didn't."""
        engine = _make_engine()
        engine._send_alert = MagicMock(return_value=False)  # nothing delivered

        engine._record_event(
            EventType.ALERT,
            "Emergency close",
            severity="critical",
            component="execution",
            error_code="EMERGENCY_CLOSE",
            alert=True,
        )

        engine._send_alert.assert_called_once_with("Emergency close")
        _, kwargs = engine.db_manager.log_event.call_args
        assert kwargs["alert_sent"] is False
        assert kwargs["alert_method"] is None

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


class TestSendAlert:
    """_send_alert must report the REAL delivery outcome so alert_sent is honest:
    True only on a 2xx webhook response; False when unconfigured, on a non-2xx
    status (requests does not raise on those), or on a network exception."""

    @staticmethod
    def _engine(webhook: str | None) -> LiveTradingEngine:
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.alert_webhook_url = webhook
        return engine

    def test_no_webhook_returns_false(self):
        """No webhook configured -> not delivered (False); no POST attempted."""
        assert self._engine(None)._send_alert("hi") is False

    def test_2xx_returns_true(self):
        """A 2xx webhook response -> alert delivered (True)."""
        engine = self._engine("http://hook.test")
        with patch("requests.post", autospec=True) as post:
            post.return_value.raise_for_status.return_value = None  # 2xx
            assert engine._send_alert("hi") is True
            post.assert_called_once()

    def test_non_2xx_returns_false(self):
        """A 4xx/5xx response (raise_for_status raises) means NOT delivered."""
        engine = self._engine("http://hook.test")
        with patch("requests.post", autospec=True) as post:
            post.return_value.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
            assert engine._send_alert("hi") is False

    def test_post_exception_returns_false(self):
        """A network exception during the POST -> not delivered (False)."""
        engine = self._engine("http://hook.test")
        with patch("requests.post", autospec=True, side_effect=ConnectionError("network down")):
            assert engine._send_alert("hi") is False


class TestWarnIfNoAlertChannel:
    """Startup must make a missing alert channel LOUD, not silent: with no
    webhook, _send_alert is a no-op so critical events page nobody. The guard
    records that blind spot in system_events on every startup."""

    @staticmethod
    def _engine(webhook: str | None, live: bool) -> LiveTradingEngine:
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.alert_webhook_url = webhook
        engine.enable_live_trading = live
        engine._record_event = MagicMock()
        return engine

    def test_no_event_when_channel_configured(self):
        engine = self._engine("http://hook.test", live=True)
        engine._warn_if_no_alert_channel()
        engine._record_event.assert_not_called()

    def test_warning_event_when_unset_in_paper(self):
        engine = self._engine(None, live=False)
        engine._warn_if_no_alert_channel()
        engine._record_event.assert_called_once()
        args, kwargs = engine._record_event.call_args
        assert args[0] == EventType.WARNING
        assert kwargs["error_code"] == "NO_ALERT_CHANNEL"
        assert kwargs["component"] == "engine"
        assert kwargs["severity"] == "warning"

    def test_critical_event_when_unset_in_live(self):
        engine = self._engine(None, live=True)
        engine._warn_if_no_alert_channel()
        _, kwargs = engine._record_event.call_args
        assert kwargs["severity"] == "critical"
        assert kwargs["error_code"] == "NO_ALERT_CHANNEL"

    def test_never_raises_if_record_event_fails(self):
        """Observability must never break startup."""
        engine = self._engine(None, live=True)
        engine._record_event.side_effect = RuntimeError("db down")
        engine._warn_if_no_alert_channel()  # must not raise


class TestRunnerWebhookEnvFallback:
    """--webhook-url falls back to $ALERT_WEBHOOK_URL so production (started via a
    fixed command) can configure operator paging via env (RC-A delivery fix)."""

    def test_falls_back_to_env(self, monkeypatch):
        import sys

        from src.engines.live.runner import parse_args

        monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://env-hook.test")
        monkeypatch.setattr(sys, "argv", ["prog", "ml_basic"])
        assert parse_args().webhook_url == "http://env-hook.test"

    def test_flag_overrides_env(self, monkeypatch):
        import sys

        from src.engines.live.runner import parse_args

        monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://env-hook.test")
        monkeypatch.setattr(sys, "argv", ["prog", "ml_basic", "--webhook-url", "http://flag.test"])
        assert parse_args().webhook_url == "http://flag.test"

    def test_none_when_unset(self, monkeypatch):
        import sys

        from src.engines.live.runner import parse_args

        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        monkeypatch.setattr(sys, "argv", ["prog", "ml_basic"])
        assert parse_args().webhook_url is None


class TestLoopCrashEvent:
    """An abnormal loop death must page + emit a distinct system_event so it is
    distinguishable from a clean ENGINE_STOP — the 2026-05-19 zombie-bot class
    where the process stayed up but the trading loop was dead (#853)."""

    def test_unhandled_loop_crash_pages_and_flags(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine._loop_crashed = False
        engine._record_event = MagicMock()
        engine._trading_loop = MagicMock(side_effect=RuntimeError("boom"))

        # Must not propagate — the wrapper exists precisely to catch it.
        engine._run_trading_loop("BTCUSDT", "1h")

        assert engine._loop_crashed is True
        engine._record_event.assert_called_once()
        args, kwargs = engine._record_event.call_args
        assert args[0] == EventType.ALERT
        assert kwargs["error_code"] == "LOOP_CRASH"
        assert kwargs["severity"] == "critical"
        assert kwargs["alert"] is True
        assert isinstance(kwargs["exc"], RuntimeError)


class TestOrderTrackerAlertAdapter:
    """OrderTracker calls this adapter while holding its per-order lock, so the
    (blocking, up-to-10s) alert webhook must be delivered OFF-thread — not inline
    under the lock (#853; the webhook-under-lock class PR #857's review flagged)."""

    def test_dispatches_record_event_off_thread(self):
        import threading

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        delivered = threading.Event()
        engine._record_event = MagicMock(side_effect=lambda *a, **k: delivered.set())

        # Returns immediately (does not block on _record_event / the webhook).
        engine._order_tracker_alert("position orphaned!", "ORDER_ORPHANED")

        assert delivered.wait(timeout=2.0), "adapter did not deliver the event off-thread"
        args, kwargs = engine._record_event.call_args
        assert args[0] == EventType.ALERT
        assert kwargs["error_code"] == "ORDER_ORPHANED"
        assert kwargs["component"] == "order_tracker"
        assert kwargs["severity"] == "critical"
        assert kwargs["alert"] is True
