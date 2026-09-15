"""Regression tests for the live-loop status/account-snapshot heartbeats (#1170).

Two independent defects, both in `LiveTradingEngine`:

1. The `📊 Status:` line was gated on `total_trades % 10 == 0 or
   position_count > 0`. A flat session (`position_count == 0`) sitting on a
   trade count not divisible by 10 stopped emitting the line even though the
   loop was still alive — `.claude/LESSONS.md` §5 treats this line as
   liveness evidence, so this silently broke that signal. It only ever looked
   unconditional because a fresh session starts at 0 trades (`0 % 10 == 0`).
   Fixed by making `_log_status_heartbeat` a wall-clock-interval gate that
   never consults trade count.

2. `_log_periodic_account_state` compared elapsed time via
   `timedelta.seconds`, which is the "seconds within the current day"
   component, not total elapsed seconds. A gap spanning more than 24h (e.g. a
   quiet weekend) silently reports a much smaller elapsed time than actually
   passed. Fixed by using `timedelta.total_seconds()`.

A third, review-caught defect: firing far more often (every wall-clock
interval, not just on trade-count multiples) raised the odds of ``log_status``
hitting a transient bad read. Unlike its sibling ``log_account_snapshot``,
``AccountMonitor.log_status`` had no fault isolation, so a single malformed
position could propagate an exception into the trading loop's generic error
handler -- silencing the heartbeat exactly when something is already wrong.
Fixed by wrapping it the same way.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.config.constants import DEFAULT_STATUS_LOG_INTERVAL
from src.data_providers.data_provider import DataProvider
from src.engines.live.trading_engine import LiveTradingEngine

pytestmark = pytest.mark.fast


def _make_engine() -> LiveTradingEngine:
    """A paper-mode engine with the database manager patched out."""
    from src.strategies.ml_basic import create_ml_basic_strategy

    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=Mock(spec=DataProvider),
            initial_balance=10000,
            enable_live_trading=False,
        )


# --------------------------------------------------------------------------- #
# Item 1: status heartbeat must not depend on trade-count modulo
# --------------------------------------------------------------------------- #


class TestStatusHeartbeatCadence:
    def test_fires_when_flat_with_trade_count_not_divisible_by_ten(self):
        """The exact bug symptom: flat (position_count == 0), total_trades %
        10 != 0. The old gate skipped this; the heartbeat must still fire."""
        engine = _make_engine()
        engine._log_status = Mock()
        engine.performance_tracker.get_metrics = Mock(return_value=Mock(total_trades=7))
        assert engine.live_position_tracker.position_count == 0  # flat session
        engine.last_status_log = None

        engine._log_status_heartbeat("BTCUSDT", 100.0)

        engine._log_status.assert_called_once_with("BTCUSDT", 100.0)

    def test_does_not_refire_before_interval_elapses(self):
        engine = _make_engine()
        engine._log_status = Mock()
        engine.last_status_log = datetime.now(UTC)

        engine._log_status_heartbeat("BTCUSDT", 100.0)

        engine._log_status.assert_not_called()

    def test_refires_once_interval_elapses(self):
        engine = _make_engine()
        engine._log_status = Mock()
        engine.last_status_log = datetime.now(UTC) - timedelta(
            seconds=DEFAULT_STATUS_LOG_INTERVAL + 1
        )

        engine._log_status_heartbeat("BTCUSDT", 100.0)

        engine._log_status.assert_called_once()

    def test_fires_unconditionally_on_first_call(self):
        """No prior timestamp — must fire regardless of interval config."""
        engine = _make_engine()
        engine._log_status = Mock()
        engine.last_status_log = None

        engine._log_status_heartbeat("BTCUSDT", 100.0)

        engine._log_status.assert_called_once_with("BTCUSDT", 100.0)
        assert engine.last_status_log is not None


# --------------------------------------------------------------------------- #
# Item 3: log_status must not crash the trading loop on a bad read
# --------------------------------------------------------------------------- #


class TestLogStatusIsFaultIsolated:
    def test_a_malformed_position_does_not_propagate(self):
        """The real symptom: a position whose unrealized_pnl cannot be
        floated must not raise out of log_status and into the trading loop's
        generic error handler."""
        engine = _make_engine()
        bad_position = Mock()
        bad_position.unrealized_pnl = object()  # float() raises TypeError
        engine.live_position_tracker._positions = {"order-1": bad_position}

        # Must not raise.
        engine.account_monitor.log_status("BTCUSDT", 100.0)

    def test_a_healthy_call_still_logs_normally(self):
        """Sanity check the wrap doesn't swallow the happy path."""
        engine = _make_engine()
        engine.live_position_tracker._positions = {}

        with patch("src.engines.live.monitoring.account_monitor.logger") as mock_logger:
            engine.account_monitor.log_status("BTCUSDT", 100.0)

        mock_logger.info.assert_called_once()
        mock_logger.error.assert_not_called()


# --------------------------------------------------------------------------- #
# Item 2: account-snapshot interval must use total elapsed seconds, not
# timedelta.seconds (which drops whole days)
# --------------------------------------------------------------------------- #


class TestAccountSnapshotIntervalAcrossDayBoundary:
    def test_fires_after_25_hours_with_2_hour_interval(self):
        """25h elapsed, 2h (7200s) configured interval.

        `timedelta(hours=25).seconds == 3600` (the "seconds within the day"
        component) while `.total_seconds() == 90000`. With the buggy
        `.seconds` comparison, 3600 >= 7200 is False, so the snapshot would
        wrongly NOT fire despite 25h having elapsed — more than ten times the
        configured interval. The fix must fire.
        """
        engine = _make_engine()
        engine._log_account_snapshot = Mock()
        engine.account_synchronizer = None
        engine.account_snapshot_interval = 7200  # 2 hours
        engine.last_account_snapshot = datetime.now(UTC) - timedelta(hours=25)

        # Sanity-check the trap this test is pinned against.
        elapsed = datetime.now(UTC) - engine.last_account_snapshot
        assert elapsed.seconds < engine.account_snapshot_interval
        assert elapsed.total_seconds() > engine.account_snapshot_interval

        engine._log_periodic_account_state()

        engine._log_account_snapshot.assert_called_once()
        assert engine.last_account_snapshot is not None

    def test_does_not_fire_before_interval_elapses(self):
        engine = _make_engine()
        engine._log_account_snapshot = Mock()
        engine.account_synchronizer = None
        engine.account_snapshot_interval = 1800
        engine.last_account_snapshot = datetime.now(UTC)

        engine._log_periodic_account_state()

        engine._log_account_snapshot.assert_not_called()
