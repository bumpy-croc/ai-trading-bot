"""Tests for live day-start balance recovery (regression for #766).

``LiveEventLogger._get_day_start_balance_from_db`` queried a DatabaseManager
method that did not exist, and nothing invoked the recovery helper — so after
any restart the daily P&L baseline silently reset to the restart-time balance.
Recovery is now wired into the first ``_check_and_update_trading_date`` call.
"""

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

from src.database.manager import DatabaseManager
from src.engines.live.logging.event_logger import LiveEventLogger

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _logger_with_db(mock_db) -> LiveEventLogger:
    return LiveEventLogger(
        db_manager=mock_db,
        log_to_database=True,
        log_trades_to_file=False,
        session_id=7,
    )


class TestDayStartBalanceRecovery:
    def test_first_check_recovers_day_start_from_db(self):
        """An intraday restart anchors daily P&L to the day's first snapshot,
        not the restart-time balance. Autospec enforces the real
        get_first_snapshot_of_day signature."""
        mock_db = create_autospec(DatabaseManager, instance=True)
        mock_db.get_first_snapshot_of_day.return_value = SimpleNamespace(balance=1000.0)

        event_logger = _logger_with_db(mock_db)
        event_logger._check_and_update_trading_date(current_balance=1080.0)

        mock_db.get_first_snapshot_of_day.assert_called_once()
        assert mock_db.get_first_snapshot_of_day.call_args.kwargs["session_id"] == 7
        assert event_logger._day_start_balance == 1000.0

    def test_falls_back_to_current_balance_without_snapshot(self):
        """Fresh day / fresh session (no snapshot yet) keeps old behavior."""
        mock_db = create_autospec(DatabaseManager, instance=True)
        mock_db.get_first_snapshot_of_day.return_value = None

        event_logger = _logger_with_db(mock_db)
        event_logger._check_and_update_trading_date(current_balance=1080.0)

        assert event_logger._day_start_balance == 1080.0

    def test_data_shape_failure_degrades_to_current_balance(self):
        """Data-shape errors (bad snapshot payload) degrade to the
        restart-time balance instead of raising."""
        mock_db = create_autospec(DatabaseManager, instance=True)
        mock_db.get_first_snapshot_of_day.side_effect = ValueError("bad balance")

        event_logger = _logger_with_db(mock_db)
        event_logger._check_and_update_trading_date(current_balance=1080.0)

        assert event_logger._day_start_balance == 1080.0

    def test_unexpected_db_error_propagates(self):
        """Unexpected DB-level errors are NOT swallowed here — they propagate
        to the engine's snapshot wrapper, which logs with a traceback and
        skips the snapshot (per PR review: a broad except hid connectivity
        failures as data noise)."""
        mock_db = create_autospec(DatabaseManager, instance=True)
        mock_db.get_first_snapshot_of_day.side_effect = RuntimeError("db down")

        event_logger = _logger_with_db(mock_db)

        with pytest.raises(RuntimeError, match="db down"):
            event_logger._check_and_update_trading_date(current_balance=1080.0)

    def test_no_db_or_session_returns_none(self):
        """Recovery helper degrades cleanly without a DB manager or session."""
        event_logger = LiveEventLogger(
            db_manager=None, log_to_database=False, log_trades_to_file=False, session_id=None
        )

        assert event_logger._get_day_start_balance_from_db() is None

    def test_rollover_resets_to_current_balance(self):
        """A date rollover uses the live balance, not a stale DB snapshot."""
        mock_db = create_autospec(DatabaseManager, instance=True)
        mock_db.get_first_snapshot_of_day.return_value = SimpleNamespace(balance=1000.0)

        event_logger = _logger_with_db(mock_db)
        event_logger._check_and_update_trading_date(current_balance=1080.0)
        assert event_logger._day_start_balance == 1000.0

        # Simulate the date rolling over since the first call (UTC date —
        # local date.today() here is flaky on UTC+N hosts during the window
        # between local and UTC midnight, the exact skew this PR fixes)
        from datetime import UTC, datetime, timedelta

        event_logger._current_trading_date = datetime.now(UTC).date() - timedelta(days=1)
        event_logger._check_and_update_trading_date(current_balance=1200.0)

        # Rollover re-anchors to the live balance; the DB is not re-queried
        assert event_logger._day_start_balance == 1200.0
        assert mock_db.get_first_snapshot_of_day.call_count == 1

    def test_set_day_start_balance_uses_utc_date(self):
        """The explicit setter stamps the UTC trading date."""
        from datetime import UTC, datetime
        from unittest.mock import patch

        fixed_date = datetime.now(UTC).date()
        event_logger = _logger_with_db(Mock())
        with patch.object(event_logger, "_utc_today", return_value=fixed_date):
            event_logger.set_day_start_balance(999.0)

        assert event_logger._day_start_balance == 999.0
        assert event_logger._current_trading_date == fixed_date


class TestEngineSnapshotRouting:
    """The live engine must route snapshots through LiveEventLogger (#766
    review): it previously wrote snapshots directly with daily_pnl=0 and the
    recovery wiring was dead code on the live path."""

    @pytest.mark.fast
    def test_engine_snapshot_delegates_to_event_logger(self):
        """_log_account_snapshot forwards balance/positions/pnl/peak to the
        event logger (which owns daily P&L tracking and recovery)."""
        from unittest.mock import MagicMock

        from src.engines.live.trading_engine import LiveTradingEngine

        mock_engine = MagicMock()
        mock_engine.current_balance = 1080.0
        perf = MagicMock()
        perf.total_pnl = 80.0
        perf.peak_balance = 1100.0
        mock_engine.performance_tracker.get_metrics.return_value = perf
        mock_engine.live_position_tracker.positions = {}

        LiveTradingEngine._log_account_snapshot(mock_engine)

        kwargs = mock_engine.event_logger.log_account_snapshot.call_args.kwargs
        assert kwargs["balance"] == 1080.0
        assert kwargs["total_pnl"] == 80.0
        assert kwargs["peak_balance"] == 1100.0
        assert kwargs["positions"] == {}

    @pytest.mark.fast
    def test_engine_snapshot_failure_does_not_raise(self):
        """Snapshot failures must never crash the trading loop."""
        from unittest.mock import MagicMock

        from src.engines.live.trading_engine import LiveTradingEngine

        mock_engine = MagicMock()
        mock_engine.performance_tracker.get_metrics.side_effect = RuntimeError("boom")

        LiveTradingEngine._log_account_snapshot(mock_engine)
