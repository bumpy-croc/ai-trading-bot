"""Unit tests for the loop-driven cold-boot margin-equity retry (#1226 review of #659).

Both `architecture-reviewer` and `code-reviewer` found that
`AccountSynchronizer._pending_startup_equity_retry` bypasses the internal sync
throttle for exactly one call, but in production the only scheduled caller of
`sync_account_data()` besides startup is `_log_periodic_account_state`, gated
behind the one-hour `DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL`. So a cold-boot skip
that survives its own inline retry would sit armed but unused for up to an
hour -- exactly the oversized-sizing window #659 describes.

`LiveTradingEngine._check_pending_startup_equity_retry` drives the retry from
the loop's own (seconds-scale) cadence instead, bounded by
`DEFAULT_STARTUP_EQUITY_LOOP_MAX_ATTEMPTS` so a permanently-broken equity
endpoint can't spin forever.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.config.constants import DEFAULT_STARTUP_EQUITY_LOOP_MAX_ATTEMPTS
from src.engines.live.account_sync import SyncResult
from src.engines.live.trading_engine import LiveTradingEngine

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_engine(
    *, account_synchronizer=None, enable_live_trading: bool = True
) -> LiveTradingEngine:
    """Bare engine with only the collaborators the retry check touches."""
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.account_synchronizer = account_synchronizer
    engine.enable_live_trading = enable_live_trading
    engine._active_symbol = "BTCUSDT"
    engine._balance_lock = threading.Lock()
    engine.current_balance = 99.89
    engine._startup_equity_retry_in_progress = False
    engine._startup_equity_retry_attempts = 0
    return engine


def _sync_result(
    *, success: bool, balance_sync: dict | None = None, message: str = ""
) -> SyncResult:
    return SyncResult(
        success=success,
        message=message,
        data={"balance_sync": balance_sync or {}},
        timestamp=datetime.now(UTC),
    )


class TestNoOpConditions:
    def test_noop_when_no_synchronizer(self):
        engine = _make_engine(account_synchronizer=None)
        engine._check_pending_startup_equity_retry()  # must not raise

    def test_noop_when_live_trading_disabled(self):
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        engine = _make_engine(account_synchronizer=sync, enable_live_trading=False)

        engine._check_pending_startup_equity_retry()

        sync.sync_account_data.assert_not_called()

    def test_noop_when_flag_not_armed_and_no_campaign_in_progress(self):
        sync = MagicMock()
        sync.pending_startup_equity_retry = False
        engine = _make_engine(account_synchronizer=sync)

        engine._check_pending_startup_equity_retry()

        sync.sync_account_data.assert_not_called()


class TestDrivesForcedSync:
    def test_calls_force_sync_when_flag_armed(self):
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.return_value = _sync_result(
            success=True, balance_sync={"corrected": False}
        )
        engine = _make_engine(account_synchronizer=sync)

        engine._check_pending_startup_equity_retry()

        sync.sync_account_data.assert_called_once_with(force=True, symbol="BTCUSDT")

    def test_applies_balance_correction_when_resolved_with_correction(self):
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.return_value = _sync_result(
            success=True, balance_sync={"corrected": True, "new_balance": 84.14}
        )
        engine = _make_engine(account_synchronizer=sync)

        engine._check_pending_startup_equity_retry()

        assert engine.current_balance == pytest.approx(84.14)
        # Resolved -- the campaign must not still be marked in-progress.
        assert engine._startup_equity_retry_in_progress is False
        assert engine._startup_equity_retry_attempts == 0

    def test_resolved_without_correction_still_stops_campaign(self):
        """Equity became readable but no book-down was needed (already in
        sync, or a position is held) -- also a resolution, not a retry case."""
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.return_value = _sync_result(
            success=True, balance_sync={"reason": "position held"}
        )
        engine = _make_engine(account_synchronizer=sync)

        engine._check_pending_startup_equity_retry()

        assert engine._startup_equity_retry_in_progress is False


class TestKeepsRetryingUntilResolved:
    def test_keeps_retrying_across_iterations_while_still_unavailable(self):
        """Discriminates the fix: without loop-driven retry, nothing calls
        sync_account_data() again until the hourly snapshot tick. With it,
        calling the check on successive (simulated) loop iterations keeps
        retrying immediately."""
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.side_effect = [
            _sync_result(success=True, balance_sync={"reason": "equity unavailable"}),
            _sync_result(success=True, balance_sync={"reason": "equity unavailable"}),
            _sync_result(success=True, balance_sync={"corrected": True, "new_balance": 84.14}),
        ]
        engine = _make_engine(account_synchronizer=sync)

        # Three simulated loop iterations.
        engine._check_pending_startup_equity_retry()
        engine._check_pending_startup_equity_retry()
        engine._check_pending_startup_equity_retry()

        assert sync.sync_account_data.call_count == 3
        assert engine.current_balance == pytest.approx(84.14)
        assert engine._startup_equity_retry_in_progress is False

    def test_exchange_level_failure_also_keeps_retrying(self):
        """#1226 P2 scenario: the exchange-level sync itself fails before any
        equity read is attempted. That must be treated as still-unresolved
        (keep retrying), not as a resolution that stops the campaign."""
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.side_effect = [
            _sync_result(success=False, message="Exchange sync failed: boom"),
            _sync_result(success=True, balance_sync={"corrected": True, "new_balance": 84.14}),
        ]
        engine = _make_engine(account_synchronizer=sync)

        engine._check_pending_startup_equity_retry()
        assert engine._startup_equity_retry_in_progress is True  # not resolved yet

        engine._check_pending_startup_equity_retry()
        assert sync.sync_account_data.call_count == 2
        assert engine.current_balance == pytest.approx(84.14)
        assert engine._startup_equity_retry_in_progress is False

    def test_raised_exception_does_not_crash_the_loop_and_keeps_state_in_progress(self):
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.side_effect = RuntimeError("network blip")
        engine = _make_engine(account_synchronizer=sync)

        engine._check_pending_startup_equity_retry()  # must not raise

        assert engine._startup_equity_retry_in_progress is True
        assert engine._startup_equity_retry_attempts == 1


class TestBoundedAttempts:
    def test_gives_up_after_max_attempts_and_stops_calling_sync(self):
        sync = MagicMock()
        sync.pending_startup_equity_retry = True
        sync.sync_account_data.return_value = _sync_result(
            success=True, balance_sync={"reason": "equity unavailable"}
        )
        engine = _make_engine(account_synchronizer=sync)

        for _ in range(DEFAULT_STARTUP_EQUITY_LOOP_MAX_ATTEMPTS + 3):
            engine._check_pending_startup_equity_retry()

        assert sync.sync_account_data.call_count == DEFAULT_STARTUP_EQUITY_LOOP_MAX_ATTEMPTS
        assert engine._startup_equity_retry_in_progress is False
