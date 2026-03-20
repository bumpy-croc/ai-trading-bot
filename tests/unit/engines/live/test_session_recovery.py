"""Unit tests for LiveTradingEngine session recovery logic.

Covers the _recover_existing_session() method which restores balance across
both crash restarts (active session) and clean restarts (recent inactive session).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine(enable_live_trading: bool = False) -> LiveTradingEngine:
    """Return a minimal LiveTradingEngine with all external I/O mocked out.

    The DatabaseManager is replaced with a MagicMock so tests can configure
    return values without touching a real database.  The data provider is
    similarly mocked.  _active_symbol is pre-set to mimic what start() does
    before calling _recover_existing_session().
    """
    mock_data_provider = MagicMock()
    strategy = create_ml_basic_strategy()

    with patch("src.engines.live.trading_engine.DatabaseManager"):
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=1000.0,
            enable_live_trading=enable_live_trading,
        )

    # Replace the db_manager created during __init__ with a fresh MagicMock
    # so each test gets a clean, unconfigured mock to assert against.
    engine.db_manager = MagicMock()
    # Simulate what start() sets before calling _recover_existing_session().
    engine._active_symbol = "BTCUSDT"
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_recovery_falls_back_to_recent_inactive_session():
    """No active session → falls back to most-recent session within 24 hours.

    Critically: trading_session_id must remain None so start() creates a
    fresh session. Reusing the closed session ID would cause every balance
    update to fail with "No active trading session".
    """
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    engine.db_manager.get_last_session_id = MagicMock(return_value=42)
    engine.db_manager.recover_last_balance = MagicMock(return_value=1234.56)

    result = engine._recover_existing_session()

    assert result == 1234.56
    engine.db_manager.get_last_session_id.assert_called_once()
    # Session ID must NOT be set — start() must create a new active session.
    assert engine.trading_session_id is None


@pytest.mark.fast
def test_active_session_recovery_reuses_session_id():
    """Crash recovery (active session) reuses the existing session ID.

    This ensures trades after a crash are still attributed to the same
    session row rather than opening a duplicate.
    """
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=77)
    engine.db_manager.recover_last_balance = MagicMock(return_value=850.0)

    result = engine._recover_existing_session()

    assert result == 850.0
    assert engine.trading_session_id == 77


@pytest.mark.fast
def test_recovery_ignores_sessions_older_than_7_days():
    """Stale sessions (> 7 days) are not recovered.

    The 7-day window prevents ancient session carryover while covering
    reasonable paper trading durations that may span multiple days.
    """
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    # get_last_session_id returns None — no session within the 7-day window.
    engine.db_manager.get_last_session_id = MagicMock(return_value=None)

    result = engine._recover_existing_session()

    assert result is None


@pytest.mark.fast
def test_recovery_prefers_active_session_over_recent():
    """An active session always wins over recent inactive fallback."""
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=10)
    engine.db_manager.recover_last_balance = MagicMock(return_value=999.0)
    engine.db_manager.get_last_session_id = MagicMock()

    result = engine._recover_existing_session()

    assert result == 999.0
    engine.db_manager.get_last_session_id.assert_not_called()


@pytest.mark.fast
def test_fresh_start_env_var_bypasses_recovery(monkeypatch):
    """TRADING_FRESH_START=true skips all session recovery."""
    monkeypatch.setenv("TRADING_FRESH_START", "true")
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=10)
    engine.db_manager.recover_last_balance = MagicMock(return_value=999.0)

    result = engine._recover_existing_session()

    assert result is None
    engine.db_manager.get_active_session_id.assert_not_called()


@pytest.mark.fast
def test_recovery_returns_none_when_session_found_but_no_balance():
    """A session exists but recover_last_balance returns None → start fresh."""
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=10)
    engine.db_manager.recover_last_balance = MagicMock(return_value=None)

    result = engine._recover_existing_session()

    assert result is None


@pytest.mark.fast
def test_paper_mode_stop_preserves_open_positions():
    """In paper trading, stop() must NOT force-close positions."""
    engine = make_engine(enable_live_trading=False)
    engine.is_running = True  # simulate a running engine
    engine._execute_exit = MagicMock()  # inject BEFORE stop()
    mock_pos = MagicMock()
    mock_pos.symbol = "BTCUSDT"
    mock_pos.order_id = "paper_order_1"
    # Use the public API to inject the position so _position_db_ids stays consistent.
    engine.live_position_tracker.track_recovered_position(mock_pos, db_id=None)

    engine.stop()

    engine._execute_exit.assert_not_called()


@pytest.mark.fast
def test_live_mode_stop_closes_positions():
    """In live trading, stop() must close all positions."""
    engine = make_engine(enable_live_trading=True)
    engine.is_running = True  # simulate a running engine
    mock_pos = MagicMock()
    mock_pos.symbol = "BTCUSDT"
    mock_pos.order_id = "live_order_1"
    # Use the public API to inject the position so _position_db_ids stays consistent.
    engine.live_position_tracker.track_recovered_position(mock_pos, db_id=None)
    engine.data_provider.get_current_price = MagicMock(return_value=90000.0)
    engine._execute_exit = MagicMock()

    engine.stop()

    engine._execute_exit.assert_called_once_with(
        mock_pos, "Engine shutdown", None, 90000.0, None, None, None
    )


@pytest.mark.fast
def test_live_mode_stop_with_invalid_price_skips_execute_exit():
    """In live mode, stop() logs critical but does NOT call _execute_exit if price is None."""
    engine = make_engine(enable_live_trading=True)
    engine.is_running = True  # simulate a running engine
    mock_pos = MagicMock()
    mock_pos.symbol = "BTCUSDT"
    mock_pos.order_id = "live_order_1"
    # Use the public API to inject the position so _position_db_ids stays consistent.
    engine.live_position_tracker.track_recovered_position(mock_pos, db_id=None)
    engine.data_provider.get_current_price = MagicMock(return_value=None)
    engine._execute_exit = MagicMock()

    engine.stop()

    engine._execute_exit.assert_not_called()


# ---------------------------------------------------------------------------
# Regression tests for session recovery window extension (24h → 7 days)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_recovers_session_from_47_hours_ago():
    """Regression test: sessions within 7 days MUST be recovered.

    This reproduces the bug from Session #107 which ran for ~47 hours
    before crashing. The 24-hour window incorrectly rejected this session,
    causing the balance and unrealized PnL to be lost.
    """
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    # Simulate a session from 47 hours ago being found
    engine.db_manager.get_last_session_id = MagicMock(return_value=107)
    engine.db_manager.recover_last_balance = MagicMock(return_value=999.89)

    result = engine._recover_existing_session()

    assert result == 999.89
    # Verify the 7-day window was used (within_hours=168)
    engine.db_manager.get_last_session_id.assert_called_once_with(
        within_hours=168,
        strategy_name=engine._strategy_name(),
        symbol="BTCUSDT",
    )
    # Session ID must NOT be set — inactive sessions create a new session
    assert engine.trading_session_id is None


@pytest.mark.fast
def test_recovers_session_from_6_days_ago():
    """Edge case: sessions near the 7-day limit are still recovered."""
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    engine.db_manager.get_last_session_id = MagicMock(return_value=99)
    engine.db_manager.recover_last_balance = MagicMock(return_value=1050.00)

    result = engine._recover_existing_session()

    assert result == 1050.00


@pytest.mark.fast
def test_does_not_recover_session_from_8_days_ago():
    """Boundary test: sessions older than 7 days are NOT recovered."""
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    # get_last_session_id returns None — session outside 7-day window
    engine.db_manager.get_last_session_id = MagicMock(return_value=None)

    result = engine._recover_existing_session()

    assert result is None


@pytest.mark.fast
def test_live_mode_session_recovery_within_7_days():
    """Live trading mode must also recover sessions within 7 days.

    Losing balance recovery in live mode would be catastrophic — funds
    would be stranded. This test ensures live mode gets the same 7-day
    protection as paper trading.
    """
    engine = make_engine(enable_live_trading=True)
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    engine.db_manager.get_last_session_id = MagicMock(return_value=42)
    engine.db_manager.recover_last_balance = MagicMock(return_value=5000.00)

    result = engine._recover_existing_session()

    assert result == 5000.00
    # Verify the full 7-day window is used for live trading
    engine.db_manager.get_last_session_id.assert_called_once_with(
        within_hours=168,  # 7 days
        strategy_name=engine._strategy_name(),
        symbol="BTCUSDT",
    )


@pytest.mark.fast
def test_active_live_session_always_recovered_regardless_of_age():
    """Critical: active live sessions are recovered even if older than 7 days.

    An active session means the engine crashed while trading. For live mode,
    we MUST recover the session and continue trading to:
    1. Preserve trade attribution
    2. Avoid abandoning open positions
    3. Maintain balance continuity

    The 7-day window only applies to inactive (closed) sessions. Active
    sessions bypass this check entirely via get_active_session_id().
    """
    engine = make_engine(enable_live_trading=True)
    # Simulate an active session that's been running for 10 days
    engine.db_manager.get_active_session_id = MagicMock(return_value=123)
    engine.db_manager.recover_last_balance = MagicMock(return_value=12500.00)
    # get_last_session_id should not even be called for active sessions
    engine.db_manager.get_last_session_id = MagicMock()

    result = engine._recover_existing_session()

    assert result == 12500.00
    assert engine.trading_session_id == 123
    # Active session path bypasses the 7-day window
    engine.db_manager.get_last_session_id.assert_not_called()
