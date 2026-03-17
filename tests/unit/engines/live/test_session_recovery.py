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
    """No active session → falls back to most-recent session within 24 hours."""
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    engine.db_manager.get_last_session_id = MagicMock(return_value=42)
    engine.db_manager.recover_last_balance = MagicMock(return_value=1234.56)

    result = engine._recover_existing_session()

    assert result == 1234.56
    engine.db_manager.get_last_session_id.assert_called_once()


@pytest.mark.fast
def test_recovery_ignores_sessions_older_than_24h():
    """Stale sessions (> 24 hours) are not recovered."""
    engine = make_engine()
    engine.db_manager.get_active_session_id = MagicMock(return_value=None)
    # get_last_session_id returns None — no session within the time window.
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
