"""Unit tests for LiveStartupSequencer (#486).

The engine bootstrap sequence was extracted from ``LiveTradingEngine.start`` into
``LiveStartupSequencer`` (the public ``start`` delegates to ``run``). These tests
drive individual phases directly with a mocked engine-state backref; the full
end-to-end bootstrap is covered by the live-engine integration tests that call
``engine.start()``.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, create_autospec

import pytest

from src.engines.live.startup import LiveStartupEngineState, LiveStartupSequencer

pytestmark = pytest.mark.fast


def _make_state() -> object:
    state = create_autospec(LiveStartupEngineState, instance=True)
    # Protocol attributes are annotation-only, so assign the ones the phases read.
    state.is_running = False
    state.enable_live_trading = False
    state.current_balance = 1000.0
    state.max_position_size = 1.0
    state.check_interval = 60
    state.trading_session_id = None
    state.strategy = MagicMock()
    # Annotation-only Protocol attributes aren't auto-created by autospec.
    state.db_manager = MagicMock()
    state.strategy_manager = None
    return state


def test_begin_session_runtime_sets_flags():
    state = _make_state()
    LiveStartupSequencer(engine_state=state).begin_session_runtime("BTCUSDT", "1h")
    assert state.is_running is True
    assert state._active_symbol == "BTCUSDT"
    assert state.timeframe == "1h"


def test_begin_session_runtime_threads_symbol_to_strategy_manager():
    """Hot-swapped strategies must select models for the traded pair (#867)."""
    state = _make_state()
    state.strategy_manager = MagicMock()
    LiveStartupSequencer(engine_state=state).begin_session_runtime("ETHUSDT", "1h")
    assert state.strategy_manager.symbol == "ETHUSDT"


def test_begin_session_runtime_tolerates_missing_strategy_manager():
    """Engines constructed without hot-swapping keep working."""
    state = _make_state()
    state.strategy_manager = None
    LiveStartupSequencer(engine_state=state).begin_session_runtime("ETHUSDT", "1h")
    assert state._active_symbol == "ETHUSDT"


def test_run_returns_early_when_already_running():
    state = _make_state()
    state.is_running = True
    LiveStartupSequencer(engine_state=state).run("BTCUSDT", timeframe="1h")
    # Guard short-circuits before any bootstrap work.
    state.db_manager.create_trading_session.assert_not_called()
    state._recover_existing_session.assert_not_called()


def test_self_heal_calls_db_heal_when_session_present():
    state = _make_state()
    state.trading_session_id = 5
    state.db_manager.heal_positions_with_terminal_trades.return_value = 0
    LiveStartupSequencer(engine_state=state).self_heal_terminal_positions()
    state.db_manager.heal_positions_with_terminal_trades.assert_called_once_with(5)


def test_self_heal_noop_without_session():
    state = _make_state()
    state.trading_session_id = None
    LiveStartupSequencer(engine_state=state).self_heal_terminal_positions()
    state.db_manager.heal_positions_with_terminal_trades.assert_not_called()


def test_synchronize_account_persists_balance_correction():
    """An exchange balance correction is applied in memory AND persisted to the DB.

    Regression guard: the persist block reads the pending-correction flag off the
    engine backref (`state`), not the sequencer (`self`) — a `self`/`state` mixup
    here silently drops the DB write on the capital-critical account-sync path.
    """
    state = _make_state()
    state.enable_live_trading = True
    state.trading_session_id = 7
    state._active_symbol = "BTCUSDT"
    state._balance_lock = threading.Lock()  # real CM; annotation-only on the protocol
    state.account_synchronizer = MagicMock()
    sync_result = MagicMock()
    sync_result.success = True
    sync_result.data = {
        "balance_sync": {"corrected": True, "old_balance": 1000.0, "new_balance": 1234.5}
    }
    state.account_synchronizer.sync_account_data.return_value = sync_result

    LiveStartupSequencer(engine_state=state).synchronize_account_on_start()

    assert state.current_balance == 1234.5
    state.db_manager.update_balance.assert_called_once_with(1234.5, "account_sync", "system", 7)


def test_synchronize_account_persists_post_reconcile_balance():
    """The persisted balance must be the POST-reconcile in-memory value.

    Reconciliation (step b, between the exchange sync and the DB persist) may
    legitimately adjust ``current_balance``; persisting the pre-reconcile
    snapshot writes a stale value that diverges the DB row from memory — the
    next restart's ``recover_last_balance`` then reads the wrong balance
    (audit 2026-07-12 F2)."""
    state = _make_state()
    state.enable_live_trading = True
    state.trading_session_id = 7
    state._active_symbol = "BTCUSDT"
    state._balance_lock = threading.Lock()  # real CM; annotation-only on the protocol
    state.account_synchronizer = MagicMock()
    sync_result = MagicMock()
    sync_result.success = True
    sync_result.data = {
        "balance_sync": {"corrected": True, "old_balance": 1000.0, "new_balance": 1234.5}
    }
    state.account_synchronizer.sync_account_data.return_value = sync_result

    def _reconcile_adjusts_balance():
        state.current_balance = 1200.0

    state._reconcile_positions_with_exchange.side_effect = _reconcile_adjusts_balance

    LiveStartupSequencer(engine_state=state).synchronize_account_on_start()

    state.db_manager.update_balance.assert_called_once_with(1200.0, "account_sync", "system", 7)
