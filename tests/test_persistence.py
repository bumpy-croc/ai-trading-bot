import os
from pathlib import Path
from datetime import datetime
import pytest

from database.manager import DatabaseManager
from database.models import PositionSide, TradeSource


def _create_db_manager(db_file: Path) -> DatabaseManager:
    """Utility to get a DatabaseManager bound to a temporary SQLite file"""
    db_url = f"sqlite:///{db_file}"
    # Ensure parent directory exists
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return DatabaseManager(db_url)


@pytest.fixture()
def temp_db(tmp_path) -> DatabaseManager:
    """Create a fresh database for each test in a temporary directory"""
    db_file = tmp_path / "persistent_test.db"
    return _create_db_manager(db_file)


def test_balance_and_position_persistence(temp_db: DatabaseManager, tmp_path):
    """Ensure balance & positions persist after simulated restart."""
    # 1️⃣  START TRADING SESSION & CREATE STATE
    initial_balance = 1000.0
    session_id = temp_db.create_trading_session(
        strategy_name="persistence_test_strategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=initial_balance,
    )

    # Set initial balance record
    assert temp_db.update_balance(initial_balance, "initial_balance", "test", session_id)

    # Log a position
    position_id = temp_db.log_position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=45000.0,
        size=0.1,  # 10 % of balance
        strategy_name="persistence_test_strategy",
        order_id="unit_test_order",
        stop_loss=44000.0,
        take_profit=46000.0,
        session_id=session_id,
    )
    assert position_id is not None

    # Verify state exists before restart
    pre_restart_balance = temp_db.get_current_balance(session_id)
    assert pre_restart_balance == initial_balance
    pre_positions = temp_db.get_active_positions(session_id)
    assert len(pre_positions) == 1

    # 2️⃣  SIMULATE RESTART (new DatabaseManager instance bound to same DB file)
    # We need to point to the same sqlite file used in fixture
    db_file = tmp_path / "persistent_test.db"
    assert db_file.exists(), "Database file should exist after initial operations"
    new_db_manager = _create_db_manager(db_file)

    # 3️⃣  RECOVER STATE IN NEW INSTANCE
    recovered_balance = new_db_manager.recover_last_balance(session_id)
    assert recovered_balance == pre_restart_balance, "Recovered balance mismatch"

    recovered_positions = new_db_manager.get_active_positions(session_id)
    assert len(recovered_positions) == 1, "Positions did not persist after restart"
    recovered_pos = recovered_positions[0]
    assert recovered_pos["symbol"] == "BTCUSDT"
    assert recovered_pos["side"] == "long"
    assert recovered_pos["entry_price"] == 45000.0


@pytest.mark.parametrize("adjustment", [500, -200])
def test_manual_balance_adjustment_persists(temp_db: DatabaseManager, tmp_path, adjustment):
    """Manual balance adjustments should persist across restarts."""
    # Create session and set balance
    start_balance = 1000.0
    session_id = temp_db.create_trading_session(
        strategy_name="manual_adjust_test",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=start_balance,
    )
    temp_db.update_balance(start_balance, "initial_balance", "test", session_id)

    # Manual adjustment
    new_balance = start_balance + adjustment
    assert temp_db.manual_balance_adjustment(new_balance, f"test_adjust_{adjustment}", "unit_test")

    # Restart simulation
    db_file = tmp_path / "persistent_test.db"
    new_db = _create_db_manager(db_file)

    recovered_balance = new_db.get_current_balance(session_id)
    assert recovered_balance == new_balance, "Manual adjustment did not persist after restart"