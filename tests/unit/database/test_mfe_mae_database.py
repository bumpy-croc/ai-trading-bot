from datetime import datetime

import pytest
from src.database.manager import DatabaseManager
from src.database.models import TradeSource


def test_log_position_with_mfe_mae(sqlite_memory_db_url: str = "sqlite:///:memory:"):
    db = DatabaseManager(sqlite_memory_db_url)
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.BACKTEST,
        initial_balance=1000.0,
    )
    pos_id = db.log_position(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        size=0.1,
        strategy_name="TestStrategy",
        order_id="order-1",
        session_id=session_id,
        mfe=0.02,
        mae=-0.01,
    )
    positions = db.get_active_positions(session_id=session_id)
    assert positions
    assert float(positions[0]["mfe"]) == pytest.approx(0.02)
    assert float(positions[0]["mae"]) == pytest.approx(-0.01)


def test_update_position_mfe_mae(sqlite_memory_db_url: str = "sqlite:///:memory:"):
def test_update_position_mfe_mae():
    db = DatabaseManager("sqlite:///:memory:")
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.BACKTEST,
        initial_balance=1000.0,
    )
    pos_id = db.log_position(
        symbol="BTCUSDT",
        side="short",
        entry_price=200.0,
        size=0.2,
        strategy_name="TestStrategy",
        order_id="order-2",
        session_id=session_id,
    )
    now = datetime.utcnow()
    db.update_position(
        position_id=pos_id,
        current_price=195.0,
        mfe=0.01,
        mae=-0.005,
        mfe_price=195.0,
        mae_price=202.0,
        mfe_time=now,
        mae_time=now,
    )
    positions = db.get_active_positions(session_id=session_id)
    assert float(positions[0]["mfe"]) == pytest.approx(0.01)
    assert float(positions[0]["mae"]) == pytest.approx(-0.005)


def test_log_trade_with_mfe_mae(sqlite_memory_db_url: str = "sqlite:///:memory:"):
def test_log_trade_with_mfe_mae():
    db = DatabaseManager("sqlite:///:memory:")
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.BACKTEST,
        initial_balance=1000.0,
    )
    trade_id = db.log_trade(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        exit_price=110.0,
        size=0.1,
        entry_time=datetime.utcnow(),
        exit_time=datetime.utcnow(),
        pnl=10.0,
        exit_reason="test",
        strategy_name="TestStrategy",
        source=TradeSource.BACKTEST,
        session_id=session_id,
        mfe=0.03,
        mae=-0.01,
    )
    trades = db.get_recent_trades(limit=5, session_id=session_id)
    assert trades
    assert float(trades[0]["mfe"]) == pytest.approx(0.03)
    assert float(trades[0]["mae"]) == pytest.approx(-0.01)