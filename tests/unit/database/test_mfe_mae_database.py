from datetime import UTC, datetime

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
    db.log_position(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        size=0.1,
        strategy_name="TestStrategy",
        entry_order_id="order-1",
        session_id=session_id,
        mfe=0.02,
        mae=-0.01,
    )
    positions = db.get_active_positions(session_id=session_id)
    assert positions
    assert float(positions[0]["mfe"]) == pytest.approx(0.02)
    assert float(positions[0]["mae"]) == pytest.approx(-0.01)


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
        entry_order_id="order-2",
        session_id=session_id,
    )
    now = datetime.now(UTC)
    db.update_position(
        position_id=pos_id,
        current_price=195.0,
        mfe=0.025,
        mae=-0.01,
        mfe_price=195.0,
        mae_price=202.0,
        mfe_time=now,
        mae_time=now,
    )
    positions = db.get_active_positions(session_id=session_id)
    # Readers derive mfe/mae from the price companions (short, entry 200):
    # mfe = (200 - 195) / 200, mae = (200 - 202) / 200.
    assert float(positions[0]["mfe"]) == pytest.approx(0.025)
    assert float(positions[0]["mae"]) == pytest.approx(-0.01)


def test_readers_derive_excursion_from_price_companions_for_pre_fix_rows():
    """Rows written by the pre-fix tracker stored sized, fee-netted mfe/mae.

    Readers must return the excursion implied by the _price companions (always
    raw extreme prices) instead of trusting the corrupted stored column.
    Values from prod trade 9 (#966): stored mfe 0.00320157 vs implied 3.265%.
    """
    db = DatabaseManager("sqlite:///:memory:")
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="ETHUSDT",
        timeframe="1h",
        mode=TradeSource.BACKTEST,
        initial_balance=1000.0,
    )
    entry = 1673.21
    db.log_trade(
        symbol="ETHUSDT",
        side="long",
        entry_price=entry,
        exit_price=1731.44,
        size=0.144,
        entry_time=datetime.now(UTC),
        exit_time=datetime.now(UTC),
        pnl=5.0,
        exit_reason="Stop loss",
        strategy_name="TestStrategy",
        source=TradeSource.BACKTEST,
        session_id=session_id,
        mfe=0.00320157,  # sized, fee-netted (pre-fix writer)
        mae=-0.00736083,
        mfe_price=1727.84,
        mae_price=1605.11,
    )
    trades = db.get_recent_trades(limit=5, session_id=session_id)
    assert trades
    assert float(trades[0]["mfe"]) == pytest.approx((1727.84 - entry) / entry)
    assert float(trades[0]["mae"]) == pytest.approx((1605.11 - entry) / entry)


def test_excursion_from_price_helper():
    # Long: favorable above entry
    assert DatabaseManager._excursion_from_price(100.0, 110.0, "long") == pytest.approx(0.10)
    # Short: favorable below entry
    assert DatabaseManager._excursion_from_price(100.0, 90.0, "SHORT") == pytest.approx(0.10)
    # Underivable inputs fall through to None
    assert DatabaseManager._excursion_from_price(None, 110.0, "long") is None
    assert DatabaseManager._excursion_from_price(100.0, None, "long") is None
    assert DatabaseManager._excursion_from_price(0.0, 110.0, "long") is None
    assert DatabaseManager._excursion_from_price(100.0, 110.0, "sideways") is None
    assert DatabaseManager._excursion_from_price(100.0, 110.0, None) is None


def test_excursion_or_stored_falls_back_to_stored_column():
    # No companion price: stored value is returned as-is
    assert DatabaseManager.excursion_or_stored(100.0, None, 0.05, "long") == pytest.approx(0.05)
    # Companion present: derived value wins over an inconsistent stored one
    assert DatabaseManager.excursion_or_stored(100.0, 110.0, 0.001, "long") == pytest.approx(0.10)
    # Neither derivable nor stored: default 0.0
    assert DatabaseManager.excursion_or_stored(100.0, None, None, "long") == 0.0


def test_log_trade_with_mfe_mae():
    db = DatabaseManager("sqlite:///:memory:")
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.BACKTEST,
        initial_balance=1000.0,
    )
    db.log_trade(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        exit_price=110.0,
        size=0.1,
        entry_time=datetime.now(UTC),
        exit_time=datetime.now(UTC),
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
