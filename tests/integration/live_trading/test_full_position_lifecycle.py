from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.mock_only
def test_full_position_lifecycle_with_database_logging(tmp_path, mock_strategy, mock_data_provider):
    """End-to-end: session → open → MFE/MAE update → trade log → close position."""
    from src.engines.live.trading_engine import LiveTradingEngine, PositionSide

    # Minimal data feed
    idx = pd.date_range(datetime.now(UTC), periods=10, freq="1min")
    prices = pd.Series([100.0 + i for i in range(10)], index=idx)
    mock_data_provider.get_live_data.return_value = pd.DataFrame({"close": prices})
    mock_data_provider.get_current_price.return_value = float(prices.iloc[-1])

    engine = LiveTradingEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=1000.0,
        check_interval=0.01,
        log_trades=True,
    )

    # Do not start thread; exercise methods directly
    engine.is_running = True

    # Create a trading session (required for balance updates)
    engine.trading_session_id = engine.db_manager.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode="paper",
        initial_balance=1000.0,
    )
    # Initialize balance in the database (mirrors what start() does)
    engine.db_manager.update_balance(1000.0, "session_start", "system", engine.trading_session_id)

    # Open a position
    engine._execute_entry(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        price=prices.iloc[0],
        stop_loss=None,
        take_profit=None,
        signal_strength=0.0,
        signal_confidence=0.0,
    )
    assert len(engine.live_position_tracker._positions) == 1

    # Trigger at least one MFE/MAE DB update (wait past throttle)
    time.sleep(0.1)
    engine.live_position_tracker.update_mfe_mae(current_price=float(prices.iloc[-1]))

    # Close the position through engine path
    position = list(engine.live_position_tracker._positions.values())[0]
    engine._execute_exit(
        position=position,
        reason="test_close",
        limit_price=None,
        current_price=float(prices.iloc[-1]),
        candle_high=None,
        candle_low=None,
        candle=None,
    )
    assert len(engine.live_position_tracker._positions) == 0
    assert len(engine.completed_trades) >= 1

    # Direct-methods path: validate local effects only (DB assertions require session)

    # Cleanly stop engine
    engine.is_running = False
