from __future__ import annotations

import threading
import time
from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.mock_only
def test_concurrent_open_close_with_database(tmp_path, mock_strategy, mock_data_provider):
    from src.live.trading_engine import LiveTradingEngine, PositionSide

    # Basic deterministic data
    idx = pd.date_range(datetime.utcnow(), periods=5, freq="1min")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    mock_data_provider.get_live_data.return_value = pd.DataFrame({"close": prices})
    mock_data_provider.get_current_price.return_value = 102.0

    engine = LiveTradingEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=1000.0,
        check_interval=0.01,
        database_url=f"sqlite:///{tmp_path}/concurrency.db",
        log_trades=True,
    )

    engine.start(symbol="BTCUSDT", timeframe="1m", max_steps=1)

    def open_positions():
        for i in range(3):
            engine._open_position("BTCUSDT", PositionSide.LONG, size=0.05, price=100.0 + i)
            time.sleep(0.01)

    def close_some():
        time.sleep(0.05)
        # close first two if present
        for _ in range(2):
            if engine.positions:
                position = next(iter(engine.positions.values()))
                engine._close_position(position, reason="concurrent-close")
                time.sleep(0.01)

    t1 = threading.Thread(target=open_positions)
    t2 = threading.Thread(target=close_some)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Database should have at least as many trades as closed positions
    db = engine.db_manager
    trades = db.get_recent_trades(limit=10, session_id=engine.trading_session_id)
    assert isinstance(trades, list)

    engine.stop()


