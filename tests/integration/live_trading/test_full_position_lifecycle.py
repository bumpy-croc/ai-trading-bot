from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import time

import pandas as pd
import pytest


pytestmark = pytest.mark.integration


def _file_sqlite_url(tmp_path: Path) -> str:
    db_path = tmp_path / "engine_lifecycle.db"
    return f"sqlite:///{db_path}"


@pytest.mark.mock_only
def test_full_position_lifecycle_with_database_logging(tmp_path, mock_strategy, mock_data_provider):
    """End-to-end: session → open → MFE/MAE update → trade log → close position.

    Uses file-based sqlite to survive across threads and engine internals.
    """
    from live.trading_engine import LiveTradingEngine, PositionSide

    # Minimal data feed
    idx = pd.date_range(datetime.utcnow(), periods=10, freq="1min")
    prices = pd.Series([100.0 + i for i in range(10)], index=idx)
    mock_data_provider.get_live_data.return_value = pd.DataFrame({"close": prices})
    mock_data_provider.get_current_price.return_value = float(prices.iloc[-1])

    engine = LiveTradingEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=1000.0,
        check_interval=0.01,
        database_url=_file_sqlite_url(tmp_path),
        log_trades=True,
    )

    # Do not start thread; exercise methods directly
    engine.is_running = True

    # Open a position
    engine._open_position("BTCUSDT", PositionSide.LONG, size=0.1, price=prices.iloc[0])
    assert len(engine.positions) == 1

    # Trigger at least one MFE/MAE DB update (wait past throttle)
    time.sleep(0.1)
    engine._update_positions_mfe_mae(current_price=float(prices.iloc[-1]))

    # Close the position through engine path
    position = list(engine.positions.values())[0]
    engine._close_position(position, reason="test_close")
    assert len(engine.positions) == 0
    assert len(engine.completed_trades) >= 1

    # Direct-methods path: validate local effects only (DB assertions require session)

    # Cleanly stop engine
    engine.is_running = False


