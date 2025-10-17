
import pandas as pd
import pytest

from src.data_providers.mock_data_provider import MockDataProvider
from src.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy


@pytest.mark.integration
@pytest.mark.mock_only
def test_live_engine_records_mfe_mae():
    # Mock data provider with fast updates
    provider = MockDataProvider(interval_seconds=1, num_candles=50)

    strategy = create_ml_basic_strategy()

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        check_interval=0.01,  # fast loop
        initial_balance=1000.0,
        enable_live_trading=False,
        log_trades=True,
        account_snapshot_interval=0,  # disable snapshots for speed
        resume_from_last_balance=False,  # disable session recovery for test isolation
        # Use default PostgreSQL test container via DatabaseManager
        database_url=None,
        enable_dynamic_risk=False,
        enable_hot_swapping=False,
    )

    # Run a short session
    engine.start(symbol="BTCUSDT", timeframe="1h", max_steps=8)

    # Verify at least one trade with MFE/MAE recorded
    assert engine.trading_session_id is not None
    trades = engine.db_manager.get_recent_trades(limit=5, session_id=engine.trading_session_id)
    assert isinstance(trades, list)
    assert len(trades) >= 1

    t0 = trades[0]
    # MFE/MAE fields should be present and numeric (decimals returned by DB manager are acceptable)
    assert "mfe" in t0 and "mae" in t0
    mfe = float(t0.get("mfe") or 0.0)
    mae = float(t0.get("mae") or 0.0)
    # Expect sign consistency: MFE >= 0, MAE <= 0
    assert mfe >= 0.0
    assert mae <= 0.0