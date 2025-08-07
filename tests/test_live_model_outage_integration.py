import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.live.trading_engine import LiveTradingEngine, PositionSide
from src.strategies.ml_basic import MlBasic
from src.data_providers.mock_data_provider import MockDataProvider


@pytest.mark.integration
def test_live_engine_applies_model_outage_policy(monkeypatch):
    # Strategy with no prediction engine (model outage scenario)
    strategy = MlBasic(prediction_engine=None, sequence_length=5)

    # Mock provider with stable data and ATR column
    provider = MockDataProvider()
    df = pd.DataFrame({
        'open': np.linspace(100, 100, 50),
        'high': np.linspace(100.5, 100.5, 50),
        'low': np.linspace(99.5, 99.5, 50),
        'close': np.linspace(100, 100, 50),
        'volume': np.full(50, 1000.0),
        'atr': np.full(50, 1.0),
    }, index=pd.date_range('2024-01-01', periods=50, freq='1h'))

    provider.get_live_data = Mock(return_value=df)

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        check_interval=1,
        enable_live_trading=False,
        log_trades=False,
        resume_from_last_balance=False,
        database_url=None,
    )

    # Open a paper position manually to simulate existing trade
    engine.current_balance = 10000.0
    price = 100.0
    engine._open_position('BTCUSDT', PositionSide.LONG, 0.1, price, price * 0.98, price * 1.04)
    # Force position to be aged beyond time-stop for outage policy
    for pos in engine.positions.values():
        pos.entry_time = datetime.now() - timedelta(hours=48)

    # Run one loop iteration parts: fetch df and check exits only
    latest_df = engine._get_latest_data('BTCUSDT', '1h')
    assert latest_df is not None
    latest_df = strategy.calculate_indicators(latest_df)
    # Ensure we have rows after warmup columns by forward-filling simple fallbacks
    latest_df = latest_df.ffill()
    current_index = len(latest_df) - 1
    current_price = float(latest_df['close'].iloc[current_index])

    # Apply exit checks; should trigger model outage time-based exit
    engine._update_position_pnl(current_price)
    engine._check_exit_conditions(latest_df, current_index, current_price)

    # Expect that positions are closed by the outage policy
    assert len(engine.positions) == 0


