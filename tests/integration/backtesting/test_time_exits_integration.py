from datetime import datetime, timedelta, time

import pandas as pd

from src.backtesting.engine import Backtester
from src.position_management.time_exits import TimeExitPolicy, MarketSessionDef
from src.strategies.ml_basic import MlBasic


def test_backtester_time_exit_max_holding(mock_data_provider):
    # Build simple hourly data where strategy could enter/hold
    idx = pd.date_range("2024-01-01 00:00:00", periods=48, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1000] * len(idx),
            # add a basic prediction_confidence to allow sizing
            "prediction_confidence": [0.9] * len(idx),
        },
        index=idx,
    )
    mock_data_provider.get_historical_data.return_value = df

    strategy = MlBasic()

    # Exit after 1 hour
    policy = TimeExitPolicy(max_holding_hours=1)
    bt = Backtester(strategy=strategy, data_provider=mock_data_provider, time_exit_policy=policy, enable_time_limit_exit=True)

    results = bt.run("BTCUSDT", "1h", start=idx[0].to_pydatetime(), end=idx[-1].to_pydatetime())

    assert isinstance(results, dict)
    # Should not accumulate a very long open trade due to time exit
    # We can't assert exact counts because strategy logic drives entries, but ensure it runs
    assert "final_balance" in results


def test_backtester_time_exit_end_of_day(mock_data_provider):
    # Create a session that closes at 21:00 UTC, Mon-Fri
    session = MarketSessionDef(
        name="US_EQUITIES",
        timezone="UTC",
        open_time=time(14, 30),
        close_time=time(21, 0),
        days_of_week=[1, 2, 3, 4, 5],
        is_24h=False,
    )
    policy = TimeExitPolicy(end_of_day_flat=True, market_session=session)

    # 2 days of hourly data
    idx = pd.date_range("2024-01-02 14:00:00", periods=16, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1000] * len(idx),
            "prediction_confidence": [0.9] * len(idx),
        },
        index=idx,
    )
    mock_data_provider.get_historical_data.return_value = df

    strategy = MlBasic()
    bt = Backtester(strategy=strategy, data_provider=mock_data_provider, time_exit_policy=policy, enable_time_limit_exit=True)

    results = bt.run("BTCUSDT", "1h", start=idx[0].to_pydatetime(), end=idx[-1].to_pydatetime())
    assert isinstance(results, dict)
