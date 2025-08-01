import pandas as pd
from datetime import datetime, timedelta

from backtesting.engine import Backtester
from strategies.ml_basic import MlBasic
from data_providers.mock_data_provider import MockDataProvider


def test_backtester_initialization_and_run(mock_data_provider):
    """Smoke-test Backtester with MlBasic strategy on mock data."""
    strategy = MlBasic()
    backtester = Backtester(
        strategy=strategy,
        data_provider=mock_data_provider,
        initial_balance=10_000,
        log_to_database=False,
    )

    start = datetime.now() - timedelta(days=1)
    end = datetime.now()

    results = backtester.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)

    # Assert minimal keys exist in results
    assert "total_trades" in results
    assert "final_balance" in results
    assert backtester.balance >= 0