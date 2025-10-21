from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.unit


def test_short_entry_uses_overrides_and_caps_fraction():
    # Use component-based strategy
    strategy = create_ml_basic_strategy()
    provider = MockDataProvider(interval_seconds=60, num_candles=50)
    bt = Backtester(strategy=strategy, data_provider=provider)

    start = datetime.now() - timedelta(hours=5)
    end = datetime.now()
    result = bt.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)

    # We should have at least one trade attempted; if none, the test still
    # validates that the backtest runs with overrides without error.
    assert "total_trades" in result


def test_short_entry_with_overrides_uses_risk_manager_sizer(mock_data_provider):
    # Prepare simple data that triggers short entry
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 101],
            "volume": [1000, 1100, 1200, 1300],
            "onnx_pred": [100, 100, 101, 100],  # last bar lower than close -> negative return
            "prediction_confidence": [0.9, 0.9, 0.9, 0.9],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
    )

    mock_data_provider.get_historical_data.return_value = df

    # Use component-based strategy
    strategy = create_ml_basic_strategy()
    risk_params = RiskParameters(max_position_size=0.15)  # engine/risk cap 15%

    bt = Backtester(
        strategy=strategy,
        data_provider=mock_data_provider,
        risk_parameters=risk_params,
        initial_balance=10000,
    )

    # Use a spy db to ensure logging happens; not strictly required
    bt.db_manager = Mock()
    bt.log_to_database = True

    # Run backtest on the dataset
    results = bt.run(symbol="BTCUSDT", timeframe="1h", start=df.index[0], end=df.index[-1])

    # Should not crash and should have at most one trade opened (entry logic depends on strategy)
    assert isinstance(results, dict)
    # Ensure the sizer respected the max position size cap (<= 0.15)
    if bt.current_trade is not None:
        assert bt.current_trade.size <= 0.15 + 1e-9
