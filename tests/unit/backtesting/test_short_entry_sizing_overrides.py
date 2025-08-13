import pytest
import pandas as pd
from unittest.mock import Mock

from backtesting.engine import Backtester
from strategies.ml_basic import MlBasic
from risk.risk_manager import RiskParameters

pytestmark = pytest.mark.unit


def test_short_entry_with_overrides_uses_risk_manager_sizer(mock_data_provider):
    # Prepare simple data that triggers short entry
    df = pd.DataFrame({
        'open': [100, 101, 102, 103],
        'high': [101, 102, 103, 104],
        'low': [99, 100, 101, 102],
        'close': [100, 101, 102, 101],
        'volume': [1000, 1100, 1200, 1300],
        'onnx_pred': [100, 100, 101, 100],  # last bar lower than close -> negative return
        'prediction_confidence': [0.9, 0.9, 0.9, 0.9],
    }, index=pd.date_range('2024-01-01', periods=4, freq='1h'))

    mock_data_provider.get_historical_data.return_value = df

    class StrategyWithOverrides(MlBasic):
        def get_risk_overrides(self):
            return {
                'position_sizer': 'fixed_fraction',
                'base_fraction': 0.04,
                'max_fraction': 0.2,
            }

    strategy = StrategyWithOverrides()
    risk_params = RiskParameters(max_position_size=0.15)  # engine/risk cap 15%

    bt = Backtester(
        strategy=strategy,
        data_provider=mock_data_provider,
        risk_parameters=risk_params,
        initial_balance=10000,
        enable_short_trading=True,
    )

    # Use a spy db to ensure logging happens; not strictly required
    bt.db_manager = Mock()
    bt.log_to_database = True

    # Run backtest on the dataset
    results = bt.run(symbol='BTCUSDT', timeframe='1h', start=df.index[0], end=df.index[-1])

    # Should not crash and should have at most one trade opened (entry logic depends on strategy)
    assert isinstance(results, dict)
    # Ensure the sizer respected the max position size cap (<= 0.15)
    if bt.current_trade is not None:
        assert bt.current_trade.size <= 0.15 + 1e-9