import pytest
import numpy as np
from strategies.ml_basic import MlBasic

pytestmark = pytest.mark.unit


class TestMlBasicStrategy:
    def test_ml_basic_strategy_initialization(self):
        strategy = MlBasic()
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'model_path')
        assert hasattr(strategy, 'sequence_length')
        assert hasattr(strategy, 'stop_loss_pct')
        assert hasattr(strategy, 'take_profit_pct')

    def test_ml_basic_exit_conditions(self, sample_ohlcv_data):
        strategy = MlBasic()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        current_price = df['close'].iloc[-1]
        entry_price = current_price * 0.98
        df.at[df.index[-1], 'onnx_pred'] = current_price * 1.05
        assert not strategy.check_exit_conditions(df, len(df)-1, entry_price)
        df.at[df.index[-1], 'onnx_pred'] = current_price * 0.95
        assert strategy.check_exit_conditions(df, len(df)-1, entry_price)

    def test_ml_basic_strategy_parameters(self):
        strategy = MlBasic()
        params = strategy.get_parameters()
        for key in ['name', 'model_path', 'sequence_length', 'stop_loss_pct', 'take_profit_pct']:
            assert key in params