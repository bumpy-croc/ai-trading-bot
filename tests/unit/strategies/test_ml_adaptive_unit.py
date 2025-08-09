import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

from strategies.ml_adaptive import MlAdaptive

pytestmark = pytest.mark.unit


class TestMlAdaptiveBasics:
    def test_adaptive_strategy_initialization(self):
        strategy = MlAdaptive()
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTC-USD') is not None
        assert hasattr(strategy, 'base_stop_loss_pct')
        assert hasattr(strategy, 'base_take_profit_pct')
        assert hasattr(strategy, 'base_position_size')

    def test_indicator_calculation(self, sample_ohlcv_data):
        strategy = MlAdaptive()
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        original_columns = set(sample_ohlcv_data.columns)
        new_columns = set(df_with_indicators.columns)
        added_columns = new_columns - original_columns
        assert len(added_columns) > 0

    def test_atr_or_presence(self, sample_ohlcv_data):
        strategy = MlAdaptive()
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        atr_columns = [col for col in df_with_indicators.columns if 'atr' in col.lower()]
        assert len(atr_columns) > 0

    def test_strategy_parameters(self):
        strategy = MlAdaptive()
        params = strategy.get_parameters()
        assert isinstance(params, dict)
        if 'base_risk_per_trade' in params:
            assert 0 < params['base_risk_per_trade'] <= 0.1


class TestMlAdaptiveEdgeCases:
    def test_insufficient_data_handling(self):
        strategy = MlAdaptive()
        minimal_data = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49800, 49900],
            'close': [50100, 50200],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1h'))
        try:
            df = strategy.calculate_indicators(minimal_data)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == len(minimal_data)
        except Exception as e:
            assert any(k in str(e).lower() for k in ['insufficient', 'data', 'length', 'period'])

    def test_entry_conditions_oob(self, sample_ohlcv_data):
        strategy = MlAdaptive()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        assert strategy.check_entry_conditions(df, -1) is False
        assert strategy.check_entry_conditions(df, len(df) + 10) is False
        assert strategy.check_entry_conditions(df, 0) is False

    def test_position_size_edge_cases(self, sample_ohlcv_data):
        strategy = MlAdaptive()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        if len(df) > 0:
            assert strategy.calculate_position_size(df, len(df)-1, 0) == 0
            small = strategy.calculate_position_size(df, len(df)-1, 1)
            assert 0 <= small <= 1

    def test_missing_indicator_data(self):
        strategy = MlAdaptive()
        problematic = pd.DataFrame({
            'open': [50000, np.nan, 50200],
            'high': [50200, 50300, np.nan],
            'low': [49800, 49900, 50000],
            'close': [50100, 50200, 50300],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))
        try:
            df = strategy.calculate_indicators(problematic)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == len(problematic)
        except Exception as e:
            assert any(k in str(e).lower() for k in ['nan', 'missing', 'invalid', 'data'])