import pytest

from strategies.base import BaseStrategy

pytestmark = pytest.mark.unit


class TestBaseStrategy:
    def test_base_strategy_is_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy("TestStrategy")

    def test_base_strategy_interface(self, mock_strategy):
        strategy = mock_strategy
        assert hasattr(strategy, "calculate_indicators")
        assert hasattr(strategy, "check_entry_conditions")
        assert hasattr(strategy, "check_exit_conditions")
        assert hasattr(strategy, "calculate_position_size")
        assert hasattr(strategy, "get_parameters")
        if hasattr(strategy, "get_trading_pair"):
            assert hasattr(strategy, "set_trading_pair")

    def test_trading_pair_management(self, mock_strategy):
        strategy = mock_strategy
        assert hasattr(strategy, "trading_pair")
        assert isinstance(strategy.trading_pair, str)
        assert strategy.trading_pair == "BTCUSDT"
