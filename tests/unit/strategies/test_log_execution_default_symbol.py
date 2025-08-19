from src.strategies.base import BaseStrategy


class DummyDB:
    def __init__(self):
        self.last = None

    def log_strategy_execution(self, **kwargs):
        self.last = kwargs


class S(BaseStrategy):
    def __init__(self):
        super().__init__("S")

    def calculate_indicators(self, df):
        return df

    def check_entry_conditions(self, df, index):
        return False

    def check_exit_conditions(self, df, index, entry_price):
        return False

    def calculate_position_size(self, df, index, balance):
        return 0.0

    def calculate_stop_loss(self, df, index, price, side="long"):
        return price

    def get_parameters(self):
        return {}


def test_log_execution_defaults_to_strategy_pair_when_symbol_missing():
    s = S()
    s.trading_pair = "ETHUSDT"
    db = DummyDB()
    s.set_database_manager(db)
    s.log_execution(signal_type="entry", action_taken="none", price=1000.0)
    assert db.last is not None
    assert db.last["symbol"].upper() == "ETHUSDT"


from unittest.mock import Mock

import pytest

from strategies.base import BaseStrategy

pytestmark = pytest.mark.unit


class _DummyStrategy(BaseStrategy):
    def calculate_indicators(self, df):
        return df

    def check_entry_conditions(self, df, index: int) -> bool:
        return False

    def check_exit_conditions(self, df, index: int, entry_price: float) -> bool:
        return False

    def calculate_position_size(self, df, index: int, balance: float) -> float:
        return 0.0

    def calculate_stop_loss(self, df, index: int, price: float, side: str = "long") -> float:
        return price * (0.99 if side == "long" else 1.01)

    def get_parameters(self) -> dict:
        return {}


def test_log_execution_defaults_symbol_to_trading_pair():
    strategy = _DummyStrategy("dummy")
    # Ensure default trading_pair
    assert strategy.trading_pair == "BTCUSDT"

    mock_db = Mock()
    strategy.set_database_manager(mock_db, session_id=123)

    # Call without providing symbol -> should default to strategy.trading_pair
    strategy.log_execution(
        signal_type="entry",
        action_taken="no_action",
        price=100.0,
        symbol=None,
        timeframe="1m",
        reasons=["unit_test"],
    )

    mock_db.log_strategy_execution.assert_called()
    _, kwargs = mock_db.log_strategy_execution.call_args
    # Symbol should be Binance-normalized BTCUSDT
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["strategy_name"] == strategy.__class__.__name__
