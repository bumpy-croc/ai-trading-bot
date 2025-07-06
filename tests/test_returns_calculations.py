import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime
from src.data_providers.data_provider import DataProvider
from src.strategies.base import BaseStrategy
from src.backtesting.engine import Backtester
from typing import Optional


class DummyDataProvider(DataProvider):
    """A lightweight data provider for unit tests that returns a pre-built DataFrame."""

    def __init__(self, df):
        super().__init__()
        self._df = df

    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None):
        # Ignore params and just return the stored frame
        return self._df.copy()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100):
        raise NotImplementedError

    def update_live_data(self, symbol: str, timeframe: str):
        raise NotImplementedError


class BuyEveryYearStrategy(BaseStrategy):
    """Enter at first candle of a year, exit at last candle of that same year."""

    def __init__(self):
        super().__init__("BuyEveryYearStrategy")
        self.in_position = False
        self.current_year = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # No indicators needed

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        ts_year = df.index[index].year
        if (not self.in_position) and (self.current_year is None or ts_year != self.current_year):
            # Ready to enter at first candle of new year
            self.current_year = ts_year
            return True
        return False

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        # Exit if it's the last candle OR next candle belongs to a different year
        ts_year = df.index[index].year
        last_index = len(df) - 1
        next_year_change = (index < last_index) and (df.index[index + 1].year != ts_year)
        if self.in_position and (index == last_index or next_year_change):
            # Position exits now; reset state
            self.in_position = False
            return True
        return False

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if balance <= 0:
            return 0.0
        # Use 100% balance each trade for clarity
        self.in_position = True
        return balance

    def calculate_stop_loss(self, df, index, price, side: str = 'long') -> float:
        return price * 0.5  # arbitrary – not hit in this test

    def get_parameters(self) -> dict:
        return {}


def generate_test_dataframe():
    # Four candles covering two years with clear up & down moves
    timestamps = [
        datetime(2020, 1, 1, 0),  # 2020 start
        datetime(2020, 12, 31, 23),  # 2020 end – price up
        datetime(2021, 1, 1, 0),  # 2021 start
        datetime(2021, 12, 31, 23),  # 2021 end – price down
    ]
    closes = [100.0, 200.0, 200.0, 100.0]
    df = pd.DataFrame({
        'open': closes,
        'high': closes,
        'low': closes,
        'close': closes,
        'volume': np.ones(len(closes)),
    }, index=pd.DatetimeIndex(timestamps))
    return df


def test_yearly_returns_positive_and_negative():
    df = generate_test_dataframe()
    provider = DummyDataProvider(df)
    strategy = BuyEveryYearStrategy()

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=None,
        initial_balance=1000,
        log_to_database=False,
    )

    result = backtester.run(
        symbol="TEST",
        timeframe="1h",
        start=df.index[0],
        end=df.index[-1],
    )

    yr = result['yearly_returns']
    assert '2020' in yr and '2021' in yr, "Yearly returns keys missing"
    assert yr['2020'] > 0, "2020 should be positive"
    assert yr['2021'] < 0, "2021 should be negative (not zero)"

    # Total return should equal compounded yearly performance
    total_return_factor = (1 + yr['2020']/100) * (1 + yr['2021']/100)
    expected_total = (total_return_factor - 1) * 100
    assert abs(expected_total - result['total_return']) < 1e-6, "Total return mismatch"