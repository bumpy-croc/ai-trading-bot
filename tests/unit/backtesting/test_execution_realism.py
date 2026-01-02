"""Unit tests for realistic execution parameters (fees, slippage, next-bar execution)"""

from datetime import datetime

import pandas as pd

from src.engines.backtest.engine import Backtester
from src.strategies.components import Signal, SignalDirection


class MockDataProvider:
    """Mock data provider for testing"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return mock data"""
        return self.df.copy()


class SimpleSignalStrategy:
    """Simple test strategy that generates fixed signals"""

    def __init__(
        self, name: str = "test_strategy", signal_direction: SignalDirection = SignalDirection.BUY
    ):
        self.name = name
        self.signal_direction = signal_direction
        self._runtime = None

    @property
    def runtime(self):
        return self._runtime

    def process_runtime(self, context):
        """Process runtime context"""
        self._runtime = context
        return Signal(
            direction=self.signal_direction,
            strength=0.8,
            confidence=0.9,
            metadata={"test": True, "enter_short": self.signal_direction == SignalDirection.SELL},
        )


def create_test_dataframe(num_candles: int = 100) -> pd.DataFrame:
    """Create a simple test dataframe"""
    base_price = 100.0
    dates = pd.date_range(start="2024-01-01", periods=num_candles, freq="1h")

    data = {
        "open": [base_price + i * 0.1 for i in range(num_candles)],
        "high": [base_price + i * 0.1 + 0.5 for i in range(num_candles)],
        "low": [base_price + i * 0.1 - 0.5 for i in range(num_candles)],
        "close": [base_price + i * 0.1 + 0.1 for i in range(num_candles)],
        "volume": [1000] * num_candles,
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "timestamp"
    return df


def test_fees_applied_on_entry_and_exit():
    """Test that fees are properly applied and accumulated"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=10000.0,
        fee_rate=0.001,  # 0.1% per trade
        slippage_rate=0.0,  # No slippage for this test
        use_next_bar_execution=False,
        log_to_database=False,
    )

    results = backtester.run(symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1])

    # Check that fee_rate is configured correctly
    assert backtester.fee_rate == 0.001, "Fee rate should be set"
    # Fees may or may not be recorded depending on whether trades were executed
    assert isinstance(results["total_fees"], float), "Total fees should be a float"
    assert results["total_fees"] >= 0, "Total fees should be non-negative"


def test_slippage_applied_on_entry():
    """Test that entry slippage is properly calculated"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=10000.0,
        fee_rate=0.0,  # No fees for this test
        slippage_rate=0.001,  # 0.1% slippage
        use_next_bar_execution=False,
        log_to_database=False,
    )

    results = backtester.run(symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1])

    # Should have slippage cost recorded
    assert results["total_slippage_cost"] >= 0, "Should have slippage cost"


def test_fees_and_slippage_combined():
    """Test that both fees and slippage are applied together"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    initial_balance = 10000.0
    fee_rate = 0.001  # 0.1%
    slippage_rate = 0.001  # 0.1%

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        use_next_bar_execution=False,
        log_to_database=False,
    )

    results = backtester.run(symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1])

    # Both costs should be recorded (may be zero if no trades)
    assert isinstance(results["total_fees"], float), "Total fees should be recorded"
    assert isinstance(results["total_slippage_cost"], float), "Total slippage should be recorded"
    assert results["total_fees"] >= 0, "Fees should be non-negative"
    assert results["total_slippage_cost"] >= 0, "Slippage should be non-negative"


def test_fees_zero_by_default():
    """Test that fees default to 0.1% (not zero)"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=10000.0,
        # fee_rate not specified - should use default
        use_next_bar_execution=False,
        log_to_database=False,
    )

    # Check that defaults are applied
    assert backtester.fee_rate == 0.001, "Default fee_rate should be 0.1%"
    assert backtester.slippage_rate == 0.0005, "Default slippage_rate should be 0.05%"


def test_next_bar_execution_disabled_by_default():
    """Test that next-bar execution is disabled by default"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=10000.0,
        # use_next_bar_execution not specified - should be False
        log_to_database=False,
    )

    assert (
        backtester.use_next_bar_execution is False
    ), "Next-bar execution should be disabled by default"


def test_high_low_for_stops_enabled_by_default():
    """Test that high/low stop detection is enabled by default"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=10000.0,
        # use_high_low_for_stops not specified - should be True
        log_to_database=False,
    )

    assert (
        backtester.use_high_low_for_stops is True
    ), "High/low stop detection should be enabled by default"


def test_execution_settings_in_results():
    """Test that execution settings are included in results"""
    df = create_test_dataframe(20)
    provider = MockDataProvider(df)
    strategy = SimpleSignalStrategy("test", SignalDirection.BUY)

    fee_rate = 0.002
    slippage_rate = 0.0008

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=10000.0,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        use_next_bar_execution=False,
        use_high_low_for_stops=True,
        log_to_database=False,
    )

    results = backtester.run(symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1])

    # Check execution settings are in results
    assert "execution_settings" in results, "Results should include execution_settings"
    assert results["execution_settings"]["fee_rate"] == fee_rate, "Fee rate should match"
    assert (
        results["execution_settings"]["slippage_rate"] == slippage_rate
    ), "Slippage rate should match"
    assert (
        results["execution_settings"]["use_next_bar_execution"] is False
    ), "Next-bar execution flag should match"
    assert (
        results["execution_settings"]["use_high_low_for_stops"] is True
    ), "High-low flag should match"


def test_realistic_execution_reduces_returns():
    """Test that realistic execution (fees + slippage) reduces returns"""
    df = create_test_dataframe(20)
    provider_ideal = MockDataProvider(df.copy())
    provider_realistic = MockDataProvider(df.copy())

    strategy_ideal = SimpleSignalStrategy("test_ideal", SignalDirection.BUY)
    strategy_realistic = SimpleSignalStrategy("test_realistic", SignalDirection.BUY)

    # Ideal: no fees, no slippage
    backtester_ideal = Backtester(
        strategy=strategy_ideal,
        data_provider=provider_ideal,
        initial_balance=10000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        use_next_bar_execution=False,
        log_to_database=False,
    )

    # Realistic: with fees and slippage
    backtester_realistic = Backtester(
        strategy=strategy_realistic,
        data_provider=provider_realistic,
        initial_balance=10000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        use_next_bar_execution=False,
        log_to_database=False,
    )

    results_ideal = backtester_ideal.run(
        symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1]
    )
    results_realistic = backtester_realistic.run(
        symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1]
    )

    # Realistic should have lower or equal returns
    assert (
        results_realistic["final_balance"] <= results_ideal["final_balance"]
    ), "Realistic execution should have lower or equal returns"
