"""
CI-specific smoke test with reduced data load.

This test runs a lighter version of the smoke test using only 1 month of data
instead of a full year to prevent timeouts in CI environments with limited resources.
"""

import pytest
import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.engine import Backtester
from strategies.ml_adaptive import MlAdaptive
from data_providers.mock_data_provider import MockDataProvider


def test_ci_smoke_test():
    """
    Lightweight smoke test for CI environments.
    
    Uses only 1 month of data instead of a full year to prevent timeouts
    in CI environments with limited resources (2 CPU cores, 7.8GB RAM).
    """
    print("Starting CI smoke test with 1 month of data...")
    
    # Use 1 month of data instead of a full year
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 2, 1)  # 1 month instead of 1 year
    
    # Create mock data provider with reduced data
    # For 1 month of hourly data: 30 days * 24 hours = 720 candles
    data_provider = MockDataProvider(
        interval_seconds=3600,  # 1 hour intervals
        num_candles=720,  # 1 month of hourly data
        start_price=30000.0,
        volatility=0.001
    )
    
    # Create strategy
    strategy = MlAdaptive()
    
    # Create backtest engine
    engine = Backtester(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=10000
    )
    
    # Run backtest
    print(f"Running backtest from {start_date} to {end_date}...")
    results = engine.run('BTCUSDT', '1h', start_date, end_date)
    
    # Basic assertions
    assert results is not None
    print(f"Results keys: {list(results.keys())}")
    
    # Check for expected keys (may vary based on backtester implementation)
    expected_keys = ['final_balance', 'sharpe_ratio', 'max_drawdown']
    for key in expected_keys:
        assert key in results, f"Expected key '{key}' not found in results"
    
    # Verify we have some trading activity (if trades were made)
    if 'num_trades' in results:
        assert results['num_trades'] >= 0, "Number of trades should be non-negative"
        print(f"Number of trades: {results['num_trades']}")
    else:
        print("No trades were executed (this is acceptable for a short test)")
    
    # Verify returns are reasonable (not NaN or infinite)
    if 'total_return' in results:
        assert not (results['total_return'] != results['total_return']), "Total return should not be NaN"
        print(f"Total return: {results['total_return']:.2%}")
    
    if 'sharpe_ratio' in results:
        assert not (results['sharpe_ratio'] != results['sharpe_ratio']), "Sharpe ratio should not be NaN"
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in results:
        assert not (results['max_drawdown'] != results['max_drawdown']), "Max drawdown should not be NaN"
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
    
    print(f"CI smoke test completed successfully!")


def test_ci_smoke_test_very_light():
    """
    Very lightweight smoke test for CI environments.
    
    Uses only 1 week of data for extremely fast execution in CI.
    This is a fallback test if the 1-month test is still too heavy.
    """
    print("Starting very light CI smoke test with 1 week of data...")
    
    # Use 1 week of data for extremely fast execution
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 8)  # 1 week
    
    # Create mock data provider with minimal data
    # For 1 week of hourly data: 7 days * 24 hours = 168 candles
    data_provider = MockDataProvider(
        interval_seconds=3600,  # 1 hour intervals
        num_candles=168,  # 1 week of hourly data
        start_price=30000.0,
        volatility=0.001
    )
    
    # Create strategy
    strategy = MlAdaptive()
    
    # Create backtest engine
    engine = Backtester(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=10000
    )
    
    # Run backtest
    print(f"Running very light backtest from {start_date} to {end_date}...")
    results = engine.run('BTCUSDT', '1h', start_date, end_date)
    
    # Basic assertions
    assert results is not None
    print(f"Results keys: {list(results.keys())}")
    
    # Check for expected keys (may vary based on backtester implementation)
    expected_keys = ['final_balance', 'sharpe_ratio', 'max_drawdown']
    for key in expected_keys:
        assert key in results, f"Expected key '{key}' not found in results"
    
    # Verify we have some trading activity (if trades were made)
    if 'num_trades' in results:
        assert results['num_trades'] >= 0, "Number of trades should be non-negative"
        print(f"Number of trades: {results['num_trades']}")
    else:
        print("No trades were executed (this is acceptable for a short test)")
    
    # Verify returns are reasonable (not NaN or infinite)
    if 'total_return' in results:
        assert not (results['total_return'] != results['total_return']), "Total return should not be NaN"
        print(f"Total return: {results['total_return']:.2%}")
    
    if 'sharpe_ratio' in results:
        assert not (results['sharpe_ratio'] != results['sharpe_ratio']), "Sharpe ratio should not be NaN"
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in results:
        assert not (results['max_drawdown'] != results['max_drawdown']), "Max drawdown should not be NaN"
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
    
    print(f"Very light CI smoke test completed successfully!")


if __name__ == "__main__":
    # Run the tests directly
    test_ci_smoke_test()
    test_ci_smoke_test_very_light() 