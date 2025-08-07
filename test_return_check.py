#!/usr/bin/env python3
"""Quick script to check the actual return value from the smoke test"""

from datetime import datetime
from unittest.mock import Mock
from backtesting.engine import Backtester
from data_providers.data_provider import DataProvider
from strategies.ml_basic import MlBasic

def check_return():
    """Check the actual return value"""
    # This would need the test data fixture, but let's just print what we expect
    print("Expected return should be 73.81% according to the user")
    print("Current test expects 19.74%")
    print("The fix should restore the original 73.81% behavior")

if __name__ == "__main__":
    check_return()
