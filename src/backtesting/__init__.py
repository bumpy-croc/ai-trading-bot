from importlib import import_module
from .engine import Backtester, Trade
# Expose BacktestDashboard at package level for easy import
BacktestDashboard = import_module('backtesting.dashboard.dashboard').BacktestDashboard  # type: ignore