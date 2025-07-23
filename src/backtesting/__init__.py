from importlib import import_module

# Expose BacktestDashboard at package level for easy import
BacktestDashboard = import_module('backtesting.dashboard.dashboard').BacktestDashboard  # type: ignore