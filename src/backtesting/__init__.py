from importlib import import_module
# * Import Backtester and Trade for convenient package-level access
from .engine import Backtester, Trade
# Expose BacktestDashboard at package level for easy import
BacktestDashboard = import_module('backtesting.dashboard.dashboard').BacktestDashboard  # type: ignore

# * Public symbols exported by the backtesting package
__all__: list[str] = [
    "Backtester",
    "Trade",
    "BacktestDashboard",
]