"""Execution module for backtesting trade execution logic.

This module contains handlers for trade entry/exit execution, position tracking,
and realistic execution simulation (fees, slippage, next-bar execution).
"""

from src.engines.backtest.execution.entry_handler import EntryHandler
from src.engines.backtest.execution.execution_engine import ExecutionEngine
from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker

__all__ = [
    "EntryHandler",
    "ExecutionEngine",
    "ExitHandler",
    "PositionTracker",
]
