"""Execution module for backtesting trade execution logic.

This module contains handlers for trade entry/exit execution, position tracking,
and realistic execution simulation (fees, slippage, next-bar execution).
"""

from src.backtesting.execution.entry_handler import EntryHandler
from src.backtesting.execution.execution_engine import ExecutionEngine
from src.backtesting.execution.exit_handler import ExitHandler
from src.backtesting.execution.position_tracker import PositionTracker

__all__ = [
    "EntryHandler",
    "ExecutionEngine",
    "ExitHandler",
    "PositionTracker",
]
