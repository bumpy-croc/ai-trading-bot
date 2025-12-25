"""Execution handlers for live trading.

This module provides modular components for trade execution:
- LivePositionTracker: Multi-position state and MFE/MAE tracking
- LiveExecutionEngine: Order execution with fees and slippage
- LiveEntryHandler: Entry signal processing
- LiveExitHandler: Exit condition checking
"""

from src.live.execution.entry_handler import (
    LiveEntryHandler,
    LiveEntryResult,
    LiveEntrySignal,
)
from src.live.execution.execution_engine import (
    EntryExecutionResult,
    ExitExecutionResult,
    LiveExecutionEngine,
    LiveExecutionResult,
)
from src.live.execution.exit_handler import (
    LiveExitCheck,
    LiveExitHandler,
    LiveExitResult,
)
from src.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PartialExitResult,
    PositionCloseResult,
    PositionSide,
    ScaleInResult,
)

__all__ = [
    # Position tracker
    "LivePosition",
    "LivePositionTracker",
    "PositionCloseResult",
    "PositionSide",
    "PartialExitResult",
    "ScaleInResult",
    # Execution engine
    "LiveExecutionEngine",
    "LiveExecutionResult",
    "EntryExecutionResult",
    "ExitExecutionResult",
    # Entry handler
    "LiveEntryHandler",
    "LiveEntrySignal",
    "LiveEntryResult",
    # Exit handler
    "LiveExitHandler",
    "LiveExitCheck",
    "LiveExitResult",
]
