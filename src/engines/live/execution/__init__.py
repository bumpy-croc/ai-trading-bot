"""Execution handlers for live trading.

This module provides modular components for trade execution:
- LivePositionTracker: Multi-position state and MFE/MAE tracking
- LiveExecutionEngine: Order execution with fees and slippage
- LiveEntryHandler: Entry signal processing
- LiveExitHandler: Exit condition checking
"""

from src.engines.live.execution.entry_handler import (
    LiveEntryHandler,
    LiveEntryResult,
    LiveEntrySignal,
)
from src.engines.live.execution.execution_engine import (
    EntryExecutionResult,
    ExitExecutionResult,
    LiveExecutionEngine,
    LiveExecutionResult,
)
from src.engines.live.execution.exit_handler import (
    LiveExitCheck,
    LiveExitHandler,
    LiveExitResult,
)
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionCloseResult,
)
from src.engines.shared.models import (
    PartialExitResult,
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
