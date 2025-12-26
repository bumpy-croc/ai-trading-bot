"""Shared models for backtest and live trading engines.

This module provides unified data models used by both the backtesting
and live trading engines to ensure consistency in position and trade
representation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Epsilon for floating-point comparisons in financial calculations
EPSILON = 1e-9


class PositionSide(Enum):
    """Side of a trading position."""

    LONG = "long"
    SHORT = "short"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "PositionSide":
        """Create a PositionSide from a string value.

        Args:
            value: String value ('long' or 'short').

        Returns:
            PositionSide enum value.

        Raises:
            ValueError: If value is not a valid side.
        """
        value_lower = value.lower()
        if value_lower == "long":
            return cls.LONG
        elif value_lower == "short":
            return cls.SHORT
        else:
            raise ValueError(f"Invalid position side: {value}")


class OrderStatus(Enum):
    """Status of an order."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class BasePosition:
    """Base class for active trading positions.

    This class contains all fields that are common between the
    backtesting ActiveTrade and live Position classes.

    Attributes:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        side: Position side (LONG or SHORT).
        entry_price: Price at which the position was entered.
        entry_time: Timestamp when the position was opened.
        size: Position size as fraction of balance (0-1).
        stop_loss: Stop loss price level.
        take_profit: Take profit price level (optional).
        entry_balance: Account balance at entry time (optional).
        original_size: Original position size before partial operations.
        current_size: Current position size after partial operations.
        partial_exits_taken: Number of partial exits executed.
        scale_ins_taken: Number of scale-in operations executed.
        trailing_stop_activated: Whether trailing stop is active.
        trailing_stop_price: Current trailing stop price level.
        breakeven_triggered: Whether breakeven has been triggered.
    """

    symbol: str
    side: PositionSide | str  # Support both enum and string for backward compatibility
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_balance: float | None = None
    # Partial operations runtime state
    original_size: float | None = None
    current_size: float | None = None
    partial_exits_taken: int = 0
    scale_ins_taken: int = 0
    # Trailing stop state
    trailing_stop_activated: bool = False
    trailing_stop_price: float | None = None
    breakeven_triggered: bool = False
    # Extended metadata for engine-specific data
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass construction.

        IMPORTANT: This method mutates field values to ensure valid state:
        1. Normalizes string side to PositionSide enum for type safety
        2. Clamps size to maximum of 1.0 (100% of balance) to prevent over-leverage
        3. Auto-initializes original_size and current_size from size if not provided

        These mutations ensure positions are always in a valid state, even if
        constructed with invalid or incomplete data.
        """
        # Normalize side to PositionSide enum if string
        if isinstance(self.side, str):
            self.side = PositionSide.from_string(self.side)

        # Limit position size to 100% of balance to prevent over-leverage
        if self.size > 1.0 + EPSILON:  # Use epsilon for float comparison
            logger.warning(
                "Position size %.2f exceeds maximum 1.0, clamping to 1.0",
                self.size,
            )
            self.size = 1.0

        # Validate size is non-negative
        if self.size < -EPSILON:
            logger.warning("Position size %.2f is negative, setting to 0.0", self.size)
            self.size = 0.0

        # Initialize original/current size if not provided
        if self.original_size is None:
            self.original_size = self.size
        if self.current_size is None:
            self.current_size = self.size

    @property
    def side_str(self) -> str:
        """Get the side as a string for backward compatibility."""
        if isinstance(self.side, PositionSide):
            return self.side.value
        return str(self.side)

    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.side == PositionSide.LONG or self.side_str == "long"

    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.side == PositionSide.SHORT or self.side_str == "short"


@dataclass
class BaseTrade:
    """Base class for completed trades.

    This class contains all fields that are common between the
    backtesting Trade and live Trade classes.

    Attributes:
        symbol: Trading symbol.
        side: Trade side (LONG or SHORT).
        entry_price: Entry price of the trade.
        exit_price: Exit price of the trade.
        entry_time: Timestamp when the trade was opened.
        exit_time: Timestamp when the trade was closed.
        size: Trade size as fraction of balance.
        pnl: Realized profit/loss in account currency.
        pnl_percent: Sized percentage return (decimal, e.g., 0.02 = +2%).
        exit_reason: Reason for exiting the trade.
        stop_loss: Stop loss price that was set.
        take_profit: Take profit price that was set.
        mfe: Maximum favorable excursion (peak unrealized profit %).
        mae: Maximum adverse excursion (max drawdown during trade %).
        mfe_price: Price at which MFE occurred.
        mae_price: Price at which MAE occurred.
        mfe_time: Timestamp of MFE.
        mae_time: Timestamp of MAE.
    """

    symbol: str
    side: PositionSide | str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    pnl: float
    pnl_percent: float | None = None
    exit_reason: str = "unknown"
    stop_loss: float | None = None
    take_profit: float | None = None
    # MFE/MAE tracking fields
    mfe: float = 0.0
    mae: float = 0.0
    mfe_price: float | None = None
    mae_price: float | None = None
    mfe_time: datetime | None = None
    mae_time: datetime | None = None
    # Extended metadata for engine-specific data
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass construction."""
        # Normalize side to PositionSide enum if string
        if isinstance(self.side, str):
            self.side = PositionSide.from_string(self.side)

    @property
    def side_str(self) -> str:
        """Get the side as a string for backward compatibility."""
        if isinstance(self.side, PositionSide):
            return self.side.value
        return str(self.side)

    def is_winning(self) -> bool:
        """Check if this trade was profitable."""
        return self.pnl > 0

    def is_long(self) -> bool:
        """Check if this was a long trade."""
        return self.side == PositionSide.LONG or self.side_str == "long"

    def is_short(self) -> bool:
        """Check if this was a short trade."""
        return self.side == PositionSide.SHORT or self.side_str == "short"

    def duration_seconds(self) -> float:
        """Calculate trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()

    def duration_hours(self) -> float:
        """Calculate trade duration in hours."""
        return self.duration_seconds() / 3600


@dataclass
class PartialExitResult:
    """Result of a partial exit execution.

    Attributes:
        realized_pnl: Cash profit/loss realized from the partial exit.
        new_current_size: Updated current position size after exit.
        partial_exits_taken: Total count of partial exits taken.
    """

    realized_pnl: float
    new_current_size: float
    partial_exits_taken: int


@dataclass
class ScaleInResult:
    """Result of a scale-in execution.

    Attributes:
        new_size: Updated total position size after scale-in.
        new_current_size: Updated current position size after scale-in.
        scale_ins_taken: Total count of scale-ins taken.
    """

    new_size: float
    new_current_size: float
    scale_ins_taken: int


# Type aliases for backward compatibility
Position = BasePosition
Trade = BaseTrade

__all__ = [
    "PositionSide",
    "OrderStatus",
    "BasePosition",
    "BaseTrade",
    "Position",
    "Trade",
    "PartialExitResult",
    "ScaleInResult",
]
