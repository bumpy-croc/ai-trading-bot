"""Shared models for backtest and live trading engines.

This module provides unified data models used by both the backtesting
and live trading engines to ensure consistency in position and trade
representation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
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
    def from_string(cls, value: str) -> PositionSide:
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


def normalize_side(side: Any) -> str:
    """Normalize a position side to a lowercase string.

    Handles PositionSide enum, string, or any object with a .value attribute.
    This utility ensures consistent side representation across engines.

    Args:
        side: Position side as PositionSide enum, string, or object with value.

    Returns:
        Lowercase string 'long' or 'short'.

    Examples:
        >>> normalize_side(PositionSide.LONG)
        'long'
        >>> normalize_side("SHORT")
        'short'
        >>> normalize_side("long")
        'long'
    """
    if side is None:
        return "long"
    if isinstance(side, PositionSide):
        return side.value
    if hasattr(side, "value"):
        return str(side.value).lower()
    return str(side).lower()


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
        quantity: Actual quantity of asset bought/sold at entry.
        original_size: Original position size before partial operations.
        current_size: Current position size after partial operations.
        partial_exits_taken: Number of partial exits executed.
        scale_ins_taken: Number of scale-in operations executed.
        trailing_stop_activated: Whether trailing stop is active.
        trailing_stop_price: Current trailing stop price level.
        breakeven_triggered: Whether breakeven has been triggered.
        unrealized_pnl: Current unrealized profit/loss in account currency.
        unrealized_pnl_percent: Current unrealized P&L as percentage.
        order_id: Exchange order ID (live) or None (backtest).
    """

    symbol: str
    # DEPRECATED: String side support will be removed in a future version.
    # Use PositionSide enum instead. String support maintained for backward compatibility.
    side: PositionSide | str
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_balance: float | None = None
    quantity: float | None = None
    # Partial operations runtime state
    original_size: float | None = None
    current_size: float | None = None
    partial_exits_taken: int = 0
    scale_ins_taken: int = 0
    # Trailing stop state
    trailing_stop_activated: bool = False
    trailing_stop_price: float | None = None
    breakeven_triggered: bool = False
    # Real-time P&L tracking (both engines should store)
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    # Order tracking (live: exchange order ID, backtest: None or synthetic)
    order_id: str | None = None
    # Extended metadata for engine-specific data (currently unused, reserved for future extensions)
    # Example use cases: custom risk parameters, strategy-specific tags, A/B test groups
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass construction.

        Validates position constraints and auto-initializes derived fields.
        Raises ValueError for invalid position parameters instead of silently
        mutating them, ensuring callers are aware of validation failures.

        Raises:
            ValueError: If size exceeds 1.0 or is negative.
        """
        def _coerce_float(
            value: float | None, field_name: str, *, required: bool = False
        ) -> float | None:
            if value is None:
                if required:
                    raise ValueError(f"{field_name} is required and must be a real number")
                return None
            if isinstance(value, bool):
                raise ValueError(f"{field_name} must be a real number, not boolean")
            if isinstance(value, Decimal):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be a real number") from exc

        # Normalize side to PositionSide enum if string
        if isinstance(self.side, str):
            self.side = PositionSide.from_string(self.side)

        self.entry_price = _coerce_float(self.entry_price, "entry_price", required=True)
        self.size = _coerce_float(self.size, "size", required=True)
        self.stop_loss = _coerce_float(self.stop_loss, "stop_loss")
        self.take_profit = _coerce_float(self.take_profit, "take_profit")
        self.entry_balance = _coerce_float(self.entry_balance, "entry_balance")
        self.original_size = _coerce_float(self.original_size, "original_size")
        self.current_size = _coerce_float(self.current_size, "current_size")
        self.trailing_stop_price = _coerce_float(
            self.trailing_stop_price, "trailing_stop_price"
        )
        self.unrealized_pnl = _coerce_float(
            self.unrealized_pnl, "unrealized_pnl"
        ) or 0.0
        self.unrealized_pnl_percent = (
            _coerce_float(self.unrealized_pnl_percent, "unrealized_pnl_percent") or 0.0
        )

        # Validate position size does not exceed 100% of balance
        if self.size > 1.0 + EPSILON:  # Use epsilon for float comparison
            raise ValueError(
                f"Position size {self.size} exceeds maximum 1.0 (100% of balance). "
                "Reduce position size to comply with risk limits."
            )

        # Validate size is non-negative
        if self.size < -EPSILON:
            raise ValueError(
                f"Position size {self.size} is negative. "
                "Position size must be a positive value between 0 and 1."
            )

        # Auto-initialize original/current size from size if not provided.
        # These fields track partial operations and default to the initial size.
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
    "normalize_side",
    "BasePosition",
    "BaseTrade",
    "Position",
    "Trade",
    "PartialExitResult",
    "ScaleInResult",
]
