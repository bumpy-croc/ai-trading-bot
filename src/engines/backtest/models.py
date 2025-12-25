from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """Represents a single backtest trade outcome."""

    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float  # fraction of balance (0-1)
    pnl: float
    pnl_percent: float
    exit_reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    # MFE/MAE tracking fields
    mfe: float = 0.0
    mae: float = 0.0
    mfe_price: float | None = None
    mae_price: float | None = None
    mfe_time: datetime | None = None
    mae_time: datetime | None = None


@dataclass
class ActiveTrade:
    """Represents an active trade during backtest iteration.

    Tracks all state for a position including entry details, risk levels,
    trailing stop state, and partial operation history.
    """

    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    size: float  # Position size as fraction of balance (0-1), capped at 1.0
    stop_loss: float
    take_profit: float | None = None
    entry_balance: float | None = None
    # Exit tracking
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None
    # Partial operations runtime state
    original_size: float = field(init=False)
    current_size: float = field(init=False)
    partial_exits_taken: int = 0
    scale_ins_taken: int = 0
    # Trailing state
    trailing_stop_activated: bool = False
    breakeven_triggered: bool = False
    trailing_stop_price: float | None = None
    # Component runtime tracking
    component_notional: float | None = None

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass construction."""
        # Limit position size to 100% of balance
        self.size = min(self.size, 1.0)
        self.original_size = self.size
        self.current_size = self.size
