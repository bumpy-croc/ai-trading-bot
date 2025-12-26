from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.engines.shared.models import BasePosition


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
class ActiveTrade(BasePosition):
    """Represents an active trade during backtest iteration.

    Extends BasePosition with backtest-specific exit tracking fields.
    All core position fields are inherited from BasePosition.

    Note: component_notional removed - compute on-demand as:
        notional = current_size * balance
    """

    # Backtest-specific: temporary exit tracking during iteration
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None
