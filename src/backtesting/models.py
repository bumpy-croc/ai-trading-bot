from __future__ import annotations

from dataclasses import dataclass
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
    exit_reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
