from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.engines.shared.models import BasePosition, BaseTrade


# Backtest Trade is identical to BaseTrade - just use it directly
Trade = BaseTrade


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
