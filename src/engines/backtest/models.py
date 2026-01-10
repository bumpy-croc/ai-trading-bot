from __future__ import annotations

from dataclasses import dataclass

from src.engines.shared.models import BasePosition, BaseTrade

# Backtest Trade is identical to BaseTrade - just use it directly
Trade = BaseTrade


@dataclass
class ActiveTrade(BasePosition):
    """Represents an active trading position during backtest iteration.

    Extends BasePosition with no additional fields. All position tracking
    is handled by the base class. Exit information is tracked separately
    when converting to a completed Trade.

    Note: component_notional removed - compute on-demand as:
        notional = current_size * balance
    """

    pass
