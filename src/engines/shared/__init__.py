"""Shared engine components used by both backtest and live trading engines.

This module provides unified implementations for common trading engine functionality:
- Position and Trade models
- Cost calculations (fees, slippage)
- Trailing stop management
- Performance tracking (future)
- Partial operations (future)
"""

from src.engines.shared.cost_calculator import CostCalculator, CostResult
from src.engines.shared.models import (
    BasePosition,
    BaseTrade,
    OrderStatus,
    Position,
    PositionSide,
    Trade,
)
from src.engines.shared.trailing_stop_manager import (
    TrailingStopManager,
    TrailingStopUpdate,
)

__all__ = [
    # Models
    "PositionSide",
    "OrderStatus",
    "BasePosition",
    "BaseTrade",
    "Position",
    "Trade",
    # Cost calculation
    "CostCalculator",
    "CostResult",
    # Trailing stop management
    "TrailingStopManager",
    "TrailingStopUpdate",
]
