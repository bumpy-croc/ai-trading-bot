"""Shared execution modeling for live and backtest engines.

This module provides a unified execution model layer that determines
fill eligibility and pricing based on order type and available market data.
"""

from src.engines.shared.execution.execution_decision import (
    ExecutionDecision,
    LiquidityType,
)
from src.engines.shared.execution.execution_model import (
    ExecutionModel,
    FillModelProtocol,
)
from src.engines.shared.execution.fill_policy import (
    FillPolicy,
    default_fill_policy,
    resolve_fill_policy,
)
from src.engines.shared.execution.market_snapshot import MarketSnapshot
from src.engines.shared.execution.ohlc_fill_model import OhlcFillModel
from src.engines.shared.execution.order_intent import OrderIntent

__all__ = [
    "ExecutionDecision",
    "ExecutionModel",
    "FillModelProtocol",
    "FillPolicy",
    "LiquidityType",
    "MarketSnapshot",
    "OhlcFillModel",
    "OrderIntent",
    "default_fill_policy",
    "resolve_fill_policy",
]
