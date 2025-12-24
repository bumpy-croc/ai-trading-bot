"""Risk module for backtesting risk control logic.

This module contains handlers for correlation-based position sizing
and other risk control coordination.
"""

from src.backtesting.risk.correlation_handler import CorrelationHandler

__all__ = [
    "CorrelationHandler",
]
