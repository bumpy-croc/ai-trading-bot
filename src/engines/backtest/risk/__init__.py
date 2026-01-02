"""Risk module for backtesting risk control logic.

This module re-exports the shared CorrelationHandler for backward compatibility.
The handler is now a shared component to ensure parity between backtesting
and live trading.
"""

from src.engines.shared.correlation_handler import CorrelationHandler

__all__ = [
    "CorrelationHandler",
]
