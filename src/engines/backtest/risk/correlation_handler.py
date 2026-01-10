"""CorrelationHandler for backtest engine.

This module re-exports the shared CorrelationHandler for backward compatibility.
The handler is now a shared component in src/engines/shared/correlation_handler.py
to ensure parity between backtesting and live trading.
"""

# Re-export from shared module for backward compatibility
from src.engines.shared.correlation_handler import CorrelationHandler

__all__ = ["CorrelationHandler"]
