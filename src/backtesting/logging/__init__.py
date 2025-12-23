"""Logging module for backtesting database event logging.

This module contains handlers for coordinating database logging
of backtest events including trades, strategy executions, and sessions.
"""

from src.backtesting.logging.event_logger import EventLogger

__all__ = [
    "EventLogger",
]
