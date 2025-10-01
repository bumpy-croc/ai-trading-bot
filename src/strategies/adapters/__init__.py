"""
Strategy Adapters

This module provides adapter classes for backward compatibility between
the new component-based strategy architecture and the legacy BaseStrategy interface.
"""

from .legacy_adapter import LegacyStrategyAdapter
from .adapter_factory import AdapterFactory

__all__ = ['LegacyStrategyAdapter', 'AdapterFactory']