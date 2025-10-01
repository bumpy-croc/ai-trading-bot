"""
Strategy Components Module

This module contains the component-based strategy architecture that replaces
the monolithic strategy approach with composable, testable components.

Components:
- SignalGenerator: Generates trading signals based on market data
- RiskManager: Manages position sizing and risk controls
- PositionSizer: Calculates position sizes based on various factors
- RegimeContext: Provides market regime information to components
"""

from .signal_generator import SignalGenerator, Signal, SignalDirection
from .risk_manager import RiskManager, Position, MarketData
from .position_sizer import PositionSizer
from .regime_context import RegimeContext, TrendLabel, VolLabel, EnhancedRegimeDetector
from .strategy_manager import StrategyManager, StrategyVersion, StrategyExecution

# Re-export TrendLabel and VolLabel from existing regime module for compatibility
from src.regime.detector import TrendLabel, VolLabel

__all__ = [
    "SignalGenerator",
    "Signal", 
    "SignalDirection",
    "RiskManager",
    "Position",
    "MarketData", 
    "PositionSizer",
    "RegimeContext",
    "TrendLabel",
    "VolLabel",
    "EnhancedRegimeDetector",
    "StrategyManager",
    "StrategyVersion",
    "StrategyExecution",
]