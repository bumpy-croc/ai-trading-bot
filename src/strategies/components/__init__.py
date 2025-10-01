"""
Strategy Components Module

This module contains the component-based strategy architecture that replaces
the monolithic strategy approach with composable, testable components.

Components:
- SignalGenerator: Generates trading signals based on market data
- RiskManager: Manages position sizing and risk controls
- PositionSizer: Calculates position sizes based on various factors
- RegimeContext: Provides market regime information to components
- ML Signal Generators: ML-based signal generation components
- Technical Signal Generators: Technical indicator-based signal generation components
"""

from .signal_generator import SignalGenerator, Signal, SignalDirection
from .risk_manager import RiskManager, Position, MarketData
from .position_sizer import PositionSizer
from .regime_context import RegimeContext, TrendLabel, VolLabel, EnhancedRegimeDetector
from .strategy_manager import StrategyManager
from .ml_signal_generator import MLSignalGenerator, MLBasicSignalGenerator
from .technical_signal_generator import (
    TechnicalSignalGenerator, 
    RSISignalGenerator, 
    MACDSignalGenerator
)

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
    "MLSignalGenerator",
    "MLBasicSignalGenerator",
    "TechnicalSignalGenerator",
    "RSISignalGenerator",
    "MACDSignalGenerator",
]