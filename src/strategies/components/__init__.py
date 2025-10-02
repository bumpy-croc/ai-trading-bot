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

from src.database.models import StrategyExecution

from .ml_signal_generator import MLBasicSignalGenerator, MLSignalGenerator
from .position_sizer import PositionSizer
from .regime_context import EnhancedRegimeDetector, RegimeContext, TrendLabel, VolLabel
from .risk_manager import MarketData, Position, RiskManager
from .signal_generator import Signal, SignalDirection, SignalGenerator
from .strategy_manager import StrategyManager
from .strategy_registry import StrategyVersion
from .technical_signal_generator import (
    MACDSignalGenerator,
    RSISignalGenerator,
    TechnicalSignalGenerator,
)

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
    "MLSignalGenerator",
    "MLBasicSignalGenerator",
    "TechnicalSignalGenerator",
    "RSISignalGenerator",
    "MACDSignalGenerator",
]