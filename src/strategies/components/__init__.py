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

# ruff: noqa: I001

from .strategy import Strategy
from .runtime import (
    FeatureGeneratorSpec,
    FeatureCache,
    StrategyDataset,
    RuntimeContext,
    StrategyRuntime,
)
from .signal_generator import (
    SignalGenerator,
    Signal,
    SignalDirection,
    HoldSignalGenerator,
    RandomSignalGenerator,
    WeightedVotingSignalGenerator,
    HierarchicalSignalGenerator,
    RegimeAdaptiveSignalGenerator,
)
from .risk_manager import (
    RiskManager,
    Position,
    MarketData,
    FixedRiskManager,
    VolatilityRiskManager,
    RegimeAdaptiveRiskManager,
)
from .risk_adapter import CoreRiskAdapter
from .policies import (
    PolicyBundle,
    PartialExitPolicyDescriptor,
    TrailingStopPolicyDescriptor,
    DynamicRiskDescriptor,
)
from .position_sizer import (
    PositionSizer,
    FixedFractionSizer,
    ConfidenceWeightedSizer,
    KellySizer,
    RegimeAdaptiveSizer,
)
from .regime_context import RegimeContext, TrendLabel, VolLabel, EnhancedRegimeDetector
from .strategy_manager import ComponentStrategyManager
from .strategy_factory import StrategyFactory, StrategyBuilder
from .strategy_registry import StrategyRegistry, StrategyVersion
from .performance_tracker import PerformanceTracker
from .strategy_lineage import StrategyLineageTracker
from .ml_signal_generator import MLSignalGenerator, MLBasicSignalGenerator
from .technical_signal_generator import (
    TechnicalSignalGenerator,
    RSISignalGenerator,
    MACDSignalGenerator,
)
from .momentum_signal_generator import MomentumSignalGenerator
from .aggressive_trend_signal_generator import AggressiveTrendSignalGenerator
from .crash_detection_signal_generator import CrashDetectionSignalGenerator
from .testing.test_datasets import TestDatasetGenerator
from .testing.component_performance_tester import ComponentPerformanceTester
from .testing.regime_tester import RegimeTester
from .testing.performance_attribution import PerformanceAttributionAnalyzer
from src.database.models import StrategyExecution

__all__ = [
    # Core classes
    "Strategy",
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
    "FeatureGeneratorSpec",
    "FeatureCache",
    "StrategyDataset",
    "RuntimeContext",
    "StrategyRuntime",
    # Management classes
    "ComponentStrategyManager",
    "StrategyFactory",
    "StrategyBuilder",
    "StrategyRegistry",
    "StrategyVersion",
    "StrategyExecution",
    "PerformanceTracker",
    "StrategyLineageTracker",
    # Signal generators
    "MLSignalGenerator",
    "MLBasicSignalGenerator",
    "TechnicalSignalGenerator",
    "RSISignalGenerator",
    "MACDSignalGenerator",
    "MomentumSignalGenerator",
    "AggressiveTrendSignalGenerator",
    "CrashDetectionSignalGenerator",
    "HoldSignalGenerator",
    "RandomSignalGenerator",
    "WeightedVotingSignalGenerator",
    "HierarchicalSignalGenerator",
    "RegimeAdaptiveSignalGenerator",
    # Risk managers
    "FixedRiskManager",
    "VolatilityRiskManager",
    "RegimeAdaptiveRiskManager",
    "CoreRiskAdapter",
    "PolicyBundle",
    "PartialExitPolicyDescriptor",
    "TrailingStopPolicyDescriptor",
    "DynamicRiskDescriptor",
    # Position sizers
    "FixedFractionSizer",
    "ConfidenceWeightedSizer",
    "KellySizer",
    "RegimeAdaptiveSizer",
    # Testing framework
    "TestDatasetGenerator",
    "ComponentPerformanceTester",
    "RegimeTester",
    "PerformanceAttributionAnalyzer",
]
