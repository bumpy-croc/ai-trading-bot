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
from .momentum_signal_generator import MomentumSignalGenerator
from .performance_tracker import PerformanceTracker
from .policies import (
    DynamicRiskDescriptor,
    PartialExitPolicyDescriptor,
    PolicyBundle,
    TrailingStopPolicyDescriptor,
)
from .position_sizer import (
    ConfidenceWeightedSizer,
    FixedFractionSizer,
    KellySizer,
    PositionSizer,
    RegimeAdaptiveSizer,
)
from .regime_context import EnhancedRegimeDetector, RegimeContext, TrendLabel, VolLabel
from .risk_adapter import CoreRiskAdapter
from .risk_manager import (
    FixedRiskManager,
    MarketData,
    Position,
    RegimeAdaptiveRiskManager,
    RiskManager,
    VolatilityRiskManager,
)
from .runtime import (
    FeatureCache,
    FeatureGeneratorSpec,
    RuntimeContext,
    StrategyDataset,
    StrategyRuntime,
)
from .signal_generator import (
    HierarchicalSignalGenerator,
    HoldSignalGenerator,
    RandomSignalGenerator,
    RegimeAdaptiveSignalGenerator,
    Signal,
    SignalDirection,
    SignalGenerator,
    WeightedVotingSignalGenerator,
)
from .strategy import Strategy
from .strategy_factory import StrategyBuilder, StrategyFactory
from .strategy_lineage import StrategyLineageTracker
from .strategy_manager import ComponentStrategyManager
from .strategy_registry import StrategyRegistry, StrategyVersion
from .technical_signal_generator import (
    MACDSignalGenerator,
    RSISignalGenerator,
    TechnicalSignalGenerator,
)
from .testing.component_performance_tester import ComponentPerformanceTester
from .testing.performance_attribution import PerformanceAttributionAnalyzer
from .testing.regime_tester import RegimeTester
from .testing.test_datasets import TestDatasetGenerator

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
