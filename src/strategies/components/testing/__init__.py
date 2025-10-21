"""
Component Testing Framework

This module provides comprehensive testing capabilities for strategy components,
including performance testing, regime-specific testing, and attribution analysis.
"""

from .component_performance_tester import (
    ComponentPerformanceTester,
    ComponentTestResults,
    RiskTestResults,
    SignalTestResults,
    SizingTestResults,
)
from .performance_attribution import (
    AttributionReport,
    ComponentAttribution,
    PerformanceAttributionAnalyzer,
)
from .regime_tester import RegimeComparisonResults, RegimeTester, RegimeTestResults
from .test_datasets import MarketScenario, SyntheticDataGenerator, TestDatasetGenerator

__all__ = [
    # Component Performance Testing
    'ComponentPerformanceTester',
    'SignalTestResults',
    'RiskTestResults',
    'SizingTestResults',
    'ComponentTestResults',

    # Regime Testing
    'RegimeTester',
    'RegimeTestResults',
    'RegimeComparisonResults',

    # Performance Attribution
    'PerformanceAttributionAnalyzer',
    'ComponentAttribution',
    'AttributionReport',

    # Test Datasets
    'TestDatasetGenerator',
    'MarketScenario',
    'SyntheticDataGenerator'
]