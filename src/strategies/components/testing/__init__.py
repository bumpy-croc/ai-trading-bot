"""
Component Testing Framework

This module provides comprehensive testing capabilities for strategy components,
including performance testing, regime-specific testing, and attribution analysis.
"""

from .component_performance_tester import (
    ComponentPerformanceTester,
    SignalTestResults,
    RiskTestResults,
    SizingTestResults,
    ComponentTestResults
)
from .regime_tester import (
    RegimeTester,
    RegimeTestResults,
    RegimeComparisonResults
)
from .performance_attribution import (
    PerformanceAttributionAnalyzer,
    ComponentAttribution,
    AttributionReport
)
from .test_datasets import (
    TestDatasetGenerator,
    MarketScenario,
    SyntheticDataGenerator
)

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