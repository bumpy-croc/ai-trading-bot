"""
Strategy Testing Framework

This module provides comprehensive testing and validation tools for strategy
performance comparison, migration validation, and statistical analysis.
"""

from .performance_comparison_engine import (
    ComparisonConfig,
    PerformanceComparisonEngine,
    StrategyComparisonResult,
    quick_strategy_comparison,
    validate_migration_readiness,
)
from .performance_parity_validator import (
    MetricComparison,
    MetricType,
    PerformanceComparisonReport,
    PerformanceParityReporter,
    PerformanceParityValidator,
    ToleranceConfig,
    ValidationResult,
)
from .statistical_tests import (
    EquivalenceTests,
    FinancialStatisticalTests,
    StatisticalTestResult,
    format_test_results,
)

__all__ = [
    # Main engine
    "PerformanceComparisonEngine",
    "ComparisonConfig",
    "StrategyComparisonResult",
    
    # Performance parity validation
    "PerformanceParityValidator",
    "PerformanceComparisonReport",
    "PerformanceParityReporter",
    "ToleranceConfig",
    "MetricComparison",
    "MetricType",
    "ValidationResult",
    
    # Statistical testing
    "FinancialStatisticalTests",
    "EquivalenceTests",
    "StatisticalTestResult",
    "format_test_results",
    
    # Convenience functions
    "quick_strategy_comparison",
    "validate_migration_readiness",
]