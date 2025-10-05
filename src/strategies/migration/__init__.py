"""
Strategy Migration Module

This module provides comprehensive utilities for migrating legacy strategies
to the new component-based architecture, including conversion, validation,
cross-validation testing, rollback capabilities, and audit trail functionality.
"""

from .audit_trail import AuditEvent, AuditTrailManager, MigrationSession
from .config_mapper import (
    ComponentConfigMapping,
    ConfigMapper,
    ParameterMapping,
)
from .cross_validation import (
    CrossValidationTester,
    CrossValidationReport,
    ComparisonResult,
)
from .difference_analysis import (
    DifferenceAnalyzer,
    DifferenceAnalysisReport,
    DifferenceMetric,
)
from .regression_testing import (
    RegressionTester,
    RegressionTestCase,
    RegressionTestResult,
    RegressionTestSuite,
)
from .rollback_manager import (
    RollbackManager,
    RollbackPoint,
    RollbackResult,
)
from .rollback_validation import (
    RollbackValidator,
    RollbackValidationResult,
)
from .strategy_converter import (
    ComponentMapping,
    ConversionReport,
    StrategyConverter,
)
from .validation_utils import (
    StrategyValidator,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    # Strategy Converter
    "StrategyConverter",
    "ConversionReport",
    "ComponentMapping",

    # Configuration Mapper
    "ConfigMapper",
    "ParameterMapping",
    "ComponentConfigMapping",

    # Validation Utils
    "StrategyValidator",
    "ValidationReport",
    "ValidationResult",

    # Cross-Validation Testing
    "CrossValidationTester",
    "CrossValidationReport",
    "ComparisonResult",

    # Regression Testing
    "RegressionTester",
    "RegressionTestCase",
    "RegressionTestResult",
    "RegressionTestSuite",

    # Difference Analysis
    "DifferenceAnalyzer",
    "DifferenceAnalysisReport",
    "DifferenceMetric",

    # Rollback Management
    "RollbackManager",
    "RollbackPoint",
    "RollbackResult",
    "RollbackValidator",
    "RollbackValidationResult",

    # Audit Trail
    "AuditTrailManager",
    "AuditEvent",
    "MigrationSession",
]
