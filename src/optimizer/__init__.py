"""
Optimizer module for strategy parameter optimization, walk-forward analysis,
and strategy drift detection.

This module provides tools for optimizing trading strategy parameters,
validating strategy robustness via walk-forward analysis, and monitoring
live performance drift against backtested expectations.
"""

# Re-export key interfaces for convenience
from .schemas import ExperimentConfig, ExperimentResult, ParameterSet, Suggestion
from .strategy_drift import DriftConfig, DriftReport, DriftSeverity, StrategyDriftDetector
from .walk_forward import (
    FoldResult,
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
)

__all__ = [
    # Core schemas
    "ExperimentConfig",
    "ExperimentResult",
    "ParameterSet",
    "Suggestion",
    # Walk-forward analysis
    "FoldResult",
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardResult",
    # Strategy drift detection
    "DriftConfig",
    "DriftReport",
    "DriftSeverity",
    "StrategyDriftDetector",
]
