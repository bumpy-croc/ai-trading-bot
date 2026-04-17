"""Declarative experimentation framework for strategy improvement."""

from .runner import ExperimentRunner
from .schemas import ExperimentConfig, ExperimentResult, ParameterSet
from .suite import (
    BacktestSettings,
    ComparisonSettings,
    ExperimentSuiteRunner,
    SuiteConfig,
    SuiteResult,
    VariantSpec,
)
from .suite_loader import SuiteValidationError, load_suite, parse_suite
from .walk_forward import (
    FoldResult,
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
)

__all__ = [
    "BacktestSettings",
    "ComparisonSettings",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSuiteRunner",
    "FoldResult",
    "ParameterSet",
    "SuiteConfig",
    "SuiteResult",
    "SuiteValidationError",
    "VariantSpec",
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardResult",
    "load_suite",
    "parse_suite",
]
