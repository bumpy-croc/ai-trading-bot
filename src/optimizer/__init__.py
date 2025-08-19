"""
Optimizer module for strategy parameter optimization.

This module provides tools for optimizing trading strategy parameters.
"""

# Re-export key interfaces for convenience
from .schemas import ExperimentConfig, ExperimentResult, ParameterSet, Suggestion

__all__ = ["ExperimentConfig", "ExperimentResult", "ParameterSet", "Suggestion"]
