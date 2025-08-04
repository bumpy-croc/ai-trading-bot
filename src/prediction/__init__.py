"""
Prediction Engine Package

This package provides a modular prediction engine for the AI Trading Bot,
separating prediction logic from trading strategy logic.

Key Components:
- Configuration management
- features: Feature extraction and engineering pipeline
- models: Model loading and inference management
- ensemble: Model ensemble and aggregation (Post-MVP)
- utils: Shared utilities and caching

"""

from .engine import PredictionEngine, PredictionResult
from .config import PredictionConfig

__all__ = [
    'PredictionEngine',
    'PredictionResult', 
    'PredictionConfig'
]

__version__ = "1.0.0"
