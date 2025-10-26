"""Compatibility shim for legacy imports.

All indicator helpers now live in :mod:`src.tech.indicators.core`. This module
re-exports those functions and emits a deprecation warning so older imports
continue working during the migration.
"""

from __future__ import annotations

import warnings

from src.tech.indicators.core import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    calculate_support_resistance,
    detect_market_regime,
)

warnings.warn(
    "src.indicators.technical is deprecated; import functions from "
    "src.tech.indicators.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_moving_averages",
    "calculate_rsi",
    "calculate_atr",
    "calculate_bollinger_bands",
    "calculate_macd",
    "detect_market_regime",
    "calculate_support_resistance",
    "calculate_ema",
]
