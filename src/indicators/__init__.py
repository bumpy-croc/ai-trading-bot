"""Deprecated indicator namespace preserved for backward compatibility."""

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
    "src.indicators is deprecated; import from src.tech.indicators instead.",
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
