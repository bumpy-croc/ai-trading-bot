"""Unified regime access utilities for trading strategies.

This module provides consistent regime access patterns and multiplier calculations
to eliminate duplicated logic across position sizers and risk managers.

ARCHITECTURE:
- Single source of truth for regime-based adjustments
- Eliminates ~100 lines of duplicated code across components
- Provides safe access with null handling
- Standardizes multiplier calculations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.strategies.components.regime_context import RegimeContext


# Default multipliers for regime conditions
DEFAULT_HIGH_VOL_MULTIPLIER = 0.8  # Reduce size in high volatility
DEFAULT_BEAR_MULTIPLIER = 0.7  # Reduce size in bear markets
DEFAULT_RANGE_MULTIPLIER = 0.9  # Slight reduction in ranging markets
DEFAULT_BULL_HIGH_CONF_MULTIPLIER = 1.2  # Increase in confident bull markets
DEFAULT_LOW_CONFIDENCE_MULTIPLIER = 0.8  # Reduce when regime confidence is low
DEFAULT_MIN_MULTIPLIER = 0.1  # Floor for multiplier (10% of base)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Threshold for "low confidence"
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.7  # Threshold for "high confidence"


@dataclass(frozen=True)
class RegimeMultiplierConfig:
    """Configuration for regime-based multiplier calculations.

    Attributes:
        high_vol_multiplier: Multiplier applied in high volatility regimes.
        bear_multiplier: Multiplier applied in downtrend regimes.
        range_multiplier: Multiplier applied in range-bound regimes.
        bull_high_conf_multiplier: Multiplier for confident uptrends.
        low_confidence_multiplier: Applied when regime confidence is low.
        min_multiplier: Floor value for the final multiplier.
        confidence_threshold: Threshold below which confidence is "low".
        high_confidence_threshold: Threshold above which confidence is "high".
        scale_by_confidence: Whether to scale by regime confidence directly.
    """

    high_vol_multiplier: float = DEFAULT_HIGH_VOL_MULTIPLIER
    bear_multiplier: float = DEFAULT_BEAR_MULTIPLIER
    range_multiplier: float = DEFAULT_RANGE_MULTIPLIER
    bull_high_conf_multiplier: float = DEFAULT_BULL_HIGH_CONF_MULTIPLIER
    low_confidence_multiplier: float = DEFAULT_LOW_CONFIDENCE_MULTIPLIER
    min_multiplier: float = DEFAULT_MIN_MULTIPLIER
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD
    scale_by_confidence: bool = False


class RegimeHelper:
    """Provides consistent access to regime information.

    This class centralizes regime access patterns to eliminate
    scattered hasattr() checks and magic string comparisons.
    """

    @staticmethod
    def get_trend(regime: "RegimeContext | None") -> str | None:
        """Get regime trend value safely.

        Args:
            regime: Regime context, may be None.

        Returns:
            Trend value ('trend_up', 'trend_down', 'range') or None.
        """
        if regime is None:
            return None
        try:
            if hasattr(regime, "trend") and regime.trend is not None:
                return regime.trend.value if hasattr(regime.trend, "value") else str(regime.trend)
        except (AttributeError, TypeError):
            pass
        return None

    @staticmethod
    def get_volatility(regime: "RegimeContext | None") -> str | None:
        """Get regime volatility value safely.

        Args:
            regime: Regime context, may be None.

        Returns:
            Volatility value ('high_vol', 'low_vol', 'normal') or None.
        """
        if regime is None:
            return None
        try:
            if hasattr(regime, "volatility") and regime.volatility is not None:
                return regime.volatility.value if hasattr(regime.volatility, "value") else str(regime.volatility)
        except (AttributeError, TypeError):
            pass
        return None

    @staticmethod
    def get_confidence(regime: "RegimeContext | None") -> float:
        """Get regime confidence value safely.

        Args:
            regime: Regime context, may be None.

        Returns:
            Confidence value between 0.0 and 1.0, defaults to 1.0.
        """
        if regime is None:
            return 1.0
        try:
            if hasattr(regime, "confidence") and regime.confidence is not None:
                return float(regime.confidence)
        except (AttributeError, TypeError, ValueError):
            pass
        return 1.0

    @staticmethod
    def is_bull_market(regime: "RegimeContext | None") -> bool:
        """Check if regime indicates a bull market (uptrend).

        Args:
            regime: Regime context, may be None.

        Returns:
            True if in uptrend, False otherwise.
        """
        return RegimeHelper.get_trend(regime) == "trend_up"

    @staticmethod
    def is_bear_market(regime: "RegimeContext | None") -> bool:
        """Check if regime indicates a bear market (downtrend).

        Args:
            regime: Regime context, may be None.

        Returns:
            True if in downtrend, False otherwise.
        """
        return RegimeHelper.get_trend(regime) == "trend_down"

    @staticmethod
    def is_ranging(regime: "RegimeContext | None") -> bool:
        """Check if regime indicates a ranging/sideways market.

        Args:
            regime: Regime context, may be None.

        Returns:
            True if in range, False otherwise.
        """
        return RegimeHelper.get_trend(regime) == "range"

    @staticmethod
    def is_high_volatility(regime: "RegimeContext | None") -> bool:
        """Check if regime indicates high volatility.

        Args:
            regime: Regime context, may be None.

        Returns:
            True if high volatility, False otherwise.
        """
        return RegimeHelper.get_volatility(regime) == "high_vol"

    @staticmethod
    def is_low_volatility(regime: "RegimeContext | None") -> bool:
        """Check if regime indicates low volatility.

        Args:
            regime: Regime context, may be None.

        Returns:
            True if low volatility, False otherwise.
        """
        return RegimeHelper.get_volatility(regime) == "low_vol"


class RegimeMultiplierCalculator:
    """Calculates position/risk multipliers based on regime conditions.

    This class provides a unified implementation for regime-based multiplier
    calculations, replacing duplicated logic across position sizers and
    risk managers.

    Example usage:
        calc = RegimeMultiplierCalculator()
        multiplier = calc.calculate(regime)  # Uses default config

        # Or with custom config:
        config = RegimeMultiplierConfig(bear_multiplier=0.5)
        calc = RegimeMultiplierCalculator(config)
        multiplier = calc.calculate(regime)
    """

    def __init__(self, config: RegimeMultiplierConfig | None = None) -> None:
        """Initialize with optional custom configuration.

        Args:
            config: Custom multiplier configuration, or None for defaults.
        """
        self.config = config or RegimeMultiplierConfig()

    def calculate(self, regime: "RegimeContext | None") -> float:
        """Calculate the combined regime multiplier.

        Applies all configured adjustments and returns the final multiplier,
        bounded by the minimum value.

        Args:
            regime: Regime context for adjustment calculations.

        Returns:
            Final multiplier between config.min_multiplier and ~1.2.
        """
        if regime is None:
            return 1.0

        multiplier = 1.0

        # Apply volatility adjustment
        if RegimeHelper.is_high_volatility(regime):
            multiplier *= self.config.high_vol_multiplier

        # Apply trend adjustments
        trend = RegimeHelper.get_trend(regime)
        if trend == "trend_down":
            multiplier *= self.config.bear_multiplier
        elif trend == "range":
            multiplier *= self.config.range_multiplier
        elif trend == "trend_up":
            # Only boost in bull if confidence is high
            confidence = RegimeHelper.get_confidence(regime)
            if confidence > self.config.high_confidence_threshold:
                multiplier *= self.config.bull_high_conf_multiplier

        # Apply confidence adjustment
        confidence = RegimeHelper.get_confidence(regime)
        if confidence < self.config.confidence_threshold:
            multiplier *= self.config.low_confidence_multiplier

        # Optionally scale directly by confidence
        if self.config.scale_by_confidence:
            conf_mult = max(0.5, confidence)
            multiplier *= conf_mult

        return max(self.config.min_multiplier, multiplier)

    def calculate_conservative(self, regime: "RegimeContext | None") -> float:
        """Calculate a more conservative multiplier (no bull boost).

        Similar to calculate() but never increases position size,
        only reduces. Use this for risk-averse strategies.

        Args:
            regime: Regime context for adjustment calculations.

        Returns:
            Conservative multiplier between config.min_multiplier and 1.0.
        """
        if regime is None:
            return 1.0

        multiplier = 1.0

        # Only apply reducing factors
        if RegimeHelper.is_high_volatility(regime):
            multiplier *= self.config.high_vol_multiplier

        if RegimeHelper.is_bear_market(regime):
            multiplier *= self.config.bear_multiplier
        elif RegimeHelper.is_ranging(regime):
            multiplier *= self.config.range_multiplier

        confidence = RegimeHelper.get_confidence(regime)
        if confidence < self.config.confidence_threshold:
            multiplier *= self.config.low_confidence_multiplier

        return max(self.config.min_multiplier, min(1.0, multiplier))


# Convenience function for simple use cases
def get_regime_multiplier(
    regime: "RegimeContext | None",
    config: RegimeMultiplierConfig | None = None,
) -> float:
    """Get regime multiplier using default or custom config.

    Convenience function that wraps RegimeMultiplierCalculator for
    simple one-off calculations.

    Args:
        regime: Regime context for adjustment calculations.
        config: Optional custom configuration.

    Returns:
        Calculated multiplier value.
    """
    calculator = RegimeMultiplierCalculator(config)
    return calculator.calculate(regime)


__all__ = [
    "RegimeHelper",
    "RegimeMultiplierCalculator",
    "RegimeMultiplierConfig",
    "get_regime_multiplier",
    # Constants for external use
    "DEFAULT_HIGH_VOL_MULTIPLIER",
    "DEFAULT_BEAR_MULTIPLIER",
    "DEFAULT_RANGE_MULTIPLIER",
    "DEFAULT_BULL_HIGH_CONF_MULTIPLIER",
    "DEFAULT_LOW_CONFIDENCE_MULTIPLIER",
    "DEFAULT_MIN_MULTIPLIER",
]
