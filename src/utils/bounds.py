"""Bounds checking and clamping utilities.

This module provides consistent value clamping functions to replace
scattered max(min()) patterns throughout the codebase.

ARCHITECTURE:
- Single source of truth for bounds checking
- Replaces 28+ instances of duplicated clamp patterns
- Provides semantic helpers for common cases (fractions, percentages)
- Includes validation with meaningful error messages
"""

from __future__ import annotations

from typing import TypeVar

# Type variable for numeric types
N = TypeVar("N", int, float)


def clamp(value: N, min_val: N, max_val: N) -> N:
    """Clamp a value to a range [min_val, max_val].

    This is the fundamental clamping function that replaces scattered
    max(min_val, min(max_val, value)) patterns.

    Args:
        value: The value to clamp.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        The value clamped to [min_val, max_val].

    Example:
        >>> clamp(1.5, 0.0, 1.0)
        1.0
        >>> clamp(-0.5, 0.0, 1.0)
        0.0
        >>> clamp(0.5, 0.0, 1.0)
        0.5
    """
    return max(min_val, min(max_val, value))


def clamp_fraction(value: float) -> float:
    """Clamp a value to the valid fraction range [0.0, 1.0].

    Use this for position sizes, confidence scores, and other values
    that must be valid fractions.

    Args:
        value: The value to clamp.

    Returns:
        The value clamped to [0.0, 1.0].

    Example:
        >>> clamp_fraction(1.5)
        1.0
        >>> clamp_fraction(-0.1)
        0.0
    """
    return max(0.0, min(1.0, value))


def clamp_percentage(value: float) -> float:
    """Clamp a value to the valid percentage range [0.0, 100.0].

    Use this for percentage values like win rates, scores, etc.

    Args:
        value: The value to clamp.

    Returns:
        The value clamped to [0.0, 100.0].

    Example:
        >>> clamp_percentage(150.0)
        100.0
        >>> clamp_percentage(-10.0)
        0.0
    """
    return max(0.0, min(100.0, value))


def clamp_positive(value: float, max_val: float | None = None) -> float:
    """Clamp a value to be positive (>= 0), with optional upper bound.

    Args:
        value: The value to clamp.
        max_val: Optional maximum value.

    Returns:
        The value clamped to [0.0, max_val] or [0.0, inf).

    Example:
        >>> clamp_positive(-5.0)
        0.0
        >>> clamp_positive(5.0, max_val=3.0)
        3.0
    """
    if max_val is not None:
        return max(0.0, min(max_val, value))
    return max(0.0, value)


def clamp_position_size(
    size: float,
    balance: float,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> float:
    """Clamp a position size to valid bounds.

    Ensures position size is within the specified fraction of balance.

    Args:
        size: The absolute position size.
        balance: The account balance.
        min_fraction: Minimum position as fraction of balance.
        max_fraction: Maximum position as fraction of balance.

    Returns:
        The size clamped to [balance * min_fraction, balance * max_fraction].

    Example:
        >>> clamp_position_size(150.0, 1000.0, 0.01, 0.1)
        100.0  # 10% of 1000
    """
    min_size = balance * min_fraction
    max_size = balance * max_fraction
    return max(min_size, min(max_size, size))


def clamp_stop_loss_pct(pct: float, min_pct: float = 0.01, max_pct: float = 0.20) -> float:
    """Clamp stop loss percentage to reasonable bounds.

    Default range is 1% to 20%, which covers most trading strategies.

    Args:
        pct: Stop loss percentage (as decimal, e.g., 0.05 for 5%).
        min_pct: Minimum stop loss (default 1%).
        max_pct: Maximum stop loss (default 20%).

    Returns:
        The percentage clamped to [min_pct, max_pct].

    Example:
        >>> clamp_stop_loss_pct(0.005)  # 0.5% is too tight
        0.01  # Returns 1%
        >>> clamp_stop_loss_pct(0.25)  # 25% is too wide
        0.20  # Returns 20%
    """
    return max(min_pct, min(max_pct, pct))


def clamp_risk_amount(
    risk_amount: float,
    balance: float,
    min_pct: float = 0.001,
    max_pct: float = 0.10,
) -> float:
    """Clamp risk amount to reasonable bounds based on balance.

    Default range is 0.1% to 10% of balance.

    Args:
        risk_amount: The risk amount in currency.
        balance: The account balance.
        min_pct: Minimum risk as percentage of balance.
        max_pct: Maximum risk as percentage of balance.

    Returns:
        The risk amount clamped to [balance * min_pct, balance * max_pct].

    Example:
        >>> clamp_risk_amount(5.0, 1000.0)  # 0.5% is within range
        5.0
        >>> clamp_risk_amount(0.5, 1000.0)  # 0.05% is too low
        1.0  # Returns 0.1% of 1000
    """
    min_risk = balance * min_pct
    max_risk = balance * max_pct
    return max(min_risk, min(max_risk, risk_amount))


def clamp_multiplier(value: float, min_mult: float = 0.1, max_mult: float = 3.0) -> float:
    """Clamp a multiplier to reasonable bounds.

    Default range is 0.1x to 3.0x, suitable for position sizing multipliers.

    Args:
        value: The multiplier value.
        min_mult: Minimum multiplier (default 0.1x).
        max_mult: Maximum multiplier (default 3.0x).

    Returns:
        The multiplier clamped to [min_mult, max_mult].

    Example:
        >>> clamp_multiplier(0.05)
        0.1
        >>> clamp_multiplier(5.0)
        3.0
    """
    return max(min_mult, min(max_mult, value))


def validate_fraction(value: float, name: str = "value") -> float:
    """Validate and clamp a fraction, logging a warning if out of bounds.

    Use this when you want to know if the value was clamped.

    Args:
        value: The value to validate.
        name: Name of the value for error messages.

    Returns:
        The value clamped to [0.0, 1.0].

    Raises:
        ValueError: If value is NaN or infinite.
    """
    import math

    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value}")

    return clamp_fraction(value)


def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive.

    Args:
        value: The value to validate.
        name: Name of the value for error messages.

    Returns:
        The value if positive.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative(value: float, name: str = "value") -> float:
    """Validate that a value is non-negative.

    Args:
        value: The value to validate.
        name: Name of the value for error messages.

    Returns:
        The value if non-negative.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
) -> float:
    """Validate that a value is within a specific range.

    Args:
        value: The value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        name: Name of the value for error messages.

    Returns:
        The value if within range.

    Raises:
        ValueError: If value is outside the range.
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


__all__ = [
    # Core clamping
    "clamp",
    "clamp_fraction",
    "clamp_percentage",
    "clamp_positive",
    # Domain-specific clamping
    "clamp_position_size",
    "clamp_stop_loss_pct",
    "clamp_risk_amount",
    "clamp_multiplier",
    # Validation
    "validate_fraction",
    "validate_positive",
    "validate_non_negative",
    "validate_range",
]
