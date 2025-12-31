"""Validation utilities for financial calculations and position management.

This module provides helper functions for validating inputs to financial
calculations, preventing common errors like division by zero, NaN propagation,
and invalid position sizes.

Usage:
    from src.engines.shared.validation import (
        validate_price,
        validate_fraction,
        safe_divide,
        is_valid_fraction,
    )

    # Validate price before using in calculations
    validate_price(price, "entry_price")  # Raises ValueError if invalid

    # Safe division with fallback
    result = safe_divide(numerator, denominator, fallback=0.0)

    # Check if fraction is valid (0-1 range)
    if is_valid_fraction(position_size):
        ...
"""

from __future__ import annotations

import math
from typing import Optional

# Epsilon for floating-point comparisons in financial calculations
# This matches the EPSILON defined in models.py for consistency
EPSILON = 1e-9


def validate_price(price: float, name: str = "price") -> None:
    """Validate that a price is positive and finite.

    Args:
        price: The price value to validate.
        name: Name of the field for error messages.

    Raises:
        ValueError: If price is not positive or not finite.

    Examples:
        >>> validate_price(100.0, "entry_price")  # OK
        >>> validate_price(0, "entry_price")  # Raises ValueError
        >>> validate_price(float('nan'), "price")  # Raises ValueError
    """
    if not isinstance(price, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(price).__name__}")

    if price <= 0:
        raise ValueError(f"{name} must be positive, got {price}")

    if not math.isfinite(price):
        raise ValueError(f"{name} must be finite, got {price}")


def validate_notional(notional: float, name: str = "notional") -> None:
    """Validate that a notional value is non-negative and finite.

    Args:
        notional: The notional value to validate.
        name: Name of the field for error messages.

    Raises:
        ValueError: If notional is negative or not finite.

    Examples:
        >>> validate_notional(1000.0)  # OK
        >>> validate_notional(0.0)  # OK (zero is allowed)
        >>> validate_notional(-100.0)  # Raises ValueError
    """
    if not isinstance(notional, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(notional).__name__}")

    if notional < 0:
        raise ValueError(f"{name} must be non-negative, got {notional}")

    if not math.isfinite(notional):
        raise ValueError(f"{name} must be finite, got {notional}")


def validate_fraction(
    fraction: float, name: str = "fraction", allow_zero: bool = True
) -> None:
    """Validate that a fraction is in valid range [0, 1] and finite.

    Args:
        fraction: The fraction value to validate (0-1 range).
        name: Name of the field for error messages.
        allow_zero: If False, zero is not allowed.

    Raises:
        ValueError: If fraction is out of range or not finite.

    Examples:
        >>> validate_fraction(0.5)  # OK
        >>> validate_fraction(1.0)  # OK
        >>> validate_fraction(1.5)  # Raises ValueError
        >>> validate_fraction(0.0, allow_zero=False)  # Raises ValueError
    """
    if not isinstance(fraction, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(fraction).__name__}")

    if not math.isfinite(fraction):
        raise ValueError(f"{name} must be finite, got {fraction}")

    if not allow_zero and fraction <= 0:
        raise ValueError(f"{name} must be positive, got {fraction}")

    if fraction < 0 or fraction > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {fraction}")


def is_valid_price(price: float) -> bool:
    """Check if a price value is valid (positive and finite).

    Args:
        price: The price value to check.

    Returns:
        True if price is positive and finite, False otherwise.

    Examples:
        >>> is_valid_price(100.0)
        True
        >>> is_valid_price(0.0)
        False
        >>> is_valid_price(float('nan'))
        False
    """
    try:
        validate_price(price)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_fraction(fraction: float, allow_zero: bool = True) -> bool:
    """Check if a fraction value is valid (in 0-1 range and finite).

    Args:
        fraction: The fraction value to check.
        allow_zero: If False, zero is not considered valid.

    Returns:
        True if fraction is in valid range and finite, False otherwise.

    Examples:
        >>> is_valid_fraction(0.5)
        True
        >>> is_valid_fraction(1.5)
        False
        >>> is_valid_fraction(0.0, allow_zero=False)
        False
    """
    try:
        validate_fraction(fraction, allow_zero=allow_zero)
        return True
    except (ValueError, TypeError):
        return False


def safe_divide(
    numerator: float, denominator: float, fallback: float = 0.0
) -> float:
    """Safely divide two numbers, returning fallback if division is invalid.

    Handles division by zero, near-zero denominators, and NaN/Infinity results.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        fallback: Value to return if division is invalid.

    Returns:
        The division result, or fallback if invalid.

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, fallback=-1.0)
        -1.0
        >>> safe_divide(10, 1e-12)  # Near-zero denominator
        0.0
    """
    # Check for invalid inputs
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return fallback

    # Check for zero or near-zero denominator
    if abs(denominator) < EPSILON:
        return fallback

    result = numerator / denominator

    # Check for invalid result
    if not math.isfinite(result):
        return fallback

    return result


def is_position_fully_closed(
    current_size: float, original_size: float, epsilon: float = EPSILON
) -> bool:
    """Check if a position is fully closed (current_size near zero).

    Args:
        current_size: Current position size.
        original_size: Original position size (for ratio calculation).
        epsilon: Threshold for considering position closed.

    Returns:
        True if position is fully closed, False otherwise.

    Examples:
        >>> is_position_fully_closed(0.0, 1.0)
        True
        >>> is_position_fully_closed(1e-12, 1.0)
        True
        >>> is_position_fully_closed(0.5, 1.0)
        False
    """
    # Check absolute size
    if abs(current_size) < epsilon:
        return True

    # Check ratio (handles both small and large original sizes)
    if original_size > epsilon:
        ratio = current_size / original_size
        if abs(ratio) < epsilon:
            return True

    return False


def clamp_fraction(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a fraction to valid range.

    Args:
        value: The value to clamp.
        min_val: Minimum allowed value (default 0.0).
        max_val: Maximum allowed value (default 1.0).

    Returns:
        Value clamped to [min_val, max_val] range.

    Examples:
        >>> clamp_fraction(0.5)
        0.5
        >>> clamp_fraction(-0.1)
        0.0
        >>> clamp_fraction(1.5)
        1.0
    """
    if not math.isfinite(value):
        return min_val
    return max(min_val, min(max_val, value))


def validate_parallel_lists(
    list1: list, list2: list, name1: str = "list1", name2: str = "list2"
) -> None:
    """Validate that two parallel lists have the same length.

    Args:
        list1: First list.
        list2: Second list.
        name1: Name of first list for error messages.
        name2: Name of second list for error messages.

    Raises:
        ValueError: If lists have different lengths.

    Examples:
        >>> validate_parallel_lists([1, 2], [3, 4])  # OK
        >>> validate_parallel_lists([1, 2], [3])  # Raises ValueError
    """
    if len(list1) != len(list2):
        raise ValueError(
            f"{name1} length ({len(list1)}) must match "
            f"{name2} length ({len(list2)})"
        )
