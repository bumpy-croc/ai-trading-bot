"""Tests for validation utility functions.

These utilities ensure consistent input validation across trading engines.
"""

import math

import pytest

from src.engines.shared.validation import (
    EPSILON,
    clamp_fraction,
    is_position_fully_closed,
    is_valid_fraction,
    is_valid_price,
    safe_divide,
    validate_fraction,
    validate_notional,
    validate_parallel_lists,
    validate_price,
)


class TestValidatePrice:
    """Tests for validate_price function."""

    def test_valid_price(self):
        """Valid prices should not raise."""
        validate_price(100.0)
        validate_price(0.01)
        validate_price(1)  # Integer should work
        validate_price(1e6)

    def test_zero_price_raises(self):
        """Zero price should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_price(0)

    def test_negative_price_raises(self):
        """Negative price should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_price(-100.0)

    def test_nan_price_raises(self):
        """NaN price should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_price(float("nan"))

    def test_inf_price_raises(self):
        """Infinite price should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_price(float("inf"))

    def test_custom_name_in_error(self):
        """Custom field name should appear in error."""
        with pytest.raises(ValueError, match="entry_price"):
            validate_price(-1, "entry_price")

    def test_invalid_type_raises(self):
        """Non-numeric type should raise ValueError."""
        with pytest.raises(ValueError, match="must be a number"):
            validate_price("100")  # type: ignore


class TestValidateNotional:
    """Tests for validate_notional function."""

    def test_valid_notional(self):
        """Valid notional values should not raise."""
        validate_notional(1000.0)
        validate_notional(0.0)  # Zero is allowed
        validate_notional(1)  # Integer should work

    def test_negative_notional_raises(self):
        """Negative notional should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_notional(-100.0)

    def test_nan_notional_raises(self):
        """NaN notional should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_notional(float("nan"))

    def test_inf_notional_raises(self):
        """Infinite notional should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_notional(float("inf"))


class TestValidateFraction:
    """Tests for validate_fraction function."""

    def test_valid_fractions(self):
        """Valid fractions should not raise."""
        validate_fraction(0.0)
        validate_fraction(0.5)
        validate_fraction(1.0)

    def test_negative_fraction_raises(self):
        """Negative fraction should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_fraction(-0.1)

    def test_over_one_fraction_raises(self):
        """Fraction over 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_fraction(1.1)

    def test_zero_not_allowed(self):
        """Zero fraction raises if allow_zero=False."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_fraction(0.0, allow_zero=False)

    def test_nan_fraction_raises(self):
        """NaN fraction should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_fraction(float("nan"))


class TestIsValidPrice:
    """Tests for is_valid_price function."""

    def test_valid_prices_return_true(self):
        """Valid prices return True."""
        assert is_valid_price(100.0) is True
        assert is_valid_price(0.01) is True

    def test_invalid_prices_return_false(self):
        """Invalid prices return False."""
        assert is_valid_price(0) is False
        assert is_valid_price(-1) is False
        assert is_valid_price(float("nan")) is False
        assert is_valid_price(float("inf")) is False


class TestIsValidFraction:
    """Tests for is_valid_fraction function."""

    def test_valid_fractions_return_true(self):
        """Valid fractions return True."""
        assert is_valid_fraction(0.0) is True
        assert is_valid_fraction(0.5) is True
        assert is_valid_fraction(1.0) is True

    def test_invalid_fractions_return_false(self):
        """Invalid fractions return False."""
        assert is_valid_fraction(-0.1) is False
        assert is_valid_fraction(1.5) is False
        assert is_valid_fraction(float("nan")) is False

    def test_zero_not_allowed(self):
        """Zero returns False when allow_zero=False."""
        assert is_valid_fraction(0.0, allow_zero=False) is False
        assert is_valid_fraction(0.01, allow_zero=False) is True


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self):
        """Normal division works correctly."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(7, 2) == 3.5
        assert safe_divide(-10, 2) == -5.0

    def test_zero_denominator_returns_fallback(self):
        """Division by zero returns fallback."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, fallback=-1.0) == -1.0

    def test_near_zero_denominator_returns_fallback(self):
        """Near-zero denominator returns fallback."""
        assert safe_divide(10, 1e-12) == 0.0
        assert safe_divide(10, EPSILON / 2) == 0.0

    def test_nan_numerator_returns_fallback(self):
        """NaN numerator returns fallback."""
        assert safe_divide(float("nan"), 2) == 0.0

    def test_nan_denominator_returns_fallback(self):
        """NaN denominator returns fallback."""
        assert safe_divide(10, float("nan")) == 0.0

    def test_inf_numerator_returns_fallback(self):
        """Infinite numerator returns fallback."""
        assert safe_divide(float("inf"), 2) == 0.0

    def test_custom_fallback(self):
        """Custom fallback value is used."""
        assert safe_divide(10, 0, fallback=42.0) == 42.0


class TestIsPositionFullyClosed:
    """Tests for is_position_fully_closed function."""

    def test_zero_size_is_closed(self):
        """Zero current size is fully closed."""
        assert is_position_fully_closed(0.0, 1.0) is True

    def test_near_zero_size_is_closed(self):
        """Near-zero current size is fully closed."""
        assert is_position_fully_closed(1e-12, 1.0) is True
        assert is_position_fully_closed(EPSILON / 2, 1.0) is True

    def test_partial_size_not_closed(self):
        """Partial current size is not fully closed."""
        assert is_position_fully_closed(0.5, 1.0) is False
        assert is_position_fully_closed(0.01, 1.0) is False

    def test_full_size_not_closed(self):
        """Full current size is not fully closed."""
        assert is_position_fully_closed(1.0, 1.0) is False

    def test_small_original_size(self):
        """Works with small original sizes."""
        assert is_position_fully_closed(0.0, 0.001) is True
        assert is_position_fully_closed(0.0005, 0.001) is False


class TestClampFraction:
    """Tests for clamp_fraction function."""

    def test_value_in_range_unchanged(self):
        """Values in range are unchanged."""
        assert clamp_fraction(0.5) == 0.5
        assert clamp_fraction(0.0) == 0.0
        assert clamp_fraction(1.0) == 1.0

    def test_negative_clamped_to_zero(self):
        """Negative values are clamped to zero."""
        assert clamp_fraction(-0.1) == 0.0
        assert clamp_fraction(-100) == 0.0

    def test_over_one_clamped_to_one(self):
        """Values over 1 are clamped to 1."""
        assert clamp_fraction(1.5) == 1.0
        assert clamp_fraction(100) == 1.0

    def test_nan_returns_min(self):
        """NaN returns minimum value."""
        assert clamp_fraction(float("nan")) == 0.0

    def test_custom_range(self):
        """Custom range works correctly."""
        assert clamp_fraction(5, min_val=1, max_val=10) == 5
        assert clamp_fraction(0, min_val=1, max_val=10) == 1
        assert clamp_fraction(15, min_val=1, max_val=10) == 10


class TestValidateParallelLists:
    """Tests for validate_parallel_lists function."""

    def test_equal_length_lists(self):
        """Equal length lists should not raise."""
        validate_parallel_lists([1, 2, 3], [4, 5, 6])
        validate_parallel_lists([], [])
        validate_parallel_lists([1], [2])

    def test_unequal_length_lists_raises(self):
        """Unequal length lists should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            validate_parallel_lists([1, 2], [3])

    def test_custom_names_in_error(self):
        """Custom names should appear in error."""
        with pytest.raises(ValueError, match="exit_targets.*exit_sizes"):
            validate_parallel_lists(
                [1, 2, 3], [4, 5], name1="exit_targets", name2="exit_sizes"
            )


class TestEpsilonConsistency:
    """Tests for EPSILON constant."""

    def test_epsilon_value(self):
        """EPSILON has expected value."""
        assert EPSILON == 1e-9

    def test_epsilon_is_small_enough(self):
        """EPSILON is small enough for financial calculations."""
        # Typical financial calculation should not be affected
        result = 100.0 + EPSILON
        assert result != 100.0  # EPSILON is detectable
        assert result == pytest.approx(100.0, abs=1e-6)  # But negligible

    def test_epsilon_catches_near_zero(self):
        """EPSILON correctly identifies near-zero values."""
        assert abs(EPSILON / 2) < EPSILON
        assert abs(EPSILON * 2) > EPSILON
