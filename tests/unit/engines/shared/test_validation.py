"""Unit tests for validation module.

Tests the input validation utilities for financial calculations.
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

    def test_valid_price_passes(self) -> None:
        # Arrange & Act & Assert - no exception raised
        validate_price(100.0, "test_price")

    def test_zero_price_raises(self) -> None:
        # Arrange
        price = 0.0

        # Act & Assert
        with pytest.raises(ValueError, match="must be positive"):
            validate_price(price, "test_price")

    def test_negative_price_raises(self) -> None:
        # Arrange
        price = -10.0

        # Act & Assert
        with pytest.raises(ValueError, match="must be positive"):
            validate_price(price, "test_price")

    def test_nan_raises(self) -> None:
        # Arrange
        price = float("nan")

        # Act & Assert
        with pytest.raises(ValueError, match="must be finite"):
            validate_price(price, "test_price")

    def test_infinity_raises(self) -> None:
        # Arrange
        price = float("inf")

        # Act & Assert
        with pytest.raises(ValueError, match="must be finite"):
            validate_price(price, "test_price")

    def test_negative_infinity_raises(self) -> None:
        # Arrange
        price = float("-inf")

        # Act & Assert
        with pytest.raises(ValueError, match="must be positive"):
            validate_price(price, "test_price")

    def test_non_numeric_raises(self) -> None:
        # Arrange
        price = "100"

        # Act & Assert
        with pytest.raises(ValueError, match="must be a number"):
            validate_price(price, "test_price")  # type: ignore[arg-type]

    def test_integer_price_valid(self) -> None:
        # Arrange & Act & Assert - no exception raised
        validate_price(100, "test_price")


class TestValidateNotional:
    """Tests for validate_notional function."""

    def test_positive_notional_passes(self) -> None:
        validate_notional(1000.0, "test_notional")

    def test_zero_notional_passes(self) -> None:
        # Zero is allowed for notional
        validate_notional(0.0, "test_notional")

    def test_negative_notional_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_notional(-100.0, "test_notional")

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            validate_notional(float("nan"), "test_notional")

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            validate_notional("1000", "test_notional")  # type: ignore[arg-type]


class TestValidateFraction:
    """Tests for validate_fraction function."""

    def test_valid_fraction_passes(self) -> None:
        validate_fraction(0.5, "test_fraction")

    def test_zero_allowed_by_default(self) -> None:
        validate_fraction(0.0, "test_fraction")

    def test_one_is_valid(self) -> None:
        validate_fraction(1.0, "test_fraction")

    def test_zero_not_allowed_when_specified(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            validate_fraction(0.0, "test_fraction", allow_zero=False)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_fraction(-0.1, "test_fraction")

    def test_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_fraction(1.5, "test_fraction")

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            validate_fraction(float("nan"), "test_fraction")


class TestIsValidPrice:
    """Tests for is_valid_price function."""

    def test_valid_price_returns_true(self) -> None:
        assert is_valid_price(100.0) is True

    def test_zero_returns_false(self) -> None:
        assert is_valid_price(0.0) is False

    def test_negative_returns_false(self) -> None:
        assert is_valid_price(-10.0) is False

    def test_nan_returns_false(self) -> None:
        assert is_valid_price(float("nan")) is False

    def test_infinity_returns_false(self) -> None:
        assert is_valid_price(float("inf")) is False


class TestIsValidFraction:
    """Tests for is_valid_fraction function."""

    def test_valid_fraction_returns_true(self) -> None:
        assert is_valid_fraction(0.5) is True

    def test_zero_valid_by_default(self) -> None:
        assert is_valid_fraction(0.0) is True

    def test_zero_invalid_when_not_allowed(self) -> None:
        assert is_valid_fraction(0.0, allow_zero=False) is False

    def test_negative_returns_false(self) -> None:
        assert is_valid_fraction(-0.1) is False

    def test_greater_than_one_returns_false(self) -> None:
        assert is_valid_fraction(1.5) is False


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self) -> None:
        # Arrange
        numerator = 10.0
        denominator = 2.0

        # Act
        result = safe_divide(numerator, denominator)

        # Assert
        assert result == 5.0

    def test_division_by_zero_returns_fallback(self) -> None:
        # Arrange
        numerator = 10.0
        denominator = 0.0

        # Act
        result = safe_divide(numerator, denominator)

        # Assert
        assert result == 0.0

    def test_custom_fallback(self) -> None:
        # Arrange
        numerator = 10.0
        denominator = 0.0
        fallback = -1.0

        # Act
        result = safe_divide(numerator, denominator, fallback=fallback)

        # Assert
        assert result == -1.0

    def test_near_zero_denominator_returns_fallback(self) -> None:
        # Arrange
        numerator = 10.0
        denominator = 1e-12  # Less than EPSILON

        # Act
        result = safe_divide(numerator, denominator)

        # Assert
        assert result == 0.0

    def test_nan_numerator_returns_fallback(self) -> None:
        # Arrange
        numerator = float("nan")
        denominator = 2.0

        # Act
        result = safe_divide(numerator, denominator)

        # Assert
        assert result == 0.0

    def test_nan_denominator_returns_fallback(self) -> None:
        # Arrange
        numerator = 10.0
        denominator = float("nan")

        # Act
        result = safe_divide(numerator, denominator)

        # Assert
        assert result == 0.0

    def test_infinity_numerator_returns_fallback(self) -> None:
        # Arrange
        numerator = float("inf")
        denominator = 2.0

        # Act
        result = safe_divide(numerator, denominator)

        # Assert
        assert result == 0.0

    def test_result_overflow_returns_fallback(self) -> None:
        # Arrange - very large number divided by very small
        numerator = 1e308
        denominator = 1e-308

        # Act
        result = safe_divide(numerator, denominator)

        # Assert - would overflow to infinity
        assert result == 0.0


class TestIsPositionFullyClosed:
    """Tests for is_position_fully_closed function."""

    def test_zero_current_size_is_closed(self) -> None:
        # Arrange
        current_size = 0.0
        original_size = 1.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert
        assert result is True

    def test_near_zero_current_size_is_closed(self) -> None:
        # Arrange
        current_size = 1e-12
        original_size = 1.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert
        assert result is True

    def test_half_position_not_closed(self) -> None:
        # Arrange
        current_size = 0.5
        original_size = 1.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert
        assert result is False

    def test_full_position_not_closed(self) -> None:
        # Arrange
        current_size = 1.0
        original_size = 1.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert
        assert result is False

    def test_near_zero_ratio_is_closed(self) -> None:
        # Arrange - ratio is near zero
        current_size = 1e-12
        original_size = 100.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert
        assert result is True

    def test_zero_original_size_with_zero_current_is_closed(self) -> None:
        # Arrange - edge case: original_size is zero
        current_size = 0.0
        original_size = 0.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert - zero current size should be closed regardless
        assert result is True

    def test_zero_original_size_with_nonzero_current_not_closed(self) -> None:
        # Arrange - edge case: original_size is zero but current is not
        current_size = 0.5
        original_size = 0.0

        # Act
        result = is_position_fully_closed(current_size, original_size)

        # Assert - non-zero current size should not be closed
        assert result is False

    def test_custom_epsilon(self) -> None:
        # Arrange
        current_size = 0.001
        original_size = 1.0
        custom_epsilon = 0.01

        # Act
        result = is_position_fully_closed(current_size, original_size, epsilon=custom_epsilon)

        # Assert - 0.001 < 0.01, so should be closed
        assert result is True


class TestClampFraction:
    """Tests for clamp_fraction function."""

    def test_value_in_range_unchanged(self) -> None:
        assert clamp_fraction(0.5) == 0.5

    def test_negative_clamped_to_zero(self) -> None:
        assert clamp_fraction(-0.1) == 0.0

    def test_greater_than_one_clamped_to_one(self) -> None:
        assert clamp_fraction(1.5) == 1.0

    def test_exactly_zero(self) -> None:
        assert clamp_fraction(0.0) == 0.0

    def test_exactly_one(self) -> None:
        assert clamp_fraction(1.0) == 1.0

    def test_nan_returns_min(self) -> None:
        assert clamp_fraction(float("nan")) == 0.0

    def test_infinity_returns_max(self) -> None:
        # infinity is not finite, so returns min_val
        assert clamp_fraction(float("inf")) == 0.0

    def test_custom_range(self) -> None:
        assert clamp_fraction(0.3, min_val=0.2, max_val=0.8) == 0.3
        assert clamp_fraction(0.1, min_val=0.2, max_val=0.8) == 0.2
        assert clamp_fraction(0.9, min_val=0.2, max_val=0.8) == 0.8


class TestValidateParallelLists:
    """Tests for validate_parallel_lists function."""

    def test_equal_length_passes(self) -> None:
        # Arrange
        list1 = [1, 2, 3]
        list2 = ["a", "b", "c"]

        # Act & Assert - no exception
        validate_parallel_lists(list1, list2)

    def test_empty_lists_pass(self) -> None:
        validate_parallel_lists([], [])

    def test_mismatched_length_raises(self) -> None:
        # Arrange
        list1 = [1, 2]
        list2 = [1, 2, 3]

        # Act & Assert
        with pytest.raises(ValueError, match="length.*must match"):
            validate_parallel_lists(list1, list2, "list1", "list2")

    def test_custom_names_in_error(self) -> None:
        # Arrange
        list1 = [1]
        list2 = [1, 2]

        # Act & Assert
        with pytest.raises(ValueError, match="exit_targets.*exit_sizes"):
            validate_parallel_lists(list1, list2, "exit_targets", "exit_sizes")


class TestEpsilon:
    """Tests for EPSILON constant."""

    def test_epsilon_is_small_positive(self) -> None:
        assert EPSILON > 0
        assert EPSILON < 1e-6

    def test_epsilon_useful_for_comparisons(self) -> None:
        # Two floats that should be "equal"
        a = 0.1 + 0.2
        b = 0.3
        # They're not exactly equal due to floating point
        assert a != b
        # But they're within epsilon
        assert abs(a - b) < EPSILON
