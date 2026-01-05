"""Comprehensive edge case tests for validation helper methods.

These helpers are critical for defensive validation in the position_management
module. This test suite ensures they handle all edge cases correctly.
"""

import math

import pytest

from src.position_management.mfe_mae_analyzer import MFEMAEAnalyzer
from src.position_management.partial_manager import PartialExitPolicy, PositionState
from src.position_management.correlation_engine import CorrelationEngine


class TestSafeFloat:
    """Test _safe_float helper handles all edge cases correctly."""

    @pytest.fixture
    def analyzer(self):
        return MFEMAEAnalyzer()

    def test_none_returns_default(self, analyzer):
        """None should return the default value."""
        assert analyzer._safe_float(None) == 0.0
        assert analyzer._safe_float(None, default=5.0) == 5.0

    def test_valid_numbers_converted(self, analyzer):
        """Valid numbers should be converted to float correctly."""
        assert analyzer._safe_float(42) == 42.0
        assert analyzer._safe_float(3.14) == 3.14
        assert analyzer._safe_float("2.5") == 2.5
        assert analyzer._safe_float("100") == 100.0

    def test_nan_returns_default(self, analyzer):
        """NaN values should return default, not propagate."""
        assert analyzer._safe_float(float("nan")) == 0.0
        assert analyzer._safe_float(float("nan"), default=1.0) == 1.0
        assert analyzer._safe_float("NaN") == 0.0

    def test_infinity_returns_default(self, analyzer):
        """Infinity values should return default, not propagate."""
        assert analyzer._safe_float(float("inf")) == 0.0
        assert analyzer._safe_float(float("-inf")) == 0.0
        assert analyzer._safe_float(float("inf"), default=99.0) == 99.0

    def test_boolean_true_converts_to_one(self, analyzer):
        """Boolean True should convert to 1.0 (not treated as falsy).

        This is the expected behavior after fixing the 'value or default' pattern.
        """
        result = analyzer._safe_float(True)
        assert result == 1.0
        assert isinstance(result, float)

    def test_boolean_false_converts_to_zero(self, analyzer):
        """Boolean False should convert to 0.0 (not treated as falsy).

        This is the expected behavior after fixing the 'value or default' pattern.
        """
        result = analyzer._safe_float(False)
        assert result == 0.0
        assert isinstance(result, float)

    def test_negative_zero(self, analyzer):
        """Negative zero (-0.0) is a valid float and sign is preserved.

        Python preserves the sign of -0.0 through float conversion.
        This is IEEE 754 compliant behavior.
        """
        result = analyzer._safe_float(-0.0)
        assert result == 0.0
        # Sign is preserved: -0.0 is not the same as +0.0 in IEEE 754
        assert math.copysign(1, result) == -1.0  # Negative zero preserved

    def test_very_large_numbers(self, analyzer):
        """Very large finite numbers should be preserved."""
        large = 1e308
        result = analyzer._safe_float(large)
        assert result == large

    def test_very_small_numbers(self, analyzer):
        """Very small finite numbers should be preserved."""
        small = 1e-308
        result = analyzer._safe_float(small)
        assert result == small

    def test_invalid_strings_return_default(self, analyzer):
        """Invalid string values should return default."""
        assert analyzer._safe_float("hello") == 0.0
        assert analyzer._safe_float("") == 0.0
        assert analyzer._safe_float("abc123") == 0.0

    def test_invalid_types_return_default(self, analyzer):
        """Invalid types should return default gracefully."""
        assert analyzer._safe_float([]) == 0.0
        assert analyzer._safe_float({}) == 0.0
        assert analyzer._safe_float(object()) == 0.0

    def test_custom_default_preserved(self, analyzer):
        """Custom default values should be used correctly."""
        assert analyzer._safe_float(None, default=-1.0) == -1.0
        assert analyzer._safe_float(float("nan"), default=42.0) == 42.0


class TestSafeSizeAdd:
    """Test _safe_size_add helper validates position sizes correctly."""

    def test_valid_sizes_are_added(self):
        """Valid numeric sizes should be added to total."""
        result = CorrelationEngine._safe_size_add(10.0, 5.0)
        assert result == 15.0
        assert isinstance(result, float)

    def test_integer_sizes_converted(self):
        """Integer sizes should be converted to float."""
        result = CorrelationEngine._safe_size_add(10.0, 5)
        assert result == 15.0

    def test_negative_size_rejected(self):
        """Negative sizes should be rejected, total unchanged."""
        result = CorrelationEngine._safe_size_add(10.0, -5.0)
        assert result == 10.0  # Unchanged

    def test_nan_size_rejected(self):
        """NaN sizes should be rejected, total unchanged."""
        result = CorrelationEngine._safe_size_add(10.0, float("nan"))
        assert result == 10.0  # Unchanged

    def test_infinity_size_rejected(self):
        """Infinity sizes should be rejected, total unchanged."""
        result = CorrelationEngine._safe_size_add(10.0, float("inf"))
        assert result == 10.0  # Unchanged
        result = CorrelationEngine._safe_size_add(10.0, float("-inf"))
        assert result == 10.0  # Unchanged

    def test_zero_size_accepted(self):
        """Zero is a valid size (no position), should be accepted."""
        result = CorrelationEngine._safe_size_add(10.0, 0.0)
        assert result == 10.0

    def test_invalid_type_rejected(self):
        """Non-numeric types should be rejected, total unchanged."""
        result = CorrelationEngine._safe_size_add(10.0, "invalid")
        assert result == 10.0  # Unchanged
        result = CorrelationEngine._safe_size_add(10.0, None)
        assert result == 10.0  # Unchanged
        result = CorrelationEngine._safe_size_add(10.0, [])
        assert result == 10.0  # Unchanged

    def test_boolean_true_accepted_as_int(self):
        """Boolean True is accepted because bool is a subclass of int in Python.

        isinstance(True, int | float) returns True since bool inherits from int.
        True is converted to 1.0, which is a valid size.
        """
        result = CorrelationEngine._safe_size_add(10.0, True)
        assert result == 11.0  # 10.0 + 1.0 (True == 1)

    def test_very_large_size_accepted(self):
        """Large but finite sizes should be accepted."""
        result = CorrelationEngine._safe_size_add(0.0, 1e10)
        assert result == 1e10

    def test_very_small_size_accepted(self):
        """Small but finite sizes should be accepted."""
        result = CorrelationEngine._safe_size_add(0.0, 1e-10)
        assert result == 1e-10


class TestValidateFraction:
    """Test _validate_fraction helper validates fraction parameters correctly."""

    @pytest.fixture
    def policy(self):
        return PartialExitPolicy(
            exit_targets=[0.01, 0.02, 0.03],
            exit_sizes=[0.25, 0.25, 0.25],
        )

    def test_valid_fraction_accepted(self, policy):
        """Valid fractions between 0 and 1 should be accepted."""
        assert policy._validate_fraction(0.0) is True
        assert policy._validate_fraction(0.5) is True
        assert policy._validate_fraction(1.0) is True
        assert policy._validate_fraction(0.0001) is True
        assert policy._validate_fraction(0.9999) is True

    def test_negative_fraction_rejected(self, policy):
        """Negative fractions should be rejected."""
        assert policy._validate_fraction(-0.01) is False
        assert policy._validate_fraction(-1.0) is False
        assert policy._validate_fraction(-100.0) is False

    def test_fraction_greater_than_one_rejected(self, policy):
        """Fractions greater than 1.0 should be rejected."""
        assert policy._validate_fraction(1.001) is False
        assert policy._validate_fraction(2.0) is False
        assert policy._validate_fraction(100.0) is False

    def test_nan_fraction_rejected(self, policy):
        """NaN fractions should be rejected."""
        assert policy._validate_fraction(float("nan")) is False

    def test_infinity_fraction_rejected(self, policy):
        """Infinity fractions should be rejected."""
        assert policy._validate_fraction(float("inf")) is False
        assert policy._validate_fraction(float("-inf")) is False

    def test_invalid_types_rejected(self, policy):
        """Non-numeric types should be rejected."""
        assert policy._validate_fraction(None) is False
        assert policy._validate_fraction("0.5") is False
        assert policy._validate_fraction([0.5]) is False
        assert policy._validate_fraction({}) is False

    def test_integer_accepted(self, policy):
        """Integer values should be accepted (can be converted to float)."""
        assert policy._validate_fraction(0) is True
        assert policy._validate_fraction(1) is True

    def test_boolean_accepted_as_int_subclass(self, policy):
        """Boolean values are accepted because bool is a subclass of int in Python.

        isinstance(True, int | float) returns True since bool inherits from int.
        True (1.0) is within [0, 1] and False (0.0) is within [0, 1].
        """
        assert policy._validate_fraction(True) is True  # True == 1
        assert policy._validate_fraction(False) is True  # False == 0

    def test_boundary_values(self, policy):
        """Test exact boundary conditions."""
        # Exactly 0.0 should be accepted
        assert policy._validate_fraction(0.0) is True

        # Exactly 1.0 should be accepted
        assert policy._validate_fraction(1.0) is True

        # Just above 0.0 should be accepted
        assert policy._validate_fraction(1e-10) is True

        # Just below 1.0 should be accepted
        assert policy._validate_fraction(0.9999999999) is True

    def test_very_small_negative_rejected(self, policy):
        """Very small negative values should be rejected."""
        assert policy._validate_fraction(-1e-10) is False
        assert policy._validate_fraction(-0.0000001) is False

    def test_very_small_over_one_rejected(self, policy):
        """Values just over 1.0 should be rejected."""
        assert policy._validate_fraction(1.0000000001) is False
        assert policy._validate_fraction(1 + 1e-10) is False

    def test_custom_param_name_in_logs(self, policy, caplog):
        """Custom parameter names should appear in log messages."""
        import logging

        caplog.set_level(logging.WARNING)

        policy._validate_fraction(-1.0, param_name="test_fraction")
        assert "test_fraction" in caplog.text

        caplog.clear()
        policy._validate_fraction(2.0, param_name="custom_param")
        assert "custom_param" in caplog.text


class TestValidationHelpersIntegration:
    """Integration tests for validation helpers working together."""

    def test_position_state_validation_integration(self):
        """PositionState.__post_init__ should validate all fields correctly."""
        # Valid state should work
        state = PositionState(
            entry_price=100.0,
            side="long",
            original_size=0.5,
            current_size=0.5,
        )
        assert state.entry_price == 100.0
        assert state.side == "long"

    def test_position_state_rejects_invalid_prices(self):
        """PositionState should reject invalid entry prices."""
        with pytest.raises(ValueError, match="entry_price must be finite"):
            PositionState(
                entry_price=float("nan"),
                side="long",
                original_size=0.5,
                current_size=0.5,
            )

        with pytest.raises(ValueError, match="entry_price must be finite"):
            PositionState(
                entry_price=float("inf"),
                side="long",
                original_size=0.5,
                current_size=0.5,
            )

        with pytest.raises(ValueError, match="entry_price must be finite"):
            PositionState(
                entry_price=0.0,
                side="long",
                original_size=0.5,
                current_size=0.5,
            )

        with pytest.raises(ValueError, match="entry_price must be finite"):
            PositionState(
                entry_price=-100.0,
                side="long",
                original_size=0.5,
                current_size=0.5,
            )

    def test_position_state_rejects_invalid_sizes(self):
        """PositionState should reject invalid size values."""
        with pytest.raises(ValueError, match="original_size must be positive"):
            PositionState(
                entry_price=100.0,
                side="long",
                original_size=0.0,
                current_size=0.5,
            )

        with pytest.raises(ValueError, match="original_size must be positive"):
            PositionState(
                entry_price=100.0,
                side="long",
                original_size=-0.5,
                current_size=0.5,
            )

        with pytest.raises(ValueError, match="current_size cannot be negative"):
            PositionState(
                entry_price=100.0,
                side="long",
                original_size=0.5,
                current_size=-0.1,
            )
