"""Tests for side utility functions.

These utilities ensure consistent side handling across backtest and live engines.
"""

import pytest

from src.engines.shared.models import PositionSide
from src.engines.shared.side_utils import (
    is_long,
    is_short,
    opposite_side,
    opposite_side_string,
    to_position_side,
    to_side_string,
)


class TestToSideString:
    """Tests for to_side_string function."""

    def test_position_side_long(self):
        """Convert PositionSide.LONG to string."""
        assert to_side_string(PositionSide.LONG) == "long"

    def test_position_side_short(self):
        """Convert PositionSide.SHORT to string."""
        assert to_side_string(PositionSide.SHORT) == "short"

    def test_string_long_lowercase(self):
        """Convert lowercase 'long' string."""
        assert to_side_string("long") == "long"

    def test_string_short_lowercase(self):
        """Convert lowercase 'short' string."""
        assert to_side_string("short") == "short"

    def test_string_long_uppercase(self):
        """Convert uppercase 'LONG' string."""
        assert to_side_string("LONG") == "long"

    def test_string_short_uppercase(self):
        """Convert uppercase 'SHORT' string."""
        assert to_side_string("SHORT") == "short"

    def test_string_mixed_case(self):
        """Convert mixed case strings."""
        assert to_side_string("Long") == "long"
        assert to_side_string("ShOrT") == "short"

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid side string"):
            to_side_string("buy")

    def test_invalid_type_raises(self):
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            to_side_string(123)  # type: ignore

    def test_object_with_value_attribute(self):
        """Object with .value attribute is converted."""

        class MockEnum:
            value = "long"

        assert to_side_string(MockEnum()) == "long"


class TestToPositionSide:
    """Tests for to_position_side function."""

    def test_position_side_returns_same(self):
        """PositionSide enum returns itself."""
        assert to_position_side(PositionSide.LONG) is PositionSide.LONG
        assert to_position_side(PositionSide.SHORT) is PositionSide.SHORT

    def test_string_long(self):
        """String 'long' converts to enum."""
        assert to_position_side("long") == PositionSide.LONG

    def test_string_short(self):
        """String 'short' converts to enum."""
        assert to_position_side("short") == PositionSide.SHORT

    def test_string_uppercase(self):
        """Uppercase strings convert correctly."""
        assert to_position_side("LONG") == PositionSide.LONG
        assert to_position_side("SHORT") == PositionSide.SHORT

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            to_position_side("invalid")


class TestIsLong:
    """Tests for is_long function."""

    def test_position_side_long(self):
        """PositionSide.LONG returns True."""
        assert is_long(PositionSide.LONG) is True

    def test_position_side_short(self):
        """PositionSide.SHORT returns False."""
        assert is_long(PositionSide.SHORT) is False

    def test_string_long(self):
        """String 'long' returns True."""
        assert is_long("long") is True
        assert is_long("LONG") is True

    def test_string_short(self):
        """String 'short' returns False."""
        assert is_long("short") is False


class TestIsShort:
    """Tests for is_short function."""

    def test_position_side_short(self):
        """PositionSide.SHORT returns True."""
        assert is_short(PositionSide.SHORT) is True

    def test_position_side_long(self):
        """PositionSide.LONG returns False."""
        assert is_short(PositionSide.LONG) is False

    def test_string_short(self):
        """String 'short' returns True."""
        assert is_short("short") is True
        assert is_short("SHORT") is True

    def test_string_long(self):
        """String 'long' returns False."""
        assert is_short("long") is False


class TestOppositeSide:
    """Tests for opposite_side function."""

    def test_long_returns_short(self):
        """Long side returns short."""
        assert opposite_side(PositionSide.LONG) == PositionSide.SHORT
        assert opposite_side("long") == PositionSide.SHORT

    def test_short_returns_long(self):
        """Short side returns long."""
        assert opposite_side(PositionSide.SHORT) == PositionSide.LONG
        assert opposite_side("short") == PositionSide.LONG


class TestOppositeSideString:
    """Tests for opposite_side_string function."""

    def test_long_returns_short_string(self):
        """Long side returns 'short'."""
        assert opposite_side_string(PositionSide.LONG) == "short"
        assert opposite_side_string("long") == "short"

    def test_short_returns_long_string(self):
        """Short side returns 'long'."""
        assert opposite_side_string(PositionSide.SHORT) == "long"
        assert opposite_side_string("short") == "long"


class TestEdgeCases:
    """Edge case tests for side utilities."""

    def test_whitespace_in_string_raises(self):
        """Strings with whitespace should raise."""
        with pytest.raises(ValueError):
            to_side_string(" long ")

    def test_empty_string_raises(self):
        """Empty string should raise."""
        with pytest.raises(ValueError):
            to_side_string("")

    def test_none_raises(self):
        """None should raise."""
        with pytest.raises(ValueError):
            to_side_string(None)  # type: ignore

    def test_consistency_long(self):
        """All functions agree on long."""
        sides = [PositionSide.LONG, "long", "LONG", "Long"]
        for side in sides:
            assert to_side_string(side) == "long"
            assert to_position_side(side) == PositionSide.LONG
            assert is_long(side) is True
            assert is_short(side) is False
            assert opposite_side(side) == PositionSide.SHORT
            assert opposite_side_string(side) == "short"

    def test_consistency_short(self):
        """All functions agree on short."""
        sides = [PositionSide.SHORT, "short", "SHORT", "Short"]
        for side in sides:
            assert to_side_string(side) == "short"
            assert to_position_side(side) == PositionSide.SHORT
            assert is_long(side) is False
            assert is_short(side) is True
            assert opposite_side(side) == PositionSide.LONG
            assert opposite_side_string(side) == "long"
