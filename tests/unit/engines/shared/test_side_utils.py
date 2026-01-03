"""Unit tests for side_utils module.

Tests the side conversion and checking utilities used across engines.
"""

import pytest

from src.engines.shared.models import PositionSide
from src.engines.shared.side_utils import (
    get_position_side,
    is_long,
    is_short,
    opposite_side,
    opposite_side_string,
    to_position_side,
    to_side_string,
)


class TestToSideString:
    """Tests for to_side_string function."""

    def test_position_side_long_returns_long(self) -> None:
        # Arrange
        side = PositionSide.LONG

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "long"

    def test_position_side_short_returns_short(self) -> None:
        # Arrange
        side = PositionSide.SHORT

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "short"

    def test_string_long_lowercase(self) -> None:
        # Arrange
        side = "long"

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "long"

    def test_string_long_uppercase(self) -> None:
        # Arrange
        side = "LONG"

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "long"

    def test_string_short_mixed_case(self) -> None:
        # Arrange
        side = "Short"

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "short"

    def test_buy_maps_to_long(self) -> None:
        # Arrange
        side = "BUY"

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "long"

    def test_sell_maps_to_short(self) -> None:
        # Arrange
        side = "sell"

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "short"

    def test_object_with_value_attribute(self) -> None:
        # Arrange
        class MockEnum:
            value = "long"

        side = MockEnum()

        # Act
        result = to_side_string(side)

        # Assert
        assert result == "long"

    def test_invalid_string_raises_value_error(self) -> None:
        # Arrange
        side = "invalid"

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid side string"):
            to_side_string(side)

    def test_invalid_type_raises_value_error(self) -> None:
        # Arrange
        side = 123

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot convert"):
            to_side_string(side)


class TestToPositionSide:
    """Tests for to_position_side function."""

    def test_position_side_long_returns_same(self) -> None:
        # Arrange
        side = PositionSide.LONG

        # Act
        result = to_position_side(side)

        # Assert
        assert result is PositionSide.LONG

    def test_string_long_returns_position_side_long(self) -> None:
        # Arrange
        side = "long"

        # Act
        result = to_position_side(side)

        # Assert
        assert result == PositionSide.LONG

    def test_string_short_returns_position_side_short(self) -> None:
        # Arrange
        side = "SHORT"

        # Act
        result = to_position_side(side)

        # Assert
        assert result == PositionSide.SHORT

    def test_buy_returns_position_side_long(self) -> None:
        # Arrange
        side = "buy"

        # Act
        result = to_position_side(side)

        # Assert
        assert result == PositionSide.LONG


class TestIsLong:
    """Tests for is_long function."""

    def test_position_side_long_returns_true(self) -> None:
        assert is_long(PositionSide.LONG) is True

    def test_position_side_short_returns_false(self) -> None:
        assert is_long(PositionSide.SHORT) is False

    def test_string_long_returns_true(self) -> None:
        assert is_long("long") is True

    def test_string_short_returns_false(self) -> None:
        assert is_long("short") is False

    def test_buy_returns_true(self) -> None:
        assert is_long("BUY") is True


class TestIsShort:
    """Tests for is_short function."""

    def test_position_side_short_returns_true(self) -> None:
        assert is_short(PositionSide.SHORT) is True

    def test_position_side_long_returns_false(self) -> None:
        assert is_short(PositionSide.LONG) is False

    def test_string_short_returns_true(self) -> None:
        assert is_short("short") is True

    def test_sell_returns_true(self) -> None:
        assert is_short("SELL") is True


class TestOppositeSide:
    """Tests for opposite_side function."""

    def test_long_returns_short(self) -> None:
        # Arrange
        side = PositionSide.LONG

        # Act
        result = opposite_side(side)

        # Assert
        assert result == PositionSide.SHORT

    def test_short_returns_long(self) -> None:
        # Arrange
        side = PositionSide.SHORT

        # Act
        result = opposite_side(side)

        # Assert
        assert result == PositionSide.LONG

    def test_string_long_returns_short(self) -> None:
        assert opposite_side("long") == PositionSide.SHORT


class TestOppositeSideString:
    """Tests for opposite_side_string function."""

    def test_long_returns_short_string(self) -> None:
        assert opposite_side_string(PositionSide.LONG) == "short"

    def test_short_returns_long_string(self) -> None:
        assert opposite_side_string(PositionSide.SHORT) == "long"

    def test_string_input_works(self) -> None:
        assert opposite_side_string("long") == "short"


class TestGetPositionSide:
    """Tests for get_position_side function."""

    def test_none_returns_default(self) -> None:
        # Arrange
        position = None

        # Act
        result = get_position_side(position)

        # Assert
        assert result == "long"

    def test_none_with_custom_default(self) -> None:
        # Arrange
        position = None

        # Act
        result = get_position_side(position, default="short")

        # Assert
        assert result == "short"

    def test_object_without_side_returns_default(self) -> None:
        # Arrange
        class MockPosition:
            pass

        position = MockPosition()

        # Act
        result = get_position_side(position)

        # Assert
        assert result == "long"

    def test_object_with_none_side_returns_default(self) -> None:
        # Arrange
        class MockPosition:
            side = None

        position = MockPosition()

        # Act
        result = get_position_side(position)

        # Assert
        assert result == "long"

    def test_object_with_position_side_enum(self) -> None:
        # Arrange
        class MockPosition:
            side = PositionSide.SHORT

        position = MockPosition()

        # Act
        result = get_position_side(position)

        # Assert
        assert result == "short"

    def test_object_with_string_side(self) -> None:
        # Arrange
        class MockPosition:
            side = "LONG"

        position = MockPosition()

        # Act
        result = get_position_side(position)

        # Assert
        assert result == "long"
