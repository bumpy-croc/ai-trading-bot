"""Tests for shared validation utilities."""

from __future__ import annotations

import pytest

from src.engines.shared.validation import convert_exit_fraction_to_current


def test_convert_exit_fraction_to_current_valid() -> None:
    """Converts fraction of original to fraction of current size."""
    # Arrange
    exit_fraction_of_original = 0.25

    # Act
    exit_fraction_of_current = convert_exit_fraction_to_current(
        exit_fraction_of_original=exit_fraction_of_original,
        current_size=0.5,
        original_size=1.0,
    )

    # Assert
    assert exit_fraction_of_current == pytest.approx(0.5)


def test_convert_exit_fraction_to_current_allows_scaled_positions() -> None:
    """Scaled positions can exit more than original while staying within current size."""
    # Arrange
    exit_fraction_of_original = 1.2

    # Act
    exit_fraction_of_current = convert_exit_fraction_to_current(
        exit_fraction_of_original=exit_fraction_of_original,
        current_size=0.2,
        original_size=0.1,
    )

    # Assert
    assert exit_fraction_of_current == pytest.approx(0.6)


def test_convert_exit_fraction_to_current_returns_none_for_closed_position() -> None:
    """Closed positions return None to prevent division errors."""
    # Arrange
    exit_fraction_of_original = 0.25

    # Act
    exit_fraction_of_current = convert_exit_fraction_to_current(
        exit_fraction_of_original=exit_fraction_of_original,
        current_size=0.0,
        original_size=1.0,
    )

    # Assert
    assert exit_fraction_of_current is None


def test_convert_exit_fraction_to_current_returns_none_for_invalid_fraction() -> None:
    """Invalid fractions return None."""
    # Arrange
    exit_fraction_of_original = -0.1

    # Act
    exit_fraction_of_current = convert_exit_fraction_to_current(
        exit_fraction_of_original=exit_fraction_of_original,
        current_size=0.5,
        original_size=1.0,
    )

    # Assert
    assert exit_fraction_of_current is None


def test_convert_exit_fraction_to_current_returns_none_when_out_of_bounds() -> None:
    """Fractions over 100% of current size return None."""
    # Arrange
    exit_fraction_of_original = 0.8

    # Act
    exit_fraction_of_current = convert_exit_fraction_to_current(
        exit_fraction_of_original=exit_fraction_of_original,
        current_size=0.5,
        original_size=1.0,
    )

    # Assert
    assert exit_fraction_of_current is None
