"""Tests for atb migration commands."""

import argparse
from unittest.mock import Mock, patch

import pytest

from cli.commands.migration import _load_strategy


class TestLoadStrategy:
    """Tests for the _load_strategy function."""

    def test_loads_ml_basic_strategy(self):
        """Test that ml_basic strategy is loaded."""
        # Arrange & Act
        with patch("cli.commands.migration.create_ml_basic_strategy") as mock_create:
            mock_strategy = Mock()
            mock_create.return_value = mock_strategy

            result = _load_strategy("ml_basic")

            # Assert
            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_loads_ml_adaptive_strategy(self):
        """Test that ml_adaptive strategy is loaded."""
        # Arrange & Act
        with patch("cli.commands.migration.create_ml_adaptive_strategy") as mock_create:
            mock_strategy = Mock()
            mock_create.return_value = mock_strategy

            result = _load_strategy("ml_adaptive")

            # Assert
            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_raises_error_for_unknown_strategy(self):
        """Test that error is raised for unknown strategy."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Supported strategies"):
            _load_strategy("unknown_strategy")

    def test_case_insensitive_strategy_loading(self):
        """Test that strategy loading is case insensitive."""
        # Arrange & Act
        with patch("cli.commands.migration.create_ml_basic_strategy") as mock_create:
            mock_strategy = Mock()
            mock_create.return_value = mock_strategy

            result = _load_strategy("ML_BASIC")

            # Assert
            assert result == mock_strategy
