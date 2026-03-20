"""Unit tests for _resolve_basis_balance helper in exit_handler.

Verifies the priority chain: entry_balance > caller balance > fallback constant.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.config.constants import DEFAULT_BASIS_BALANCE_FALLBACK
from src.engines.backtest.execution.exit_handler import _resolve_basis_balance
from src.engines.backtest.models import ActiveTrade


def _make_trade(entry_balance: float | None = None) -> ActiveTrade:
    """Create a minimal ActiveTrade with the given entry_balance."""
    return ActiveTrade(
        symbol="BTCUSDT",
        side="long",
        entry_price=50000.0,
        entry_time=datetime.now(UTC),
        size=0.02,
        entry_balance=entry_balance,
    )


class TestResolveBasisBalance:
    """Test _resolve_basis_balance priority chain."""

    def test_returns_entry_balance_when_positive(self) -> None:
        """Entry balance on the trade takes priority over caller balance."""
        # Arrange
        trade = _make_trade(entry_balance=5000.0)

        # Act
        result = _resolve_basis_balance(trade, balance=8000.0)

        # Assert
        assert result == 5000.0

    def test_returns_entry_balance_ignoring_none_caller_balance(self) -> None:
        """Entry balance is used even when caller balance is None."""
        # Arrange
        trade = _make_trade(entry_balance=5000.0)

        # Act
        result = _resolve_basis_balance(trade, balance=None)

        # Assert
        assert result == 5000.0

    def test_falls_back_to_caller_balance_when_entry_balance_is_none(self) -> None:
        """When entry_balance is None, fall back to caller-supplied balance."""
        # Arrange
        trade = _make_trade(entry_balance=None)

        # Act
        result = _resolve_basis_balance(trade, balance=8000.0)

        # Assert
        assert result == 8000.0

    def test_falls_back_to_caller_balance_when_entry_balance_is_zero(self) -> None:
        """When entry_balance is zero, fall back to caller-supplied balance."""
        # Arrange
        trade = _make_trade(entry_balance=0.0)

        # Act
        result = _resolve_basis_balance(trade, balance=8000.0)

        # Assert
        assert result == 8000.0

    def test_falls_back_to_caller_balance_when_entry_balance_is_negative(self) -> None:
        """When entry_balance is negative, fall back to caller-supplied balance."""
        # Arrange
        trade = _make_trade(entry_balance=-100.0)

        # Act
        result = _resolve_basis_balance(trade, balance=8000.0)

        # Assert
        assert result == 8000.0

    def test_falls_back_to_constant_when_both_unavailable(self) -> None:
        """When both entry_balance and caller balance are None, use fallback constant."""
        # Arrange
        trade = _make_trade(entry_balance=None)

        # Act
        result = _resolve_basis_balance(trade, balance=None)

        # Assert
        assert result == DEFAULT_BASIS_BALANCE_FALLBACK

    def test_falls_back_to_constant_when_both_zero(self) -> None:
        """When both entry_balance and caller balance are zero, use fallback constant."""
        # Arrange
        trade = _make_trade(entry_balance=0.0)

        # Act
        result = _resolve_basis_balance(trade, balance=0.0)

        # Assert
        assert result == DEFAULT_BASIS_BALANCE_FALLBACK

    def test_falls_back_to_constant_when_both_negative(self) -> None:
        """When both are negative, use fallback constant."""
        # Arrange
        trade = _make_trade(entry_balance=-50.0)

        # Act
        result = _resolve_basis_balance(trade, balance=-200.0)

        # Assert
        assert result == DEFAULT_BASIS_BALANCE_FALLBACK

    def test_returns_float_type(self) -> None:
        """Result is always a float regardless of input type."""
        # Arrange
        trade = _make_trade(entry_balance=5000)  # int

        # Act
        result = _resolve_basis_balance(trade, balance=8000)

        # Assert
        assert isinstance(result, float)

    def test_trade_without_entry_balance_attribute(self) -> None:
        """Gracefully handles trade objects missing entry_balance attribute."""
        # Arrange - use a plain object without entry_balance
        @dataclass
        class MinimalTrade:
            symbol: str = "BTCUSDT"

        trade = MinimalTrade()

        # Act
        result = _resolve_basis_balance(trade, balance=7000.0)  # type: ignore[arg-type]

        # Assert - getattr returns None, falls back to caller balance
        assert result == 7000.0
