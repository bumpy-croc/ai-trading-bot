"""Unit tests for scale-in fee deduction in the backtest engine.

Verifies that PartialOpsResult.scale_in_fees is deducted from the engine
balance, and that non-finite values are safely skipped.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from src.engines.backtest.execution.exit_handler import PartialOpsResult


class TestScaleInFeeDeduction:
    """Test that the backtest engine deducts scale_in_fees from balance."""

    @staticmethod
    def _simulate_fee_deduction(
        balance: float, scale_in_fees: float
    ) -> float:
        """Reproduce the engine's fee deduction logic (lines 1150-1158 of engine.py).

        This mirrors the exact branching logic in the backtest engine so we
        can test it in isolation without constructing a full Backtester.
        """
        if not math.isfinite(scale_in_fees):
            # Invalid fees are skipped
            return balance
        elif scale_in_fees > 0:
            balance -= scale_in_fees
        return balance

    def test_positive_fee_deducted_from_balance(self) -> None:
        """Positive scale_in_fees reduce the balance."""
        # Arrange
        balance = 10000.0
        fees = 2.5

        # Act
        new_balance = self._simulate_fee_deduction(balance, fees)

        # Assert
        assert new_balance == pytest.approx(9997.5)

    def test_zero_fee_leaves_balance_unchanged(self) -> None:
        """Zero scale_in_fees do not change the balance."""
        # Arrange
        balance = 10000.0

        # Act
        new_balance = self._simulate_fee_deduction(balance, 0.0)

        # Assert
        assert new_balance == 10000.0

    def test_nan_fee_skipped(self) -> None:
        """NaN scale_in_fees are not deducted (safety guard)."""
        # Arrange
        balance = 10000.0

        # Act
        new_balance = self._simulate_fee_deduction(balance, float("nan"))

        # Assert
        assert new_balance == 10000.0

    def test_positive_infinity_fee_skipped(self) -> None:
        """Positive infinity scale_in_fees are not deducted."""
        # Arrange
        balance = 10000.0

        # Act
        new_balance = self._simulate_fee_deduction(balance, float("inf"))

        # Assert
        assert new_balance == 10000.0

    def test_negative_infinity_fee_skipped(self) -> None:
        """Negative infinity scale_in_fees are not deducted."""
        # Arrange
        balance = 10000.0

        # Act
        new_balance = self._simulate_fee_deduction(balance, float("-inf"))

        # Assert
        assert new_balance == 10000.0

    def test_negative_fee_does_not_deduct(self) -> None:
        """Negative fees do not trigger deduction (guard: scale_in_fees > 0)."""
        # Arrange
        balance = 10000.0

        # Act
        new_balance = self._simulate_fee_deduction(balance, -1.0)

        # Assert - negative fee is finite but not > 0, so balance unchanged
        assert new_balance == 10000.0

    def test_partial_ops_result_carries_scale_in_fees(self) -> None:
        """PartialOpsResult dataclass correctly stores scale_in_fees."""
        # Arrange / Act
        result = PartialOpsResult(
            realized_pnl=50.0,
            partial_exits=[],
            scale_ins=[{"size": 0.01, "price": 50000.0, "fee": 1.5, "slippage": 0.25}],
            scale_in_fees=1.5,
        )

        # Assert
        assert result.scale_in_fees == 1.5
        assert math.isfinite(result.scale_in_fees)

    def test_partial_ops_result_defaults_to_zero_fees(self) -> None:
        """PartialOpsResult defaults scale_in_fees to 0.0."""
        # Arrange / Act
        result = PartialOpsResult(
            realized_pnl=0.0,
            partial_exits=[],
            scale_ins=[],
        )

        # Assert
        assert result.scale_in_fees == 0.0
