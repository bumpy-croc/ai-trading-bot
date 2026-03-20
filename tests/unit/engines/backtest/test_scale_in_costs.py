"""Unit tests for ExecutionEngine.calculate_scale_in_costs.

Verifies that scale-in orders use the same CostCalculator as initial entries,
returning correct fee and slippage values.
"""

import pytest

from src.engines.backtest.execution.execution_engine import ExecutionEngine


class TestCalculateScaleInCosts:
    """Test ExecutionEngine.calculate_scale_in_costs method."""

    def test_returns_fee_and_slippage(self) -> None:
        """Scale-in returns a (fee, slippage_cost) tuple."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0005)

        # Act
        fee, slippage_cost = engine.calculate_scale_in_costs(
            price=50000.0, notional=1000.0, side="long"
        )

        # Assert
        assert fee == pytest.approx(1000.0 * 0.001)  # 1.0
        assert slippage_cost > 0

    def test_fee_matches_initial_entry_fee(self) -> None:
        """Scale-in fee equals the fee for an equivalent initial entry."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0005)

        # Act
        scale_fee, scale_slip = engine.calculate_scale_in_costs(
            price=50000.0, notional=2000.0, side="long"
        )

        # Compare with an initial entry of the same notional
        entry_result = engine._cost_calculator.calculate_entry_costs(
            price=50000.0, notional=2000.0, side="long"
        )

        # Assert - fees should be identical
        assert scale_fee == pytest.approx(entry_result.fee)
        assert scale_slip == pytest.approx(entry_result.slippage_cost)

    def test_short_side_slippage(self) -> None:
        """Scale-in for short positions calculates slippage correctly."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Act
        fee, slippage_cost = engine.calculate_scale_in_costs(
            price=40000.0, notional=500.0, side="short"
        )

        # Assert
        expected_fee = 500.0 * 0.001  # 0.5
        assert fee == pytest.approx(expected_fee)
        assert slippage_cost > 0

    def test_zero_fee_rate(self) -> None:
        """Scale-in with zero fee rate returns zero fee."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.0, slippage_rate=0.0005)

        # Act
        fee, slippage_cost = engine.calculate_scale_in_costs(
            price=50000.0, notional=1000.0, side="long"
        )

        # Assert
        assert fee == 0.0
        assert slippage_cost > 0

    def test_zero_slippage_rate(self) -> None:
        """Scale-in with zero slippage rate returns zero slippage."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0)

        # Act
        fee, slippage_cost = engine.calculate_scale_in_costs(
            price=50000.0, notional=1000.0, side="long"
        )

        # Assert
        assert fee == pytest.approx(1.0)
        assert slippage_cost == 0.0

    def test_accumulates_totals(self) -> None:
        """Scale-in costs accumulate in the engine's total fee tracker."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0005)
        initial_fees = engine.total_fees_paid

        # Act
        engine.calculate_scale_in_costs(
            price=50000.0, notional=1000.0, side="long"
        )

        # Assert
        assert engine.total_fees_paid > initial_fees

    def test_liquidity_parameter_forwarded(self) -> None:
        """Liquidity parameter is forwarded to the cost calculator."""
        # Arrange
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0005)

        # Act - maker liquidity should produce results without error
        fee, slippage_cost = engine.calculate_scale_in_costs(
            price=50000.0, notional=1000.0, side="long", liquidity="maker"
        )

        # Assert - just verify it runs without error and returns valid values
        assert fee >= 0
        assert slippage_cost >= 0
