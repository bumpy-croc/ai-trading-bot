"""Unit tests for shared CostCalculator.

These tests verify that the cost calculator produces consistent results
and is used identically by both backtesting and live trading engines.
"""

import pytest

from src.engines.shared.cost_calculator import CostCalculator, CostResult


class TestCostCalculatorInitialization:
    """Test CostCalculator initialization."""

    def test_default_initialization(self) -> None:
        """Test default fee and slippage rates."""
        calc = CostCalculator()
        assert calc.fee_rate == 0.001  # 0.1%
        assert calc.slippage_rate == 0.0005  # 0.05%
        assert calc.total_fees_paid == 0.0
        assert calc.total_slippage_cost == 0.0

    def test_custom_initialization(self) -> None:
        """Test custom fee and slippage rates."""
        calc = CostCalculator(fee_rate=0.002, slippage_rate=0.001)
        assert calc.fee_rate == 0.002
        assert calc.slippage_rate == 0.001

    def test_zero_costs_initialization(self) -> None:
        """Test initialization with zero costs."""
        calc = CostCalculator(fee_rate=0.0, slippage_rate=0.0)
        assert calc.fee_rate == 0.0
        assert calc.slippage_rate == 0.0


class TestEntryCosts:
    """Test entry cost calculations."""

    def test_long_entry_slippage_increases_price(self) -> None:
        """Long entry should get worse (higher) price due to slippage."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        result = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        # Price should be worse (higher) for buyer
        expected_price = 100.0 * (1 + 0.0005)
        assert result.executed_price == pytest.approx(expected_price)
        assert result.executed_price > 100.0

    def test_short_entry_slippage_decreases_price(self) -> None:
        """Short entry should get worse (lower) price due to slippage."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        result = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="short")

        # Price should be worse (lower) for seller
        expected_price = 100.0 * (1 - 0.0005)
        assert result.executed_price == pytest.approx(expected_price)
        assert result.executed_price < 100.0

    def test_entry_fee_calculation(self) -> None:
        """Entry fee should be calculated on notional."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0)
        result = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        expected_fee = 1000.0 * 0.001
        assert result.fee == pytest.approx(expected_fee)

    def test_entry_slippage_cost_calculation(self) -> None:
        """Slippage cost should be calculated correctly."""
        calc = CostCalculator(fee_rate=0.0, slippage_rate=0.001)
        result = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        # Slippage cost = |executed - price| * (notional / price)
        # = |100.1 - 100| * 10 = 1.0
        assert result.slippage_cost == pytest.approx(1.0)

    def test_entry_updates_totals(self) -> None:
        """Entry should update running totals."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert calc.total_fees_paid > 0
        assert calc.total_slippage_cost > 0

    def test_entry_invalid_price_raises(self) -> None:
        """Non-positive price should raise ValueError."""
        calc = CostCalculator()
        with pytest.raises(ValueError, match="Price must be positive"):
            calc.calculate_entry_costs(price=0.0, notional=1000.0, side="long")

        with pytest.raises(ValueError, match="Price must be positive"):
            calc.calculate_entry_costs(price=-100.0, notional=1000.0, side="long")

    def test_entry_invalid_notional_raises(self) -> None:
        """Negative notional should raise ValueError."""
        calc = CostCalculator()
        with pytest.raises(ValueError, match="Notional must be non-negative"):
            calc.calculate_entry_costs(price=100.0, notional=-1000.0, side="long")

    def test_entry_invalid_side_raises(self) -> None:
        """Invalid side should raise ValueError."""
        calc = CostCalculator()
        with pytest.raises(ValueError, match="Side must be"):
            calc.calculate_entry_costs(price=100.0, notional=1000.0, side="invalid")


class TestExitCosts:
    """Test exit cost calculations."""

    def test_long_exit_slippage_decreases_price(self) -> None:
        """Long exit (selling) should get worse (lower) price due to slippage."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        result = calc.calculate_exit_costs(price=110.0, notional=1100.0, side="long")

        # Price should be worse (lower) for seller
        expected_price = 110.0 * (1 - 0.0005)
        assert result.executed_price == pytest.approx(expected_price)
        assert result.executed_price < 110.0

    def test_short_exit_slippage_increases_price(self) -> None:
        """Short exit (buying back) should get worse (higher) price due to slippage."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        result = calc.calculate_exit_costs(price=90.0, notional=900.0, side="short")

        # Price should be worse (higher) for buyer
        expected_price = 90.0 * (1 + 0.0005)
        assert result.executed_price == pytest.approx(expected_price)
        assert result.executed_price > 90.0

    def test_exit_fee_calculation(self) -> None:
        """Exit fee should be calculated on exit notional."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0)
        result = calc.calculate_exit_costs(price=110.0, notional=1100.0, side="long")

        expected_fee = 1100.0 * 0.001
        assert result.fee == pytest.approx(expected_fee)


class TestCostAccumulation:
    """Test cost accumulation tracking."""

    def test_multiple_trades_accumulate(self) -> None:
        """Multiple trades should accumulate fees and slippage."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)

        calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        first_fees = calc.total_fees_paid
        first_slippage = calc.total_slippage_cost

        calc.calculate_exit_costs(price=110.0, notional=1100.0, side="long")

        assert calc.total_fees_paid > first_fees
        assert calc.total_slippage_cost > first_slippage

    def test_reset_totals(self) -> None:
        """Reset should clear accumulated totals."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)

        calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        assert calc.total_fees_paid > 0

        calc.reset_totals()
        assert calc.total_fees_paid == 0.0
        assert calc.total_slippage_cost == 0.0

    def test_get_settings(self) -> None:
        """Settings should return fee and slippage rates."""
        calc = CostCalculator(fee_rate=0.002, slippage_rate=0.001)
        settings = calc.get_settings()

        assert settings["fee_rate"] == 0.002
        assert settings["slippage_rate"] == 0.001


class TestCostResult:
    """Test CostResult dataclass."""

    def test_total_cost_property(self) -> None:
        """Total cost should sum fee and slippage."""
        result = CostResult(
            executed_price=100.05,
            fee=1.0,
            slippage_cost=0.5,
        )

        assert result.total_cost == pytest.approx(1.5)


class TestCalculateFee:
    """Test standalone fee calculation."""

    def test_calculate_fee_only(self) -> None:
        """Calculate fee should return just the fee amount."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        fee = calc.calculate_fee(notional=1000.0)

        assert fee == pytest.approx(1.0)

    def test_calculate_fee_does_not_accumulate(self) -> None:
        """Calculate fee should not affect totals."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)

        initial_total = calc.total_fees_paid
        calc.calculate_fee(notional=1000.0)

        # Note: This test checks current behavior. If we want calculate_fee to
        # NOT affect totals, we might need to update the implementation.
        # Currently checking it doesn't change.
        assert calc.total_fees_paid == initial_total


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_notional(self) -> None:
        """Zero notional should produce zero costs."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        result = calc.calculate_entry_costs(price=100.0, notional=0.0, side="long")

        assert result.fee == 0.0
        assert result.slippage_cost == 0.0

    def test_very_large_notional(self) -> None:
        """Very large notional should not overflow."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)
        result = calc.calculate_entry_costs(
            price=100.0, notional=1_000_000_000.0, side="long"
        )

        assert result.fee == pytest.approx(1_000_000.0)
        assert result.slippage_cost > 0

    def test_very_small_rates(self) -> None:
        """Very small rates should produce small but non-zero costs."""
        calc = CostCalculator(fee_rate=0.00001, slippage_rate=0.000001)
        result = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert result.fee > 0
        assert result.slippage_cost > 0

    def test_side_case_insensitive(self) -> None:
        """Side parameter should be case-insensitive."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)

        result_lower = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        calc.reset_totals()
        result_upper = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="LONG")

        assert result_lower.executed_price == result_upper.executed_price
