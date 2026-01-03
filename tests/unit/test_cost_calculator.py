"""Unit tests for the shared CostCalculator."""

import pytest

from src.engines.shared.cost_calculator import CostCalculator


class TestCostCalculatorBasics:
    """Test basic CostCalculator functionality."""

    def test_initialization(self):
        """CostCalculator should initialize with default rates."""
        calculator = CostCalculator()

        assert calculator.fee_rate == 0.001
        assert calculator.slippage_rate == 0.0005
        assert calculator.total_fees_paid == 0.0
        assert calculator.total_slippage_cost == 0.0

    def test_custom_rates(self):
        """CostCalculator should accept custom fee and slippage rates."""
        calculator = CostCalculator(fee_rate=0.002, slippage_rate=0.001)

        assert calculator.fee_rate == 0.002
        assert calculator.slippage_rate == 0.001

    def test_get_settings(self):
        """get_settings should return current rates."""
        calculator = CostCalculator(fee_rate=0.003, slippage_rate=0.0015)
        settings = calculator.get_settings()

        assert settings["fee_rate"] == 0.003
        assert settings["slippage_rate"] == 0.0015


class TestCostCalculatorEntryLong:
    """Test entry cost calculations for long positions."""

    def test_long_entry_basic(self):
        """Long entry should apply adverse slippage (higher price)."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(
            price=100.0,
            notional=1000.0,
            side="long",
        )

        # Long entry: price increases
        assert result.executed_price == 100.1  # 100 * (1 + 0.001)
        assert result.fee == 1.0  # 1000 * 0.001
        # Slippage cost: |100.1 - 100| * (1000 / 100) = 0.1 * 10 = 1.0
        assert result.slippage_cost == pytest.approx(1.0, abs=0.001)
        assert result.total_cost == pytest.approx(2.0, abs=0.001)

    def test_long_entry_accumulation(self):
        """Long entry should accumulate fees and slippage."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert calculator.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(1.0, abs=0.001)

    def test_long_entry_case_insensitive(self):
        """Side parameter should be case insensitive."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result1 = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="LONG")
        calculator.reset_totals()
        result2 = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="Long")
        calculator.reset_totals()
        result3 = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert result1.executed_price == result2.executed_price == result3.executed_price
        assert result1.fee == result2.fee == result3.fee
        assert result1.slippage_cost == result2.slippage_cost == result3.slippage_cost


class TestCostCalculatorEntryShort:
    """Test entry cost calculations for short positions."""

    def test_short_entry_basic(self):
        """Short entry should apply adverse slippage (lower price)."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(
            price=100.0,
            notional=1000.0,
            side="short",
        )

        # Short entry: price decreases (we get less)
        assert result.executed_price == 99.9  # 100 * (1 - 0.001)
        assert result.fee == 1.0  # 1000 * 0.001
        # Slippage cost: |99.9 - 100| * (1000 / 100) = 0.1 * 10 = 1.0
        assert result.slippage_cost == pytest.approx(1.0, abs=0.001)
        assert result.total_cost == pytest.approx(2.0, abs=0.001)

    def test_short_entry_accumulation(self):
        """Short entry should accumulate fees and slippage."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="short")

        assert calculator.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(1.0, abs=0.001)


class TestCostCalculatorExitLong:
    """Test exit cost calculations for long positions."""

    def test_long_exit_basic(self):
        """Long exit should apply adverse slippage (lower price)."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_exit_costs(
            price=100.0,
            notional=1000.0,
            side="long",
        )

        # Long exit: price decreases (we get less)
        assert result.executed_price == 99.9  # 100 * (1 - 0.001)
        assert result.fee == 1.0  # 1000 * 0.001
        # Slippage cost: |99.9 - 100| * (1000 / 100) = 0.1 * 10 = 1.0
        assert result.slippage_cost == pytest.approx(1.0, abs=0.001)
        assert result.total_cost == pytest.approx(2.0, abs=0.001)

    def test_long_exit_accumulation(self):
        """Long exit should accumulate fees and slippage."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="long")

        assert calculator.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(1.0, abs=0.001)


class TestCostCalculatorExitShort:
    """Test exit cost calculations for short positions."""

    def test_short_exit_basic(self):
        """Short exit should apply adverse slippage (higher price)."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_exit_costs(
            price=100.0,
            notional=1000.0,
            side="short",
        )

        # Short exit: price increases (we pay more)
        assert result.executed_price == 100.1  # 100 * (1 + 0.001)
        assert result.fee == 1.0  # 1000 * 0.001
        # Slippage cost: |100.1 - 100| * (1000 / 100) = 0.1 * 10 = 1.0
        assert result.slippage_cost == pytest.approx(1.0, abs=0.001)
        assert result.total_cost == pytest.approx(2.0, abs=0.001)

    def test_short_exit_accumulation(self):
        """Short exit should accumulate fees and slippage."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="short")

        assert calculator.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(1.0, abs=0.001)


class TestCostCalculatorAccumulation:
    """Test accumulation of fees and slippage across multiple trades."""

    def test_multiple_trades_accumulation(self):
        """Fees and slippage should accumulate across multiple trades."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        # Trade 1: Long entry
        calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        assert calculator.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(1.0, abs=0.001)

        # Trade 2: Long exit
        calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="long")
        assert calculator.total_fees_paid == pytest.approx(2.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(2.0, abs=0.001)

        # Trade 3: Short entry
        calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="short")
        assert calculator.total_fees_paid == pytest.approx(3.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(3.0, abs=0.001)

        # Trade 4: Short exit
        calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="short")
        assert calculator.total_fees_paid == pytest.approx(4.0, abs=0.001)
        assert calculator.total_slippage_cost == pytest.approx(4.0, abs=0.001)

    def test_reset_totals(self):
        """reset_totals should clear accumulated fees and slippage."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        # Accumulate some costs
        calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="long")

        assert calculator.total_fees_paid > 0
        assert calculator.total_slippage_cost > 0

        # Reset
        calculator.reset_totals()

        assert calculator.total_fees_paid == 0.0
        assert calculator.total_slippage_cost == 0.0


class TestCostCalculatorValidation:
    """Test input validation in CostCalculator."""

    def test_entry_zero_price(self):
        """Entry with zero price should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Price must be positive"):
            calculator.calculate_entry_costs(price=0.0, notional=1000.0, side="long")

    def test_entry_negative_price(self):
        """Entry with negative price should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Price must be positive"):
            calculator.calculate_entry_costs(price=-100.0, notional=1000.0, side="long")

    def test_entry_negative_notional(self):
        """Entry with negative notional should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Notional must be non-negative"):
            calculator.calculate_entry_costs(price=100.0, notional=-1000.0, side="long")

    def test_entry_invalid_side(self):
        """Entry with invalid side should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Side must be 'long' or 'short'"):
            calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="buy")

    def test_exit_zero_price(self):
        """Exit with zero price should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Price must be positive"):
            calculator.calculate_exit_costs(price=0.0, notional=1000.0, side="long")

    def test_exit_negative_price(self):
        """Exit with negative price should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Price must be positive"):
            calculator.calculate_exit_costs(price=-100.0, notional=1000.0, side="long")

    def test_exit_negative_notional(self):
        """Exit with negative notional should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Notional must be non-negative"):
            calculator.calculate_exit_costs(price=100.0, notional=-1000.0, side="long")

    def test_exit_invalid_side(self):
        """Exit with invalid side should raise ValueError."""
        calculator = CostCalculator()

        with pytest.raises(ValueError, match="Side must be 'long' or 'short'"):
            calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="sell")

    def test_entry_zero_notional(self):
        """Entry with zero notional should be allowed."""
        calculator = CostCalculator()

        result = calculator.calculate_entry_costs(price=100.0, notional=0.0, side="long")

        assert result.fee == 0.0
        assert result.slippage_cost == 0.0


class TestCostCalculatorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_notional(self):
        """Very small notional should calculate correctly."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(
            price=100.0,
            notional=0.01,
            side="long",
        )

        assert result.fee == pytest.approx(0.00001, abs=0.000001)
        assert result.slippage_cost == pytest.approx(0.00001, abs=0.000001)

    def test_very_large_notional(self):
        """Very large notional should calculate correctly."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(
            price=100.0,
            notional=1000000.0,
            side="long",
        )

        assert result.fee == pytest.approx(1000.0, abs=0.001)
        assert result.slippage_cost == pytest.approx(1000.0, abs=0.001)

    def test_very_small_price(self):
        """Very small price should calculate correctly."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(
            price=0.0001,
            notional=1000.0,
            side="long",
        )

        assert result.executed_price == pytest.approx(0.0001001, abs=0.0000001)
        assert result.fee == pytest.approx(1.0, abs=0.001)

    def test_very_large_price(self):
        """Very large price should calculate correctly."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(
            price=100000.0,
            notional=1000.0,
            side="long",
        )

        assert result.executed_price == pytest.approx(100100.0, abs=0.01)
        assert result.fee == pytest.approx(1.0, abs=0.001)

    def test_calculate_fee_helper(self):
        """calculate_fee helper should return correct fee."""
        calculator = CostCalculator(fee_rate=0.001)

        fee = calculator.calculate_fee(notional=1000.0)

        assert fee == 1.0


class TestCostCalculatorFinancialCorrectness:
    """Test financial correctness of cost calculations."""

    def test_round_trip_long_position(self):
        """Round trip long position should have symmetric costs."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        # Entry
        entry = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        # Exit
        exit_result = calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="long")

        # Entry price should be higher, exit price should be lower
        assert entry.executed_price > 100.0
        assert exit_result.executed_price < 100.0

        # Total costs should be symmetric
        assert entry.total_cost == pytest.approx(exit_result.total_cost, abs=0.001)

    def test_round_trip_short_position(self):
        """Round trip short position should have symmetric costs."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        # Entry
        entry = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="short")
        # Exit
        exit_result = calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="short")

        # Entry price should be lower, exit price should be higher
        assert entry.executed_price < 100.0
        assert exit_result.executed_price > 100.0

        # Total costs should be symmetric
        assert entry.total_cost == pytest.approx(exit_result.total_cost, abs=0.001)

    def test_slippage_always_adverse(self):
        """Slippage should always be adverse (costs money)."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        # Long entry: executed price > base price
        long_entry = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        assert long_entry.executed_price > 100.0

        # Long exit: executed price < base price
        long_exit = calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="long")
        assert long_exit.executed_price < 100.0

        # Short entry: executed price < base price
        short_entry = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="short")
        assert short_entry.executed_price < 100.0

        # Short exit: executed price > base price
        short_exit = calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="short")
        assert short_exit.executed_price > 100.0

    def test_total_cost_property(self):
        """CostResult total_cost property should equal fee + slippage."""
        calculator = CostCalculator(fee_rate=0.001, slippage_rate=0.001)

        result = calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert result.total_cost == result.fee + result.slippage_cost
