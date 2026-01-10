"""Integration tests for CostCalculator consistency across engines."""

import pytest

from src.engines.backtest.execution.execution_engine import (
    ExecutionEngine as BacktestExecutionEngine,
)
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.position_tracker import PositionSide
from src.engines.shared.cost_calculator import CostCalculator


@pytest.mark.integration
class TestCostCalculatorIntegration:
    """Test CostCalculator integration with both execution engines."""

    def test_shared_calculator_instance(self):
        """Both engines should use CostCalculator for cost calculations."""
        backtest_engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)
        live_engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Both should have a cost calculator
        assert hasattr(backtest_engine, "_cost_calculator")
        assert hasattr(live_engine, "_cost_calculator")

        # Both calculators should be CostCalculator instances
        assert isinstance(backtest_engine._cost_calculator, CostCalculator)
        assert isinstance(live_engine._cost_calculator, CostCalculator)

    def test_engines_produce_identical_entry_costs(self):
        """Both engines should produce identical entry costs for same inputs."""
        backtest_engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)
        live_engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Calculate entry costs using backtest engine
        backtest_result = backtest_engine._cost_calculator.calculate_entry_costs(
            price=100.0,
            notional=1000.0,
            side="long",
        )

        # Calculate entry costs using live engine
        live_result = live_engine._cost_calculator.calculate_entry_costs(
            price=100.0,
            notional=1000.0,
            side="long",
        )

        # Results should be identical
        assert backtest_result.executed_price == live_result.executed_price
        assert backtest_result.fee == live_result.fee
        assert backtest_result.slippage_cost == live_result.slippage_cost
        assert backtest_result.total_cost == live_result.total_cost

    def test_engines_produce_identical_exit_costs(self):
        """Both engines should produce identical exit costs for same inputs."""
        backtest_engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)
        live_engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Calculate exit costs using backtest engine
        backtest_result = backtest_engine._cost_calculator.calculate_exit_costs(
            price=100.0,
            notional=1000.0,
            side="short",
        )

        # Calculate exit costs using live engine
        live_result = live_engine._cost_calculator.calculate_exit_costs(
            price=100.0,
            notional=1000.0,
            side="short",
        )

        # Results should be identical
        assert backtest_result.executed_price == live_result.executed_price
        assert backtest_result.fee == live_result.fee
        assert backtest_result.slippage_cost == live_result.slippage_cost
        assert backtest_result.total_cost == live_result.total_cost

    def test_engines_accumulate_costs_identically(self):
        """Both engines should accumulate costs identically."""
        backtest_engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)
        live_engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Execute same sequence of trades on both engines
        for _ in range(5):
            backtest_engine._cost_calculator.calculate_entry_costs(
                price=100.0, notional=1000.0, side="long"
            )
            live_engine._cost_calculator.calculate_entry_costs(
                price=100.0, notional=1000.0, side="long"
            )

        # Total costs should be identical
        assert backtest_engine.total_fees_paid == live_engine.total_fees_paid
        assert backtest_engine.total_slippage_cost == live_engine.total_slippage_cost

    def test_backtest_engine_calculate_exit_costs(self):
        """Backtest engine calculate_exit_costs should use CostCalculator."""
        engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Use the public method that returns tuple
        exit_price, fee, slippage = engine.calculate_exit_costs(
            base_price=100.0,
            side="long",
            position_notional=1000.0,
        )

        # Verify it used the calculator correctly
        assert exit_price == 99.9  # Long exit: price decreases
        assert fee == pytest.approx(1.0, abs=0.001)
        assert slippage == pytest.approx(1.0, abs=0.001)

    def test_live_engine_get_execution_stats(self):
        """Live engine get_execution_stats should return correct values."""
        engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Execute some trades
        engine._cost_calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        engine._cost_calculator.calculate_exit_costs(price=100.0, notional=1000.0, side="long")

        # Get stats
        stats = engine.get_execution_stats()

        # Stats should include totals from calculator
        assert "total_fees_paid" in stats
        assert "total_slippage_cost" in stats
        assert stats["total_fees_paid"] == pytest.approx(2.0, abs=0.001)
        assert stats["total_slippage_cost"] == pytest.approx(2.0, abs=0.001)

    def test_backtest_engine_reset_clears_calculator(self):
        """Backtest engine reset should clear calculator totals."""
        engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Accumulate some costs
        engine._cost_calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert engine.total_fees_paid > 0
        assert engine.total_slippage_cost > 0

        # Reset
        engine.reset()

        assert engine.total_fees_paid == 0.0
        assert engine.total_slippage_cost == 0.0

    def test_live_engine_reset_tracking_clears_calculator(self):
        """Live engine reset_tracking should clear calculator totals."""
        engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Accumulate some costs
        engine._cost_calculator.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert engine.total_fees_paid > 0
        assert engine.total_slippage_cost > 0

        # Reset
        engine.reset_tracking()

        assert engine.total_fees_paid == 0.0
        assert engine.total_slippage_cost == 0.0

    def test_live_engine_position_side_conversion(self):
        """Live engine should correctly convert PositionSide to string."""
        engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Test static method
        assert engine._position_side_to_str(PositionSide.LONG) == "long"
        assert engine._position_side_to_str(PositionSide.SHORT) == "short"

    def test_live_engine_execute_entry_uses_calculator(self):
        """Live engine execute_entry should use CostCalculator."""
        engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        result = engine.execute_entry(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size_fraction=0.1,
            base_price=100.0,
            balance=10000.0,
        )

        # Should succeed and have costs
        assert result.success
        assert result.entry_fee == pytest.approx(1.0, abs=0.001)  # 1000 * 0.001
        assert result.slippage_cost == pytest.approx(1.0, abs=0.001)

        # Should accumulate in calculator
        assert engine.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert engine.total_slippage_cost == pytest.approx(1.0, abs=0.001)

    def test_live_engine_execute_exit_uses_calculator(self):
        """Live engine execute_exit should use CostCalculator."""
        engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        result = engine.execute_exit(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_id="test_order",
            base_price=100.0,
            position_notional=1000.0,
        )

        # Should succeed and have costs
        assert result.success
        assert result.exit_fee == pytest.approx(1.0, abs=0.001)
        assert result.slippage_cost == pytest.approx(1.0, abs=0.001)

        # Should accumulate in calculator
        assert engine.total_fees_paid == pytest.approx(1.0, abs=0.001)
        assert engine.total_slippage_cost == pytest.approx(1.0, abs=0.001)

    def test_engines_with_different_rates_produce_different_costs(self):
        """Engines with different rates should produce different costs."""
        engine1 = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)
        engine2 = BacktestExecutionEngine(fee_rate=0.002, slippage_rate=0.002)

        result1 = engine1._cost_calculator.calculate_entry_costs(
            price=100.0, notional=1000.0, side="long"
        )
        result2 = engine2._cost_calculator.calculate_entry_costs(
            price=100.0, notional=1000.0, side="long"
        )

        # Results should differ
        assert result1.executed_price != result2.executed_price
        assert result1.fee != result2.fee
        assert result1.slippage_cost != result2.slippage_cost

    def test_cost_calculator_properties_accessible(self):
        """Both engines should expose cost calculator totals via properties."""
        backtest_engine = BacktestExecutionEngine(fee_rate=0.001, slippage_rate=0.001)
        live_engine = LiveExecutionEngine(fee_rate=0.001, slippage_rate=0.001)

        # Both should have properties
        assert hasattr(backtest_engine, "total_fees_paid")
        assert hasattr(backtest_engine, "total_slippage_cost")
        assert hasattr(live_engine, "total_fees_paid")
        assert hasattr(live_engine, "total_slippage_cost")

        # Properties should be accessible
        assert backtest_engine.total_fees_paid == 0.0
        assert backtest_engine.total_slippage_cost == 0.0
        assert live_engine.total_fees_paid == 0.0
        assert live_engine.total_slippage_cost == 0.0
