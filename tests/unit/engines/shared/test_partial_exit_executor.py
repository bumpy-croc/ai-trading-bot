"""Unit tests for shared PartialExitExecutor.

These tests verify that the shared executor produces consistent results
for both backtesting and live trading engines.
"""

import pytest

from src.engines.shared.models import PositionSide
from src.engines.shared.partial_exit_executor import (
    PartialExitExecutor,
)


@pytest.fixture
def executor() -> PartialExitExecutor:
    """Create executor with standard rates."""
    return PartialExitExecutor(fee_rate=0.001, slippage_rate=0.0005)


@pytest.fixture
def zero_cost_executor() -> PartialExitExecutor:
    """Create executor with zero fees and slippage for P&L validation."""
    return PartialExitExecutor(fee_rate=0.0, slippage_rate=0.0)


class TestPartialExitExecutor:
    """Test suite for PartialExitExecutor."""

    def test_initialization_valid(self) -> None:
        """Test executor initialization with valid parameters."""
        executor = PartialExitExecutor(fee_rate=0.002, slippage_rate=0.001)
        assert executor.fee_rate == 0.002
        assert executor.slippage_rate == 0.001

    def test_initialization_zero_costs(self) -> None:
        """Test executor can be initialized with zero costs."""
        executor = PartialExitExecutor(fee_rate=0.0, slippage_rate=0.0)
        assert executor.fee_rate == 0.0
        assert executor.slippage_rate == 0.0

    def test_initialization_negative_fee_raises(self) -> None:
        """Test negative fee rate raises ValueError."""
        with pytest.raises(ValueError, match="fee_rate must be non-negative"):
            PartialExitExecutor(fee_rate=-0.001, slippage_rate=0.0005)

    def test_initialization_negative_slippage_raises(self) -> None:
        """Test negative slippage rate raises ValueError."""
        with pytest.raises(ValueError, match="slippage_rate must be non-negative"):
            PartialExitExecutor(fee_rate=0.001, slippage_rate=-0.0005)

    def test_long_profitable_exit(self, executor: PartialExitExecutor) -> None:
        """Test long position profitable partial exit."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=110.0,  # +10% profit
            position_side=PositionSide.LONG,
            exit_fraction=0.5,  # Exit 50%
            basis_balance=10000.0,
        )

        # Expected calculation:
        # P&L% = (110 - 100) / 100 * 0.5 = 0.05 (5%)
        # Gross P&L = 10000 * 0.05 = $500
        # Exit notional = 10000 * 0.5 * (110/100) = $5,500
        # Fee = 5500 * 0.001 = $5.50
        # Slippage = 5500 * 0.0005 = $2.75
        # Net P&L = 500 - 5.50 - 2.75 = $491.75

        assert result.pnl_percent == pytest.approx(0.05)
        assert result.gross_pnl == pytest.approx(500.0)
        assert result.exit_notional == pytest.approx(5500.0)
        assert result.exit_fee == pytest.approx(5.5)
        assert result.slippage_cost == pytest.approx(2.75)
        assert result.realized_pnl == pytest.approx(491.75)

    def test_long_losing_exit(self, executor: PartialExitExecutor) -> None:
        """Test long position losing partial exit."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=90.0,  # -10% loss
            position_side=PositionSide.LONG,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        # Expected calculation:
        # P&L% = (90 - 100) / 100 * 0.5 = -0.05 (-5%)
        # Gross P&L = 10000 * -0.05 = -$500
        # Exit notional = 10000 * 0.5 * (90/100) = $4,500
        # Fee = 4500 * 0.001 = $4.50
        # Slippage = 4500 * 0.0005 = $2.25
        # Net P&L = -500 - 4.50 - 2.25 = -$506.75

        assert result.pnl_percent == pytest.approx(-0.05)
        assert result.gross_pnl == pytest.approx(-500.0)
        assert result.exit_notional == pytest.approx(4500.0)
        assert result.exit_fee == pytest.approx(4.5)
        assert result.slippage_cost == pytest.approx(2.25)
        assert result.realized_pnl == pytest.approx(-506.75)

    def test_short_profitable_exit(self, executor: PartialExitExecutor) -> None:
        """Test short position profitable partial exit."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=90.0,  # -10% price move = +10% profit for short
            position_side=PositionSide.SHORT,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        # Expected calculation:
        # P&L% = (100 - 90) / 100 * 0.5 = 0.05 (5%)
        # Gross P&L = 10000 * 0.05 = $500
        # Exit notional = 10000 * 0.5 * (90/100) = $4,500
        # Fee = 4500 * 0.001 = $4.50
        # Slippage = 4500 * 0.0005 = $2.25
        # Net P&L = 500 - 4.50 - 2.25 = $493.25

        assert result.pnl_percent == pytest.approx(0.05)
        assert result.gross_pnl == pytest.approx(500.0)
        assert result.exit_notional == pytest.approx(4500.0)
        assert result.exit_fee == pytest.approx(4.5)
        assert result.slippage_cost == pytest.approx(2.25)
        assert result.realized_pnl == pytest.approx(493.25)

    def test_short_losing_exit(self, executor: PartialExitExecutor) -> None:
        """Test short position losing partial exit."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=110.0,  # +10% price move = -10% loss for short
            position_side=PositionSide.SHORT,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        # Expected calculation:
        # P&L% = (100 - 110) / 100 * 0.5 = -0.05 (-5%)
        # Gross P&L = 10000 * -0.05 = -$500
        # Exit notional = 10000 * 0.5 * (110/100) = $5,500
        # Fee = 5500 * 0.001 = $5.50
        # Slippage = 5500 * 0.0005 = $2.75
        # Net P&L = -500 - 5.50 - 2.75 = -$508.25

        assert result.pnl_percent == pytest.approx(-0.05)
        assert result.gross_pnl == pytest.approx(-500.0)
        assert result.exit_notional == pytest.approx(5500.0)
        assert result.exit_fee == pytest.approx(5.5)
        assert result.slippage_cost == pytest.approx(2.75)
        assert result.realized_pnl == pytest.approx(-508.25)

    def test_zero_cost_executor_no_fees(self, zero_cost_executor: PartialExitExecutor) -> None:
        """Test that zero-cost executor returns gross P&L as net P&L."""
        result = zero_cost_executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=110.0,
            position_side=PositionSide.LONG,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        assert result.exit_fee == 0.0
        assert result.slippage_cost == 0.0
        assert result.realized_pnl == result.gross_pnl
        assert result.realized_pnl == pytest.approx(500.0)

    def test_full_position_exit(self, executor: PartialExitExecutor) -> None:
        """Test exiting 100% of position."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=110.0,
            position_side=PositionSide.LONG,
            exit_fraction=1.0,  # Exit 100%
            basis_balance=10000.0,
        )

        # P&L% = (110 - 100) / 100 * 1.0 = 0.10 (10%)
        # Gross P&L = 10000 * 0.10 = $1000
        # Exit notional = 10000 * 1.0 * (110/100) = $11,000
        # Fee = 11000 * 0.001 = $11.00
        # Slippage = 11000 * 0.0005 = $5.50
        # Net P&L = 1000 - 11.00 - 5.50 = $983.50

        assert result.pnl_percent == pytest.approx(0.10)
        assert result.gross_pnl == pytest.approx(1000.0)
        assert result.exit_fee == pytest.approx(11.0)
        assert result.slippage_cost == pytest.approx(5.5)
        assert result.realized_pnl == pytest.approx(983.5)

    def test_small_fraction_exit(self, executor: PartialExitExecutor) -> None:
        """Test exiting small fraction (10%) of position."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=120.0,  # +20% profit
            position_side=PositionSide.LONG,
            exit_fraction=0.1,  # Exit 10%
            basis_balance=10000.0,
        )

        # P&L% = (120 - 100) / 100 * 0.1 = 0.02 (2%)
        # Gross P&L = 10000 * 0.02 = $200
        # Exit notional = 10000 * 0.1 * (120/100) = $1,200
        # Fee = 1200 * 0.001 = $1.20
        # Slippage = 1200 * 0.0005 = $0.60
        # Net P&L = 200 - 1.20 - 0.60 = $198.20

        assert result.pnl_percent == pytest.approx(0.02)
        assert result.gross_pnl == pytest.approx(200.0)
        assert result.realized_pnl == pytest.approx(198.2)

    def test_string_side_long(self, executor: PartialExitExecutor) -> None:
        """Test that string 'long' is handled correctly."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=110.0,
            position_side="long",  # String instead of enum
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        assert result.pnl_percent == pytest.approx(0.05)
        assert result.gross_pnl == pytest.approx(500.0)

    def test_string_side_short(self, executor: PartialExitExecutor) -> None:
        """Test that string 'short' is handled correctly."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=90.0,
            position_side="short",  # String instead of enum
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        assert result.pnl_percent == pytest.approx(0.05)
        assert result.gross_pnl == pytest.approx(500.0)

    def test_invalid_entry_price_raises(self, executor: PartialExitExecutor) -> None:
        """Test that non-positive entry price raises ValueError."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            executor.execute_partial_exit(
                entry_price=0.0,
                exit_price=110.0,
                position_side=PositionSide.LONG,
                exit_fraction=0.5,
                basis_balance=10000.0,
            )

    def test_invalid_exit_price_raises(self, executor: PartialExitExecutor) -> None:
        """Test that non-positive exit price raises ValueError."""
        with pytest.raises(ValueError, match="exit_price must be positive"):
            executor.execute_partial_exit(
                entry_price=100.0,
                exit_price=-10.0,
                position_side=PositionSide.LONG,
                exit_fraction=0.5,
                basis_balance=10000.0,
            )

    def test_invalid_exit_fraction_negative_raises(self, executor: PartialExitExecutor) -> None:
        """Test that negative exit fraction raises ValueError."""
        with pytest.raises(ValueError, match="exit_fraction must be in"):
            executor.execute_partial_exit(
                entry_price=100.0,
                exit_price=110.0,
                position_side=PositionSide.LONG,
                exit_fraction=-0.1,
                basis_balance=10000.0,
            )

    def test_invalid_exit_fraction_too_large_raises(self, executor: PartialExitExecutor) -> None:
        """Test that exit fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="exit_fraction must be in"):
            executor.execute_partial_exit(
                entry_price=100.0,
                exit_price=110.0,
                position_side=PositionSide.LONG,
                exit_fraction=1.1,
                basis_balance=10000.0,
            )

    def test_invalid_basis_balance_raises(self, executor: PartialExitExecutor) -> None:
        """Test that negative basis balance raises ValueError."""
        with pytest.raises(ValueError, match="basis_balance must be non-negative"):
            executor.execute_partial_exit(
                entry_price=100.0,
                exit_price=110.0,
                position_side=PositionSide.LONG,
                exit_fraction=0.5,
                basis_balance=-10000.0,
            )

    def test_invalid_side_string_raises(self, executor: PartialExitExecutor) -> None:
        """Test that invalid side string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid position side"):
            executor.execute_partial_exit(
                entry_price=100.0,
                exit_price=110.0,
                position_side="invalid",  # type: ignore
                exit_fraction=0.5,
                basis_balance=10000.0,
            )

    def test_breakeven_exit_has_only_costs(self, executor: PartialExitExecutor) -> None:
        """Test that breakeven exit (no price change) results in negative P&L due to costs."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=100.0,  # No price change
            position_side=PositionSide.LONG,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        # P&L% = 0
        # Gross P&L = 0
        # Exit notional = 10000 * 0.5 * (100/100) = $5,000
        # Fee = 5000 * 0.001 = $5.00
        # Slippage = 5000 * 0.0005 = $2.50
        # Net P&L = 0 - 5.00 - 2.50 = -$7.50

        assert result.pnl_percent == 0.0
        assert result.gross_pnl == 0.0
        assert result.exit_fee == pytest.approx(5.0)
        assert result.slippage_cost == pytest.approx(2.5)
        assert result.realized_pnl == pytest.approx(-7.5)

    def test_large_profit_exit(self, executor: PartialExitExecutor) -> None:
        """Test partial exit with large profit (3x gain)."""
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=300.0,  # 200% profit
            position_side=PositionSide.LONG,
            exit_fraction=0.25,  # Exit 25%
            basis_balance=10000.0,
        )

        # P&L% = (300 - 100) / 100 * 0.25 = 0.50 (50%)
        # Gross P&L = 10000 * 0.50 = $5000
        # Exit notional = 10000 * 0.25 * (300/100) = $7,500
        # Fee = 7500 * 0.001 = $7.50
        # Slippage = 7500 * 0.0005 = $3.75
        # Net P&L = 5000 - 7.50 - 3.75 = $4,988.75

        assert result.pnl_percent == pytest.approx(0.50)
        assert result.gross_pnl == pytest.approx(5000.0)
        assert result.exit_fee == pytest.approx(7.5)
        assert result.slippage_cost == pytest.approx(3.75)
        assert result.realized_pnl == pytest.approx(4988.75)

    def test_consistency_across_multiple_calls(self, executor: PartialExitExecutor) -> None:
        """Test that identical calls produce identical results."""
        params = {
            "entry_price": 100.0,
            "exit_price": 120.0,
            "position_side": PositionSide.LONG,
            "exit_fraction": 0.3,
            "basis_balance": 10000.0,
        }

        result1 = executor.execute_partial_exit(**params)
        result2 = executor.execute_partial_exit(**params)

        assert result1.realized_pnl == result2.realized_pnl
        assert result1.gross_pnl == result2.gross_pnl
        assert result1.exit_fee == result2.exit_fee
        assert result1.slippage_cost == result2.slippage_cost
        assert result1.pnl_percent == result2.pnl_percent
