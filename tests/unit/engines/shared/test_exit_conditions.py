"""Unit tests for shared exit conditions module."""

from datetime import datetime

import pytest

from src.engines.shared.exit_conditions import (
    ExitConditionResult,
    calculate_pnl_percent,
    calculate_sized_pnl_percent,
    check_stop_loss,
    check_take_profit,
)
from src.engines.shared.models import BasePosition, PositionSide


class TestCheckStopLoss:
    """Tests for check_stop_loss function."""

    @pytest.fixture
    def long_position(self) -> BasePosition:
        """Create a long position for testing."""
        return BasePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_time=datetime.now(),
            size=0.1,
            stop_loss=95.0,
        )

    @pytest.fixture
    def short_position(self) -> BasePosition:
        """Create a short position for testing."""
        return BasePosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=100.0,
            entry_time=datetime.now(),
            size=0.1,
            stop_loss=105.0,
        )

    def test_long_sl_triggered_when_low_breaches(self, long_position: BasePosition) -> None:
        """Long stop loss should trigger when candle low breaches SL level."""
        result = check_stop_loss(
            long_position, current_price=96.0, candle_high=101.0, candle_low=94.0
        )
        assert result.triggered is True
        assert result.exit_price == 95.0  # max(SL, low)

    def test_long_sl_not_triggered_when_low_above_sl(
        self, long_position: BasePosition
    ) -> None:
        """Long stop loss should not trigger when candle low stays above SL."""
        result = check_stop_loss(
            long_position, current_price=96.0, candle_high=101.0, candle_low=96.0
        )
        assert result.triggered is False
        assert result.exit_price is None

    def test_short_sl_triggered_when_high_breaches(
        self, short_position: BasePosition
    ) -> None:
        """Short stop loss should trigger when candle high breaches SL level."""
        result = check_stop_loss(
            short_position, current_price=104.0, candle_high=106.0, candle_low=103.0
        )
        assert result.triggered is True
        assert result.exit_price == 105.0  # min(SL, high)

    def test_short_sl_not_triggered_when_high_below_sl(
        self, short_position: BasePosition
    ) -> None:
        """Short stop loss should not trigger when candle high stays below SL."""
        result = check_stop_loss(
            short_position, current_price=102.0, candle_high=104.0, candle_low=100.0
        )
        assert result.triggered is False
        assert result.exit_price is None

    def test_sl_with_no_stop_loss_set(self, long_position: BasePosition) -> None:
        """Should return not triggered when no stop loss is set."""
        long_position.stop_loss = None
        result = check_stop_loss(long_position, current_price=90.0)
        assert result.triggered is False

    def test_sl_fallback_to_close_price_when_no_high_low(
        self, long_position: BasePosition
    ) -> None:
        """Should use close price when high/low not available."""
        # Close below SL
        result = check_stop_loss(
            long_position, current_price=94.0, use_high_low=False
        )
        assert result.triggered is True

        # Close above SL
        result = check_stop_loss(
            long_position, current_price=96.0, use_high_low=False
        )
        assert result.triggered is False


class TestCheckTakeProfit:
    """Tests for check_take_profit function."""

    @pytest.fixture
    def long_position(self) -> BasePosition:
        """Create a long position for testing."""
        return BasePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_time=datetime.now(),
            size=0.1,
            take_profit=110.0,
        )

    @pytest.fixture
    def short_position(self) -> BasePosition:
        """Create a short position for testing."""
        return BasePosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=100.0,
            entry_time=datetime.now(),
            size=0.1,
            take_profit=90.0,
        )

    def test_long_tp_triggered_when_high_reaches(
        self, long_position: BasePosition
    ) -> None:
        """Long take profit should trigger when candle high reaches TP level."""
        result = check_take_profit(
            long_position, current_price=108.0, candle_high=111.0, candle_low=105.0
        )
        assert result.triggered is True
        assert result.exit_price == 110.0

    def test_long_tp_not_triggered_when_high_below_tp(
        self, long_position: BasePosition
    ) -> None:
        """Long take profit should not trigger when candle high stays below TP."""
        result = check_take_profit(
            long_position, current_price=108.0, candle_high=109.0, candle_low=106.0
        )
        assert result.triggered is False
        assert result.exit_price is None

    def test_short_tp_triggered_when_low_reaches(
        self, short_position: BasePosition
    ) -> None:
        """Short take profit should trigger when candle low reaches TP level."""
        result = check_take_profit(
            short_position, current_price=92.0, candle_high=95.0, candle_low=89.0
        )
        assert result.triggered is True
        assert result.exit_price == 90.0

    def test_short_tp_not_triggered_when_low_above_tp(
        self, short_position: BasePosition
    ) -> None:
        """Short take profit should not trigger when candle low stays above TP."""
        result = check_take_profit(
            short_position, current_price=92.0, candle_high=95.0, candle_low=91.0
        )
        assert result.triggered is False

    def test_tp_with_no_take_profit_set(self, long_position: BasePosition) -> None:
        """Should return not triggered when no take profit is set."""
        long_position.take_profit = None
        result = check_take_profit(long_position, current_price=120.0)
        assert result.triggered is False


class TestCalculatePnlPercent:
    """Tests for calculate_pnl_percent function."""

    def test_long_position_profit(self) -> None:
        """Long position with price increase should show positive P&L."""
        pnl = calculate_pnl_percent(
            entry_price=100.0, current_price=110.0, side=PositionSide.LONG
        )
        assert abs(pnl - 0.10) < 0.0001  # 10% profit

    def test_long_position_loss(self) -> None:
        """Long position with price decrease should show negative P&L."""
        pnl = calculate_pnl_percent(
            entry_price=100.0, current_price=90.0, side=PositionSide.LONG
        )
        assert abs(pnl - (-0.10)) < 0.0001  # 10% loss

    def test_short_position_profit(self) -> None:
        """Short position with price decrease should show positive P&L."""
        pnl = calculate_pnl_percent(
            entry_price=100.0, current_price=90.0, side=PositionSide.SHORT
        )
        assert abs(pnl - 0.10) < 0.0001  # 10% profit

    def test_short_position_loss(self) -> None:
        """Short position with price increase should show negative P&L."""
        pnl = calculate_pnl_percent(
            entry_price=100.0, current_price=110.0, side=PositionSide.SHORT
        )
        assert abs(pnl - (-0.10)) < 0.0001  # 10% loss

    def test_string_side_long(self) -> None:
        """Should handle string side 'long'."""
        pnl = calculate_pnl_percent(
            entry_price=100.0, current_price=105.0, side="long"
        )
        assert abs(pnl - 0.05) < 0.0001

    def test_string_side_short(self) -> None:
        """Should handle string side 'short'."""
        pnl = calculate_pnl_percent(
            entry_price=100.0, current_price=95.0, side="short"
        )
        assert abs(pnl - 0.05) < 0.0001

    def test_zero_entry_price_returns_zero(self) -> None:
        """Should return 0 when entry price is zero to avoid division error."""
        pnl = calculate_pnl_percent(
            entry_price=0.0, current_price=100.0, side=PositionSide.LONG
        )
        assert pnl == 0.0


class TestCalculateSizedPnlPercent:
    """Tests for calculate_sized_pnl_percent function."""

    def test_sized_pnl_for_full_position(self) -> None:
        """Sized P&L with 100% position should equal raw P&L."""
        sized_pnl = calculate_sized_pnl_percent(
            entry_price=100.0,
            current_price=110.0,
            side=PositionSide.LONG,
            position_size=1.0,
        )
        assert abs(sized_pnl - 0.10) < 0.0001

    def test_sized_pnl_for_partial_position(self) -> None:
        """Sized P&L with 25% position should be 1/4 of raw P&L."""
        sized_pnl = calculate_sized_pnl_percent(
            entry_price=100.0,
            current_price=110.0,
            side=PositionSide.LONG,
            position_size=0.25,
        )
        assert abs(sized_pnl - 0.025) < 0.0001  # 10% * 0.25 = 2.5%
