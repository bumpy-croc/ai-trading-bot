"""Comprehensive validation tests for performance module.

Tests all input validation paths to ensure robust error handling.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src.performance.metrics import (
    Side,
    calmar_ratio,
    cash_pnl,
    cagr,
    expectancy,
    pnl_percent,
    sortino_ratio,
    total_return,
    value_at_risk,
)
from src.performance.tracker import PerformanceTracker


class TestPnlPercentValidation:
    """Test input validation for pnl_percent function."""

    def test_negative_entry_price_raises_error(self):
        with pytest.raises(ValueError, match="entry_price must be positive"):
            pnl_percent(-100, 105, Side.LONG)

    def test_zero_entry_price_raises_error(self):
        with pytest.raises(ValueError, match="entry_price must be positive"):
            pnl_percent(0, 105, Side.LONG)

    def test_negative_exit_price_raises_error(self):
        with pytest.raises(ValueError, match="exit_price must be positive"):
            pnl_percent(100, -105, Side.LONG)

    def test_zero_exit_price_raises_error(self):
        with pytest.raises(ValueError, match="exit_price must be positive"):
            pnl_percent(100, 0, Side.LONG)

    def test_infinity_entry_price_raises_error(self):
        with pytest.raises(ValueError, match="Prices must be finite"):
            pnl_percent(float("inf"), 105, Side.LONG)

    def test_infinity_exit_price_raises_error(self):
        with pytest.raises(ValueError, match="Prices must be finite"):
            pnl_percent(100, float("inf"), Side.LONG)

    def test_nan_entry_price_raises_error(self):
        with pytest.raises(ValueError, match="Prices must be finite"):
            pnl_percent(float("nan"), 105, Side.LONG)

    def test_nan_exit_price_raises_error(self):
        with pytest.raises(ValueError, match="Prices must be finite"):
            pnl_percent(100, float("nan"), Side.LONG)

    def test_fraction_below_zero_raises_error(self):
        with pytest.raises(ValueError, match="fraction must be in"):
            pnl_percent(100, 105, Side.LONG, fraction=-0.1)

    def test_fraction_above_one_raises_error(self):
        with pytest.raises(ValueError, match="fraction must be in"):
            pnl_percent(100, 105, Side.LONG, fraction=1.5)

    def test_valid_inputs_succeeds(self):
        result = pnl_percent(100, 105, Side.LONG, fraction=0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(0.025)  # 5% move * 50% position


class TestCashPnlValidation:
    """Test input validation for cash_pnl function."""

    def test_infinity_pnl_pct_raises_error(self):
        with pytest.raises(ValueError, match="pnl_pct must be finite"):
            cash_pnl(float("inf"), 10000)

    def test_nan_pnl_pct_raises_error(self):
        with pytest.raises(ValueError, match="pnl_pct must be finite"):
            cash_pnl(float("nan"), 10000)

    def test_negative_balance_raises_error(self):
        with pytest.raises(ValueError, match="balance_before must be non-negative"):
            cash_pnl(0.05, -10000)

    def test_infinity_balance_raises_error(self):
        with pytest.raises(ValueError, match="balance_before must be finite"):
            cash_pnl(0.05, float("inf"))

    def test_nan_balance_raises_error(self):
        with pytest.raises(ValueError, match="balance_before must be finite"):
            cash_pnl(0.05, float("nan"))

    def test_valid_inputs_succeeds(self):
        result = cash_pnl(0.05, 10000)
        assert result == pytest.approx(500.0)


class TestTotalReturnValidation:
    """Test input validation for total_return function."""

    def test_zero_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            total_return(0, 11000)

    def test_negative_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            total_return(-10000, 11000)

    def test_negative_final_balance_raises_error(self):
        with pytest.raises(ValueError, match="final_balance must be non-negative"):
            total_return(10000, -500)

    def test_infinity_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="Balances must be finite"):
            total_return(float("inf"), 11000)

    def test_infinity_final_balance_raises_error(self):
        with pytest.raises(ValueError, match="Balances must be finite"):
            total_return(10000, float("inf"))

    def test_nan_balances_raise_error(self):
        with pytest.raises(ValueError, match="Balances must be finite"):
            total_return(float("nan"), 11000)

    def test_valid_inputs_succeeds(self):
        result = total_return(10000, 11000)
        assert result == pytest.approx(10.0)  # 10% return


class TestCagrValidation:
    """Test input validation for cagr function."""

    def test_zero_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            cagr(0, 11000, 365)

    def test_negative_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            cagr(-10000, 11000, 365)

    def test_negative_final_balance_raises_error(self):
        with pytest.raises(ValueError, match="final_balance must be non-negative"):
            cagr(10000, -500, 365)

    def test_infinity_balances_raise_error(self):
        with pytest.raises(ValueError, match="Balances must be finite"):
            cagr(float("inf"), 11000, 365)

    def test_nan_balances_raise_error(self):
        with pytest.raises(ValueError, match="Balances must be finite"):
            cagr(10000, float("nan"), 365)

    def test_zero_days_returns_zero(self):
        result = cagr(10000, 11000, 0)
        assert result == 0.0

    def test_negative_days_returns_zero(self):
        result = cagr(10000, 11000, -365)
        assert result == 0.0

    def test_valid_inputs_succeeds(self):
        result = cagr(10000, 11000, 365)
        assert result > 0  # Positive return


class TestSortinoRatioValidation:
    """Test input validation for sortino_ratio function."""

    def test_infinity_risk_free_rate_raises_error(self):
        balance = pd.Series([10000, 10100, 10200, 10150, 10300])
        with pytest.raises(ValueError, match="risk_free_rate must be finite"):
            sortino_ratio(balance, risk_free_rate=float("inf"))

    def test_nan_risk_free_rate_raises_error(self):
        balance = pd.Series([10000, 10100, 10200, 10150, 10300])
        with pytest.raises(ValueError, match="risk_free_rate must be finite"):
            sortino_ratio(balance, risk_free_rate=float("nan"))

    def test_valid_inputs_succeeds(self):
        balance = pd.Series([10000, 10100, 10200, 10150, 10300])
        result = sortino_ratio(balance, risk_free_rate=0.02)
        assert isinstance(result, float)


class TestCalmarRatioValidation:
    """Test input validation for calmar_ratio function."""

    def test_infinity_annualized_return_raises_error(self):
        with pytest.raises(ValueError, match="Parameters must be finite"):
            calmar_ratio(float("inf"), 20.0)

    def test_nan_annualized_return_raises_error(self):
        with pytest.raises(ValueError, match="Parameters must be finite"):
            calmar_ratio(float("nan"), 20.0)

    def test_infinity_max_drawdown_raises_error(self):
        with pytest.raises(ValueError, match="Parameters must be finite"):
            calmar_ratio(15.0, float("inf"))

    def test_nan_max_drawdown_raises_error(self):
        with pytest.raises(ValueError, match="Parameters must be finite"):
            calmar_ratio(15.0, float("nan"))

    def test_negative_max_drawdown_raises_error(self):
        with pytest.raises(ValueError, match="max_drawdown_pct must be non-negative"):
            calmar_ratio(15.0, -5.0)

    def test_zero_drawdown_returns_capped_value(self):
        result = calmar_ratio(15.0, 0.0)
        assert result == 999.0  # Capped, not infinity

    def test_valid_inputs_succeeds(self):
        result = calmar_ratio(15.0, 20.0)
        assert result == pytest.approx(0.75)


class TestValueAtRiskValidation:
    """Test input validation for value_at_risk function."""

    def test_confidence_below_zero_raises_error(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        with pytest.raises(ValueError, match="confidence must be in"):
            value_at_risk(returns, confidence=0.0)

    def test_confidence_equal_one_raises_error(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        with pytest.raises(ValueError, match="confidence must be in"):
            value_at_risk(returns, confidence=1.0)

    def test_confidence_above_one_raises_error(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        with pytest.raises(ValueError, match="confidence must be in"):
            value_at_risk(returns, confidence=1.5)

    def test_valid_inputs_succeeds(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        result = value_at_risk(returns, confidence=0.95)
        assert isinstance(result, float)


class TestExpectancyValidation:
    """Test input validation for expectancy function."""

    def test_win_rate_below_zero_raises_error(self):
        with pytest.raises(ValueError, match="win_rate must be in"):
            expectancy(-0.1, 100, -50)

    def test_win_rate_above_one_raises_error(self):
        with pytest.raises(ValueError, match="win_rate must be in"):
            expectancy(1.5, 100, -50)

    def test_positive_avg_loss_raises_error(self):
        with pytest.raises(ValueError, match="avg_loss must be negative or zero"):
            expectancy(0.6, 100, 50)  # avg_loss should be negative

    def test_negative_avg_win_raises_error(self):
        with pytest.raises(ValueError, match="avg_win must be non-negative"):
            expectancy(0.6, -100, -50)

    def test_infinity_avg_win_raises_error(self):
        with pytest.raises(ValueError, match="avg_win and avg_loss must be finite"):
            expectancy(0.6, float("inf"), -50)

    def test_nan_avg_loss_raises_error(self):
        with pytest.raises(ValueError, match="avg_win and avg_loss must be finite"):
            expectancy(0.6, 100, float("nan"))

    def test_valid_inputs_succeeds(self):
        result = expectancy(0.6, 100, -50)
        assert isinstance(result, float)
        assert result == pytest.approx(40.0)  # 0.6*100 + 0.4*(-50)


class TestPerformanceTrackerValidation:
    """Test input validation for PerformanceTracker."""

    def test_zero_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            PerformanceTracker(initial_balance=0)

    def test_negative_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            PerformanceTracker(initial_balance=-10000)

    def test_infinity_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be finite"):
            PerformanceTracker(initial_balance=float("inf"))

    def test_nan_initial_balance_raises_error(self):
        with pytest.raises(ValueError, match="initial_balance must be finite"):
            PerformanceTracker(initial_balance=float("nan"))

    def test_negative_fee_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = 100
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="fee must be non-negative and finite"):
            tracker.record_trade(trade, fee=-10)

    def test_infinity_fee_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = 100
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="fee must be non-negative and finite"):
            tracker.record_trade(trade, fee=float("inf"))

    def test_negative_slippage_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = 100
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="slippage must be non-negative and finite"):
            tracker.record_trade(trade, slippage=-5)

    def test_infinity_slippage_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = 100
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="slippage must be non-negative and finite"):
            tracker.record_trade(trade, slippage=float("inf"))

    def test_none_pnl_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = None
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="Cannot record trade with None PnL"):
            tracker.record_trade(trade)

    def test_infinity_pnl_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = float("inf")
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="has non-finite PnL"):
            tracker.record_trade(trade)

    def test_nan_pnl_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        trade = Mock()
        trade.pnl = float("nan")
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        with pytest.raises(ValueError, match="has non-finite PnL"):
            tracker.record_trade(trade)

    def test_negative_balance_update_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        with pytest.raises(ValueError, match="balance must be non-negative"):
            tracker.update_balance(-500)

    def test_infinity_balance_update_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        with pytest.raises(ValueError, match="balance must be finite"):
            tracker.update_balance(float("inf"))

    def test_nan_balance_update_raises_error(self):
        tracker = PerformanceTracker(initial_balance=10000)
        with pytest.raises(ValueError, match="balance must be finite"):
            tracker.update_balance(float("nan"))

    def test_valid_operations_succeed(self):
        tracker = PerformanceTracker(initial_balance=10000)

        trade = Mock()
        trade.pnl = 100
        trade.entry_time = datetime.now(UTC)
        trade.exit_time = datetime.now(UTC)
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        tracker.record_trade(trade, fee=10, slippage=5)
        tracker.update_balance(10100)

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 1
        assert metrics.total_fees_paid == 10
        assert metrics.total_slippage_cost == 5
