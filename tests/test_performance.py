"""
Tests for performance metrics.

Performance metrics are critical for evaluating trading strategy effectiveness. Tests cover:
- Return calculations (absolute and percentage)
- Risk metrics (Sharpe ratio, drawdown, volatility)
- Trade analysis (win rate, profit factor, expectancy)
- Edge cases and data validation
- Mathematical correctness
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.performance.metrics import (
    Side, pnl_percent, cash_pnl, total_return, cagr, sharpe, max_drawdown
)


class TestBasicMetrics:
    """Test basic performance calculations"""

    def test_pnl_percent_calculation(self):
        """Test percentage P&L calculation"""
        # Test positive P&L
        entry_price = 100
        exit_price = 110
        pnl_pct = pnl_percent(entry_price, exit_price, Side.LONG)
        assert pnl_pct == 10.0  # 10% gain
        
        # Test negative P&L
        exit_price = 90
        pnl_pct = pnl_percent(entry_price, exit_price, Side.LONG)
        assert pnl_pct == -10.0  # 10% loss
        
        # Test short position
        entry_price = 100
        exit_price = 90
        pnl_pct = pnl_percent(entry_price, exit_price, Side.SHORT)
        assert pnl_pct == 10.0  # 10% gain on short

    def test_cash_pnl_calculation(self):
        """Test cash P&L calculation"""
        # Calculate percentage P&L first
        pnl_pct_long = pnl_percent(100, 110, Side.LONG)
        pnl_pct_short = pnl_percent(100, 90, Side.SHORT)
        
        # Convert to cash P&L
        balance = 1000
        cash_pnl_long = cash_pnl(pnl_pct_long, balance)
        cash_pnl_short = cash_pnl(pnl_pct_short, balance)
        
        assert cash_pnl_long == 100  # 10% of 1000
        assert cash_pnl_short == 100  # 10% of 1000

    def test_pnl_edge_cases(self):
        """Test P&L calculations with edge cases"""
        # Zero price change
        pnl_pct = pnl_percent(100, 100, Side.LONG)
        assert pnl_pct == 0.0
        
        cash_pnl_val = cash_pnl(0.0, 1000)
        assert cash_pnl_val == 0.0
        
        # Zero balance
        cash_pnl_val = cash_pnl(0.1, 0)
        assert cash_pnl_val == 0.0


class TestRiskMetrics:
    """Test risk-adjusted performance metrics"""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Create sample daily balance series
        daily_balance = pd.Series([1000, 1010, 1005, 1020, 1015, 1030, 1025, 1040])
        
        sharpe_ratio = sharpe(daily_balance)
        
        # Sharpe ratio should be a number
        assert isinstance(sharpe_ratio, float)
        assert not np.isnan(sharpe_ratio)
        
        # Test with constant balance (no volatility)
        constant_balance = pd.Series([1000, 1000, 1000, 1000, 1000])
        sharpe_constant = sharpe(constant_balance)
        assert sharpe_constant == 0.0

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create sample equity curve
        equity = pd.Series([100, 110, 105, 120, 115, 130, 125, 140])
        
        max_dd = max_drawdown(equity)
        
        # Max drawdown should be a percentage
        assert isinstance(max_dd, float)
        assert max_dd >= 0  # Returned as positive percentage
        
        # Test with constantly increasing equity
        increasing_equity = pd.Series([100, 110, 120, 130, 140])
        max_dd_inc = max_drawdown(increasing_equity)
        assert max_dd_inc == 0.0

    def test_total_return_calculation(self):
        """Test total return calculation"""
        initial_balance = 1000
        final_balance = 1200
        
        total_ret = total_return(initial_balance, final_balance)
        
        # Should be 20% return
        assert total_ret == 20.0
        
        # Test with loss
        final_balance_loss = 800
        total_ret_loss = total_return(initial_balance, final_balance_loss)
        assert total_ret_loss == -20.0

    def test_cagr_calculation(self):
        """Test Compound Annual Growth Rate calculation"""
        initial_balance = 1000
        final_balance = 1200
        days = 365
        
        cagr_value = cagr(initial_balance, final_balance, days)
        
        # Should be approximately 20% for 1 year
        assert abs(cagr_value - 20.0) < 1.0
        
        # Test with shorter period
        days_short = 180
        cagr_short = cagr(initial_balance, final_balance, days_short)
        assert cagr_short > 20.0  # Higher rate for shorter period


class TestPerformanceEdgeCases:
    """Test performance metrics edge cases"""

    def test_zero_initial_balance(self):
        """Test calculations with zero initial balance"""
        # Total return with zero initial balance
        total_ret = total_return(0, 1000)
        assert total_ret == 0.0
        
        # CAGR with zero initial balance
        cagr_value = cagr(0, 1000, 365)
        assert cagr_value == 0.0

    def test_zero_days_for_cagr(self):
        """Test CAGR with zero days"""
        cagr_value = cagr(1000, 1200, 0)
        assert cagr_value == 0.0

    def test_empty_series_for_sharpe(self):
        """Test Sharpe ratio with empty series"""
        empty_series = pd.Series([])
        sharpe_value = sharpe(empty_series)
        assert sharpe_value == 0.0

    def test_single_value_for_sharpe(self):
        """Test Sharpe ratio with single value"""
        single_series = pd.Series([1000])
        sharpe_value = sharpe(single_series)
        assert sharpe_value == 0.0

    def test_empty_series_for_drawdown(self):
        """Test max drawdown with empty series"""
        empty_series = pd.Series([])
        max_dd = max_drawdown(empty_series)
        assert max_dd == 0.0


class TestPerformanceIntegration:
    """Test performance metrics integration and edge cases"""

    def test_comprehensive_performance_analysis(self):
        """Test comprehensive performance analysis with realistic data"""
        # Create realistic daily balance data
        np.random.seed(42)
        n_days = 100
        
        # Generate realistic daily balance progression
        initial_balance = 10000
        daily_returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% volatility
        daily_balance = [initial_balance]
        
        for ret in daily_returns:
            daily_balance.append(daily_balance[-1] * (1 + ret))
        
        balance_series = pd.Series(daily_balance)
        
        # Calculate all available metrics
        total_ret = total_return(initial_balance, balance_series.iloc[-1])
        cagr_value = cagr(initial_balance, balance_series.iloc[-1], n_days)
        sharpe_ratio = sharpe(balance_series)
        max_dd = max_drawdown(balance_series)
        
        # All metrics should be reasonable
        assert isinstance(total_ret, float)
        assert isinstance(cagr_value, float)
        assert isinstance(sharpe_ratio, float)
        assert isinstance(max_dd, float)
        assert max_dd >= 0  # Drawdown as positive percentage

    def test_performance_consistency(self):
        """Test that performance metrics are consistent"""
        # Create sample data
        balance_series = pd.Series([1000, 1100, 1050, 1200, 1150, 1300])
        
        # Calculate metrics multiple times
        total_ret_1 = total_return(1000, 1300)
        total_ret_2 = total_return(1000, 1300)
        
        sharpe_1 = sharpe(balance_series)
        sharpe_2 = sharpe(balance_series)
        
        # Results should be identical
        assert total_ret_1 == total_ret_2
        assert sharpe_1 == sharpe_2

    def test_performance_with_extreme_values(self):
        """Test performance metrics with extreme values"""
        # Very large balance changes
        extreme_balance = pd.Series([1000, 1000000, 500000, 2000000])
        
        sharpe_extreme = sharpe(extreme_balance)
        max_dd_extreme = max_drawdown(extreme_balance)
        
        # Should handle extreme values without errors
        assert isinstance(sharpe_extreme, float)
        assert isinstance(max_dd_extreme, float)
        assert max_dd_extreme >= 0