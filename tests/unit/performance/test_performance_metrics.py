import pytest
import pandas as pd
import numpy as np

from performance.metrics import Side, pnl_percent, cash_pnl, total_return, cagr, sharpe, max_drawdown

pytestmark = pytest.mark.unit


class TestBasicMetrics:
    def test_pnl_percent_calculation(self):
        assert pnl_percent(100, 110, Side.LONG) == 0.1
        assert pnl_percent(100, 90, Side.LONG) == -0.1
        assert pnl_percent(100, 90, Side.SHORT) == 0.1

    def test_cash_pnl_calculation(self):
        pnl_pct_long = pnl_percent(100, 110, Side.LONG)
        pnl_pct_short = pnl_percent(100, 90, Side.SHORT)
        balance = 1000
        assert cash_pnl(pnl_pct_long, balance) == 100
        assert cash_pnl(pnl_pct_short, balance) == 100

    def test_pnl_edge_cases(self):
        assert pnl_percent(100, 100, Side.LONG) == 0.0
        assert cash_pnl(0.0, 1000) == 0.0
        assert cash_pnl(0.1, 0) == 0.0


class TestRiskMetrics:
    def test_sharpe_ratio_calculation(self):
        daily_balance = pd.Series([1000, 1010, 1005, 1020, 1015, 1030, 1025, 1040])
        sharpe_ratio = sharpe(daily_balance)
        assert isinstance(sharpe_ratio, float)
        assert not np.isnan(sharpe_ratio)
        constant_balance = pd.Series([1000, 1000, 1000, 1000, 1000])
        assert sharpe(constant_balance) == 0.0

    def test_max_drawdown_calculation(self):
        equity = pd.Series([100, 110, 105, 120, 115, 130, 125, 140])
        max_dd = max_drawdown(equity)
        assert isinstance(max_dd, float)
        assert max_dd >= 0
        increasing_equity = pd.Series([100, 110, 120, 130, 140])
        assert max_drawdown(increasing_equity) == 0.0

    def test_total_return_calculation(self):
        assert total_return(1000, 1200) == pytest.approx(20.0)
        assert total_return(1000, 800) == pytest.approx(-20.0)

    def test_cagr_calculation(self):
        assert abs(cagr(1000, 1200, 365) - 20.0) < 1.0
        assert cagr(1000, 1200, 180) > 20.0


class TestPerformanceEdgeCases:
    def test_zero_initial_balance(self):
        assert total_return(0, 1000) == 0.0
        assert cagr(0, 1000, 365) == 0.0

    def test_zero_days_for_cagr(self):
        assert cagr(1000, 1200, 0) == 0.0

    def test_empty_series_for_sharpe(self):
        assert sharpe(pd.Series([])) == 0.0

    def test_single_value_for_sharpe(self):
        assert sharpe(pd.Series([1000])) == 0.0

    def test_empty_series_for_drawdown(self):
        assert max_drawdown(pd.Series([])) == 0.0


class TestPerformanceIntegration:
    def test_comprehensive_performance_analysis(self):
        np.random.seed(42)
        n_days = 100
        initial_balance = 10000
        daily_returns = np.random.normal(0.001, 0.02, n_days)
        daily_balance = [initial_balance]
        for ret in daily_returns:
            daily_balance.append(daily_balance[-1] * (1 + ret))
        balance_series = pd.Series(daily_balance)
        total_ret = total_return(initial_balance, balance_series.iloc[-1])
        cagr_value = cagr(initial_balance, balance_series.iloc[-1], n_days)
        sharpe_ratio = sharpe(balance_series)
        max_dd = max_drawdown(balance_series)
        assert isinstance(total_ret, float)
        assert isinstance(cagr_value, float)
        assert isinstance(sharpe_ratio, float)
        assert isinstance(max_dd, float)
        assert max_dd >= 0

    def test_performance_consistency(self):
        balance_series = pd.Series([1000, 1100, 1050, 1200, 1150, 1300])
        total_ret_1 = total_return(1000, 1300)
        total_ret_2 = total_return(1000, 1300)
        sharpe_1 = sharpe(balance_series)
        sharpe_2 = sharpe(balance_series)
        assert total_ret_1 == total_ret_2
        assert sharpe_1 == sharpe_2

    def test_performance_with_extreme_values(self):
        extreme_balance = pd.Series([1000, 1000000, 500000, 2000000])
        sharpe_extreme = sharpe(extreme_balance)
        max_dd_extreme = max_drawdown(extreme_balance)
        assert isinstance(sharpe_extreme, float)
        assert isinstance(max_dd_extreme, float)
        assert max_dd_extreme >= 0