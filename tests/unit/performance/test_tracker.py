"""Unit tests for src.performance.tracker module."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.performance.tracker import PerformanceMetrics, PerformanceTracker


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_default_initialization(self):
        """Test PerformanceMetrics can be created with defaults."""
        metrics = PerformanceMetrics()

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0
        assert metrics.var_95 == 0.0
        assert metrics.expectancy == 0.0

    def test_to_dict_conversion(self):
        """Test PerformanceMetrics.to_dict() includes all fields."""
        metrics = PerformanceMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            var_95=-0.02,
            expectancy=50.0,
            consecutive_wins=3,
            consecutive_losses=2,
        )

        result = metrics.to_dict()

        assert result["total_trades"] == 10
        assert result["winning_trades"] == 6
        assert result["losing_trades"] == 4
        assert result["win_rate"] == 0.6
        assert result["sharpe_ratio"] == 1.5
        assert result["sortino_ratio"] == 2.0
        assert result["calmar_ratio"] == 0.8
        assert result["var_95"] == -0.02
        assert result["expectancy"] == 50.0
        assert result["consecutive_wins"] == 3
        assert result["consecutive_losses"] == 2


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_initialization(self):
        """Test PerformanceTracker initializes correctly."""
        tracker = PerformanceTracker(initial_balance=10000)

        assert tracker.initial_balance == 10000
        assert tracker.current_balance == 10000
        assert tracker.peak_balance == 10000
        assert tracker.max_drawdown == 0.0

    def test_record_winning_trade(self):
        """Test recording a winning trade updates statistics."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Create mock trade
        trade = Mock()
        trade.pnl = 100.0
        trade.entry_time = datetime.now() - timedelta(hours=2)
        trade.exit_time = datetime.now()
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        tracker.record_trade(trade, fee=1.0, slippage=0.5)

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 1.0
        assert metrics.total_pnl == 100.0
        assert metrics.total_fees_paid == 1.0
        assert metrics.total_slippage_cost == 0.5
        assert metrics.consecutive_wins == 1
        assert metrics.consecutive_losses == 0

    def test_record_losing_trade(self):
        """Test recording a losing trade updates statistics."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Create mock trade
        trade = Mock()
        trade.pnl = -50.0
        trade.entry_time = datetime.now() - timedelta(hours=1)
        trade.exit_time = datetime.now()
        trade.symbol = "ETHUSDT"
        trade.side = "short"

        tracker.record_trade(trade, fee=0.5, slippage=0.2)

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == -50.0
        assert metrics.total_fees_paid == 0.5
        assert metrics.total_slippage_cost == 0.2
        assert metrics.consecutive_wins == 0
        assert metrics.consecutive_losses == 1

    def test_consecutive_win_streak(self):
        """Test consecutive win streak tracking."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Record 3 winning trades
        for i in range(3):
            trade = Mock()
            trade.pnl = 50.0 + i * 10
            trade.entry_time = datetime.now() - timedelta(hours=i + 1)
            trade.exit_time = datetime.now() - timedelta(hours=i)
            trade.symbol = "BTCUSDT"
            trade.side = "long"
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.consecutive_wins == 3
        assert metrics.consecutive_losses == 0

    def test_consecutive_loss_streak(self):
        """Test consecutive loss streak tracking."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Record 4 losing trades
        for i in range(4):
            trade = Mock()
            trade.pnl = -30.0 - i * 5
            trade.entry_time = datetime.now() - timedelta(hours=i + 1)
            trade.exit_time = datetime.now() - timedelta(hours=i)
            trade.symbol = "ETHUSDT"
            trade.side = "short"
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.consecutive_wins == 0
        assert metrics.consecutive_losses == 4

    def test_streak_reset_on_opposite_outcome(self):
        """Test streak resets when outcome changes."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Win, win, lose
        for pnl in [100, 50, -30]:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            trade.symbol = "BTCUSDT"
            trade.side = "long"
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.consecutive_wins == 0  # Reset by loss
        assert metrics.consecutive_losses == 1

    def test_update_balance_calculates_drawdown(self):
        """Test balance updates calculate current and max drawdown."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Update to new peak
        tracker.update_balance(12000)
        assert tracker.peak_balance == 12000
        assert tracker.max_drawdown == 0.0

        # Drop from peak
        tracker.update_balance(10800)
        metrics = tracker.get_metrics()
        assert metrics.current_drawdown == (12000 - 10800) / 12000
        assert metrics.max_drawdown == (12000 - 10800) / 12000

        # Further drop
        tracker.update_balance(9600)
        metrics = tracker.get_metrics()
        assert metrics.current_drawdown == (12000 - 9600) / 12000
        assert metrics.max_drawdown == (12000 - 9600) / 12000

        # Recovery but not to peak
        tracker.update_balance(11000)
        metrics = tracker.get_metrics()
        assert metrics.current_drawdown == (12000 - 11000) / 12000
        # Max drawdown should remain at lowest point
        assert metrics.max_drawdown == (12000 - 9600) / 12000

    def test_get_balance_series_returns_pandas_series(self):
        """Test get_balance_series returns properly formatted pandas Series."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Add balance updates
        now = datetime.now()
        for i in range(5):
            tracker.update_balance(10000 + i * 100, timestamp=now + timedelta(days=i))

        series = tracker.get_balance_series()

        assert isinstance(series, pd.Series)
        assert isinstance(series.index, pd.DatetimeIndex)
        assert len(series) >= 5  # May be more after daily resampling

    def test_profit_factor_calculation(self):
        """Test profit factor is calculated correctly."""
        tracker = PerformanceTracker(initial_balance=10000)

        # 2 wins totaling $300
        for pnl in [100, 200]:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        # 3 losses totaling $150
        for pnl in [-50, -50, -50]:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        # Profit factor = gross profit / gross loss = 300 / 150 = 2.0
        assert metrics.profit_factor == pytest.approx(2.0, abs=0.01)

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        tracker = PerformanceTracker(initial_balance=10000)

        # 3 wins, 2 losses
        pnls = [100, -50, 75, -25, 50]
        for pnl in pnls:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == pytest.approx(0.6, abs=0.01)

    def test_average_win_and_loss(self):
        """Test average win and average loss calculations."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Wins: 100, 200 -> avg = 150
        # Losses: -50, -150 -> avg = -100
        pnls = [100, -50, 200, -150]
        for pnl in pnls:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.avg_win == pytest.approx(150.0, abs=0.01)
        assert metrics.avg_loss == pytest.approx(-100.0, abs=0.01)

    def test_largest_win_and_loss(self):
        """Test largest win and loss tracking."""
        tracker = PerformanceTracker(initial_balance=10000)

        pnls = [100, -50, 300, -200, 150]
        for pnl in pnls:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.largest_win == 300.0
        assert metrics.largest_loss == -200.0

    def test_average_trade_duration(self):
        """Test average trade duration calculation."""
        tracker = PerformanceTracker(initial_balance=10000)

        # 2 trades: 1 hour and 3 hours -> avg = 2 hours
        durations = [1, 3]
        for hours in durations:
            trade = Mock()
            trade.pnl = 50.0
            trade.entry_time = datetime.now() - timedelta(hours=hours)
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.avg_trade_duration_hours == pytest.approx(2.0, abs=0.1)

    def test_total_return_calculation(self):
        """Test total return percentage calculation."""
        tracker = PerformanceTracker(initial_balance=10000)

        tracker.update_balance(11000)
        metrics = tracker.get_metrics()

        # Total return should be 10%
        assert metrics.total_return_pct == pytest.approx(10.0, abs=0.1)

    def test_get_trade_history(self):
        """Test get_trade_history returns recorded trades."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Record 3 trades
        for i in range(3):
            trade = Mock()
            trade.pnl = 100.0 * (i + 1)
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            trade.symbol = f"BTC{i}"
            trade.side = "long"
            tracker.record_trade(trade, fee=1.0)

        history = tracker.get_trade_history()
        assert len(history) == 3
        assert history[0]["pnl"] == 100.0
        assert history[1]["pnl"] == 200.0
        assert history[2]["pnl"] == 300.0

    def test_reset_clears_all_state(self):
        """Test reset() clears all tracking data."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Add some trades and balance updates
        for i in range(3):
            trade = Mock()
            trade.pnl = 100.0
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        tracker.update_balance(11000)

        # Reset
        tracker.reset()

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 0
        assert metrics.total_pnl == 0.0
        assert tracker.current_balance == 10000
        assert tracker.peak_balance == 10000
        assert tracker.max_drawdown == 0.0

    def test_reset_with_new_balance(self):
        """Test reset() with new initial balance."""
        tracker = PerformanceTracker(initial_balance=10000)

        tracker.update_balance(12000)
        tracker.reset(initial_balance=15000)

        assert tracker.initial_balance == 15000
        assert tracker.current_balance == 15000
        assert tracker.peak_balance == 15000

    def test_zero_trades_returns_safe_metrics(self):
        """Test get_metrics() with zero trades doesn't raise errors."""
        tracker = PerformanceTracker(initial_balance=10000)

        metrics = tracker.get_metrics()

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0

    def test_trade_without_timestamps(self):
        """Test recording trade without entry/exit times."""
        tracker = PerformanceTracker(initial_balance=10000)

        trade = Mock()
        trade.pnl = 100.0
        trade.entry_time = None
        trade.exit_time = None
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        # Should not raise error
        tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 1
        assert metrics.avg_trade_duration_hours == 0.0

    def test_expectancy_calculation(self):
        """Test expectancy metric calculation."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Win rate: 60%, Avg win: 100, Avg loss: -50
        # Expectancy = 0.6 * 100 + 0.4 * -50 = 60 - 20 = 40
        for pnl in [100, -50, 100, -50, 100]:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.expectancy == pytest.approx(40.0, abs=1.0)

    def test_thread_safety_concurrent_updates(self):
        """Test thread safety with concurrent balance updates."""
        import threading

        tracker = PerformanceTracker(initial_balance=10000)

        def update_balance():
            for i in range(100):
                tracker.update_balance(10000 + i)

        threads = [threading.Thread(target=update_balance) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not raise any errors
        metrics = tracker.get_metrics()
        assert metrics.current_balance >= 10000


@pytest.mark.fast
class TestPerformanceMetricsFunctions:
    """Tests for integration with pure metric functions."""

    def test_sharpe_ratio_calculated_from_balance_series(self):
        """Test Sharpe ratio is calculated using balance series."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Simulate daily balance growth
        now = datetime.now()
        for i in range(30):
            balance = 10000 + i * 100  # Linear growth
            tracker.update_balance(balance, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # Should have positive Sharpe ratio for consistent growth
        assert metrics.sharpe_ratio > 0

    def test_sortino_ratio_calculated_from_balance_series(self):
        """Test Sortino ratio is calculated using balance series."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Simulate volatile but upward trending balance
        now = datetime.now()
        balances = [10000, 10100, 9950, 10200, 10050, 10300]
        for i, bal in enumerate(balances):
            tracker.update_balance(bal, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # Should have Sortino ratio calculated
        assert isinstance(metrics.sortino_ratio, float)

    def test_var_95_calculated_from_returns(self):
        """Test VaR (95%) is calculated from return series."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Add balance history with some volatility
        now = datetime.now()
        balances = [10000, 10100, 9900, 10200, 9800, 10300]
        for i, bal in enumerate(balances):
            tracker.update_balance(bal, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # VaR should be negative (represents loss)
        assert metrics.var_95 <= 0.0

    def test_calmar_ratio_with_drawdown(self):
        """Test Calmar ratio calculation with drawdown."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Simulate growth with drawdown
        now = datetime.now()
        tracker.update_balance(12000, timestamp=now)  # Peak
        tracker.update_balance(10800, timestamp=now + timedelta(days=30))  # Drawdown
        tracker.update_balance(13000, timestamp=now + timedelta(days=60))  # Recovery

        metrics = tracker.get_metrics()
        # Should have Calmar ratio > 0 if annualized return > 0
        assert isinstance(metrics.calmar_ratio, float)


@pytest.mark.fast
class TestPerformanceTrackerEdgeCases:
    """Tests for edge cases identified in code review."""

    def test_zero_pnl_trade_handling(self):
        """Test that zero-PnL trades are handled consistently."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Record trades with different PnL values
        for pnl in [100, 0, -50, 0, 150]:
            trade = Mock()
            trade.pnl = pnl
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            trade.symbol = "BTCUSDT"
            trade.side = "long"
            tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        # Should have 5 total trades: 2 wins, 1 loss, 2 zero
        assert metrics.total_trades == 3  # Only winning + losing
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        # Zero PnL trades should be tracked separately
        assert tracker._zero_pnl_trades == 2

    def test_trade_with_missing_timestamps(self):
        """Test recording trade without entry/exit times."""
        tracker = PerformanceTracker(initial_balance=10000)

        trade = Mock()
        trade.pnl = 100.0
        trade.entry_time = None
        trade.exit_time = None
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        # Should not raise error
        tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 1
        assert metrics.avg_trade_duration_hours == 0.0

    def test_trade_with_none_pnl(self):
        """Test recording trade with None PnL value."""
        tracker = PerformanceTracker(initial_balance=10000)

        trade = Mock()
        trade.pnl = None
        trade.entry_time = datetime.now()
        trade.exit_time = datetime.now()
        trade.symbol = "BTCUSDT"
        trade.side = "long"

        # Should not raise error, should log warning
        tracker.record_trade(trade)

        metrics = tracker.get_metrics()
        assert metrics.total_trades == 0  # None PnL treated as zero
        assert metrics.total_pnl == 0.0

    def test_var_with_insufficient_data(self):
        """Test VaR returns 0 with insufficient samples."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Add only 10 balance updates (less than 30 required)
        now = datetime.now()
        for i in range(10):
            tracker.update_balance(10000 + i * 10, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # VaR should be 0 with insufficient data
        assert metrics.var_95 == 0.0

    def test_var_with_sufficient_data(self):
        """Test VaR is calculated with sufficient samples."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Add 40 balance updates (more than 30 required)
        now = datetime.now()
        for i in range(40):
            # Add some volatility
            balance = 10000 + i * 10 + (i % 3 - 1) * 50
            tracker.update_balance(balance, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # VaR should be calculated (negative value representing loss)
        assert isinstance(metrics.var_95, float)

    def test_sortino_with_no_downside_returns_capped(self):
        """Test Sortino ratio caps at 999 instead of infinity."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Simulate only upward growth (no downside)
        now = datetime.now()
        for i in range(30):
            tracker.update_balance(10000 + i * 100, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # Sortino should be capped at 999.0, not infinity
        assert metrics.sortino_ratio == 999.0

    def test_calmar_ratio_with_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown returns finite value."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Simulate growth with no drawdown
        now = datetime.now()
        for i in range(30):
            tracker.update_balance(10000 + i * 50, timestamp=now + timedelta(days=i))

        metrics = tracker.get_metrics()
        # Calmar should be capped at 999.0, not infinity
        assert metrics.calmar_ratio == 999.0

    def test_memory_limit_on_trade_history(self):
        """Test trade history memory limiting."""
        tracker = PerformanceTracker(initial_balance=10000)

        # Record more trades than the memory limit
        for i in range(tracker._max_trade_history + 100):
            trade = Mock()
            trade.pnl = float(i)
            trade.entry_time = datetime.now()
            trade.exit_time = datetime.now()
            trade.symbol = f"BTC{i}"
            trade.side = "long"
            tracker.record_trade(trade)

        # Trade history should be limited
        history = tracker.get_trade_history()
        assert len(history) == tracker._max_trade_history

        # Should contain most recent trades
        assert history[-1]["pnl"] == float(tracker._max_trade_history + 99)

    def test_expectancy_validation(self):
        """Test expectancy validates avg_loss is negative."""
        from src.performance.metrics import expectancy

        # Valid case
        result = expectancy(0.6, 100.0, -50.0)
        assert result == pytest.approx(40.0, abs=1.0)

        # Invalid case: positive avg_loss should raise
        with pytest.raises(ValueError, match="avg_loss must be negative"):
            expectancy(0.6, 100.0, 50.0)

        # Invalid case: negative avg_win should raise
        with pytest.raises(ValueError, match="avg_win must be non-negative"):
            expectancy(0.6, -100.0, -50.0)

    def test_concurrent_trade_recording(self):
        """Test thread-safety of concurrent trade recording."""
        import threading

        tracker = PerformanceTracker(initial_balance=10000)

        def record_trades():
            for i in range(50):
                trade = Mock()
                trade.pnl = float(i)
                trade.entry_time = datetime.now()
                trade.exit_time = datetime.now()
                trade.symbol = "BTCUSDT"
                trade.side = "long"
                tracker.record_trade(trade)

        # Run concurrent trade recording
        threads = [threading.Thread(target=record_trades) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded all trades without errors
        metrics = tracker.get_metrics()
        assert metrics.total_trades == 250  # 50 trades * 5 threads
