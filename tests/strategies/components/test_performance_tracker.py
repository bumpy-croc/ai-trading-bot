"""
Tests for Performance Tracking System

This module tests the PerformanceTracker implementation including real-time metrics,
historical data storage, comparison utilities, and performance calculations.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from src.strategies.components.performance_tracker import (
    PerformanceMetrics,
    PerformancePeriod,
    PerformanceTracker,
    RegimePerformance,
    TradeResult,
)


class TestTradeResult:
    """Test TradeResult data class"""
    
    def test_trade_result_creation(self):
        """Test TradeResult creation"""
        timestamp = datetime.now()
        trade = TradeResult(
            timestamp=timestamp,
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            pnl=100.0,
            pnl_percent=2.0,
            duration_hours=24.0,
            strategy_id="test_strategy",
            confidence=0.8,
            regime="bull_low_vol",
            exit_reason="take_profit"
        )
        
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "long"
        assert trade.pnl == 100.0
        assert trade.pnl_percent == 2.0
        assert trade.confidence == 0.8
        assert trade.regime == "bull_low_vol"
    
    def test_trade_result_serialization(self):
        """Test TradeResult serialization and deserialization"""
        timestamp = datetime.now()
        trade = TradeResult(
            timestamp=timestamp,
            symbol="ETHUSDT",
            side="short",
            entry_price=3000.0,
            exit_price=2950.0,
            quantity=1.0,
            pnl=50.0,
            pnl_percent=1.67,
            duration_hours=12.0,
            strategy_id="test_strategy",
            confidence=0.7
        )
        
        # Test serialization
        trade_dict = trade.to_dict()
        assert trade_dict['symbol'] == "ETHUSDT"
        assert trade_dict['timestamp'] == timestamp.isoformat()
        assert trade_dict['pnl'] == 50.0
        
        # Test deserialization
        restored_trade = TradeResult.from_dict(trade_dict)
        assert restored_trade.symbol == trade.symbol
        assert restored_trade.timestamp == trade.timestamp
        assert restored_trade.pnl == trade.pnl
        assert restored_trade.confidence == trade.confidence


class TestPerformanceMetrics:
    """Test PerformanceMetrics data class"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation"""
        start_time = datetime.now()
        end_time = start_time + timedelta(days=30)
        
        metrics = PerformanceMetrics(
            total_return=1000.0,
            total_return_pct=10.0,
            annualized_return=120.0,
            volatility=15.0,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=2.0,
            max_drawdown=5.0,
            var_95=-2.5,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            avg_win=2.5,
            avg_loss=-1.5,
            profit_factor=1.67,
            expectancy=0.9,
            avg_trade_duration=24.0,
            trades_per_day=1.67,
            hit_rate=0.6,
            max_drawdown_duration=72.0,
            current_drawdown=2.0,
            drawdown_recovery_time=48.0,
            best_trade=8.5,
            worst_trade=-4.2,
            consecutive_wins=5,
            consecutive_losses=3,
            period_start=start_time,
            period_end=end_time,
            period_type=PerformancePeriod.MONTHLY
        )
        
        assert metrics.total_return == 1000.0
        assert metrics.win_rate == 0.6
        assert metrics.sharpe_ratio == 1.5
        assert metrics.period_type == PerformancePeriod.MONTHLY
    
    def test_performance_metrics_serialization(self):
        """Test PerformanceMetrics serialization"""
        start_time = datetime.now()
        end_time = start_time + timedelta(days=7)
        
        metrics = PerformanceMetrics(
            total_return=500.0, total_return_pct=5.0, annualized_return=260.0,
            volatility=12.0, sharpe_ratio=1.2, sortino_ratio=1.4, calmar_ratio=1.8,
            max_drawdown=3.0, var_95=-1.8, total_trades=25, winning_trades=15,
            losing_trades=10, win_rate=0.6, avg_win=2.0, avg_loss=-1.2,
            profit_factor=1.5, expectancy=0.6, avg_trade_duration=18.0,
            trades_per_day=3.6, hit_rate=0.6, max_drawdown_duration=24.0,
            current_drawdown=1.0, drawdown_recovery_time=12.0, best_trade=6.0,
            worst_trade=-3.5, consecutive_wins=4, consecutive_losses=2,
            period_start=start_time, period_end=end_time, period_type=PerformancePeriod.WEEKLY
        )
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict['total_return'] == 500.0
        assert metrics_dict['period_type'] == 'weekly'
        assert metrics_dict['period_start'] == start_time.isoformat()
        
        # Test deserialization
        restored_metrics = PerformanceMetrics.from_dict(metrics_dict)
        assert restored_metrics.total_return == metrics.total_return
        assert restored_metrics.period_type == metrics.period_type
        assert restored_metrics.period_start == metrics.period_start


class TestPerformanceTracker:
    """Test PerformanceTracker functionality"""
    
    @pytest.fixture
    def tracker(self):
        """Create a test performance tracker"""
        return PerformanceTracker("test_strategy_123")
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing"""
        base_time = datetime.now() - timedelta(days=10)
        trades = []
        
        # Create a mix of winning and losing trades
        trade_data = [
            (100.0, 2.0, "bull_low_vol"),
            (-50.0, -1.0, "bull_low_vol"),
            (150.0, 3.0, "bull_high_vol"),
            (-75.0, -1.5, "bear_low_vol"),
            (200.0, 4.0, "bull_low_vol"),
            (25.0, 0.5, "range_low_vol"),  # Changed to positive
            (80.0, 1.6, "bull_high_vol"),
            (-100.0, -2.0, "bear_high_vol"),
            (120.0, 2.4, "bull_low_vol"),
            (-60.0, -1.2, "bear_low_vol")
        ]
        
        for i, (pnl, pnl_pct, regime) in enumerate(trade_data):
            trades.append(TradeResult(
                timestamp=base_time + timedelta(hours=i * 6),
                symbol="BTCUSDT",
                side="long" if pnl > 0 else "short",
                entry_price=50000.0,
                exit_price=50000.0 + (pnl * 10),
                quantity=0.1,
                pnl=pnl,
                pnl_percent=pnl_pct,
                duration_hours=6.0,
                strategy_id="test_strategy_123",
                confidence=0.7,
                regime=regime
            ))
        
        return trades
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.strategy_id == "test_strategy_123"
        assert len(tracker.trades) == 0
        assert tracker.trade_count == 0
        assert tracker.current_balance == 0.0
        assert tracker.initial_balance == 0.0
        assert tracker.max_drawdown == 0.0
    
    def test_record_single_trade(self, tracker):
        """Test recording a single trade"""
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            pnl=100.0,
            pnl_percent=2.0,
            duration_hours=24.0,
            strategy_id="test_strategy_123",
            confidence=0.8,
            regime="bull_low_vol"
        )
        
        tracker.record_trade(trade)
        
        assert len(tracker.trades) == 1
        assert tracker.trade_count == 1
        assert tracker.current_balance == 100.0
        assert tracker.initial_balance == 0.0  # First trade sets baseline
        assert tracker.peak_balance == 100.0
        assert tracker.running_stats['winning_trades'] == 1
        assert tracker.running_stats['best_trade'] == 2.0
    
    def test_record_multiple_trades(self, tracker, sample_trades):
        """Test recording multiple trades"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        assert len(tracker.trades) == 10
        assert tracker.trade_count == 10
        
        # Check running statistics
        expected_total_pnl = sum(t.pnl for t in sample_trades)
        assert tracker.current_balance == expected_total_pnl
        
        winning_trades = [t for t in sample_trades if t.pnl > 0]
        losing_trades = [t for t in sample_trades if t.pnl <= 0]
        
        assert tracker.running_stats['winning_trades'] == len(winning_trades)
        assert tracker.running_stats['losing_trades'] == len(losing_trades)
    
    def test_balance_and_drawdown_tracking(self, tracker):
        """Test balance and drawdown tracking"""
        trades = [
            TradeResult(datetime.now(), "BTCUSDT", "long", 50000, 51000, 0.1, 
                       100.0, 2.0, 24.0, "test", 0.8),
            TradeResult(datetime.now(), "BTCUSDT", "long", 51000, 52000, 0.1, 
                       100.0, 2.0, 24.0, "test", 0.8),
            TradeResult(datetime.now(), "BTCUSDT", "short", 52000, 51500, 0.1, 
                       -50.0, -1.0, 12.0, "test", 0.7),
            TradeResult(datetime.now(), "BTCUSDT", "short", 51500, 50000, 0.1, 
                       -150.0, -3.0, 18.0, "test", 0.6)
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
        
        # Balance should be 100 + 100 - 50 - 150 = 0
        assert tracker.current_balance == 0.0
        
        # Peak should be 200 (after first two trades)
        assert tracker.peak_balance == 200.0
        
        # Current drawdown should be 100% (from 200 to 0)
        assert tracker.current_drawdown == 1.0
        assert tracker.max_drawdown == 1.0
    
    def test_get_performance_metrics_all_time(self, tracker, sample_trades):
        """Test getting all-time performance metrics"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        metrics = tracker.get_performance_metrics(PerformancePeriod.ALL_TIME)
        
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 6  # Positive PnL trades
        assert metrics.losing_trades == 4   # Negative PnL trades
        assert metrics.win_rate == 0.6
        
        # Check that metrics are calculated
        assert metrics.total_return == sum(t.pnl for t in sample_trades)
        assert metrics.volatility > 0
        assert metrics.period_type == PerformancePeriod.ALL_TIME
    
    def test_get_performance_metrics_empty(self, tracker):
        """Test getting metrics with no trades"""
        metrics = tracker.get_performance_metrics(PerformancePeriod.ALL_TIME)
        
        assert metrics.total_trades == 0
        assert metrics.total_return == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    def test_regime_performance_tracking(self, tracker, sample_trades):
        """Test regime-specific performance tracking"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        regime_performance = tracker.get_regime_performance()
        
        # Should have multiple regimes
        assert len(regime_performance) > 1
        assert "bull_low_vol" in regime_performance
        assert "bear_low_vol" in regime_performance
        
        # Check bull_low_vol performance
        bull_perf = regime_performance["bull_low_vol"]
        assert isinstance(bull_perf, RegimePerformance)
        assert bull_perf.regime_type == "bull_low_vol"
        assert bull_perf.trade_count > 0
        assert 0 <= bull_perf.win_rate <= 1
    
    def test_get_specific_regime_performance(self, tracker, sample_trades):
        """Test getting performance for specific regime"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        bull_performance = tracker.get_regime_performance("bull_low_vol")
        
        assert "bull_low_vol" in bull_performance
        assert len(bull_performance) == 1
        
        # Test non-existent regime
        empty_performance = tracker.get_regime_performance("non_existent")
        assert len(empty_performance) == 0
    
    def test_compare_performance(self, sample_trades):
        """Test performance comparison between trackers"""
        tracker1 = PerformanceTracker("strategy_1")
        tracker2 = PerformanceTracker("strategy_2")
        
        # Add trades to first tracker
        for trade in sample_trades[:5]:
            tracker1.record_trade(trade)
        
        # Add different trades to second tracker (modify PnL)
        for i, trade in enumerate(sample_trades[5:]):
            modified_trade = TradeResult(
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.pnl * 1.2,  # 20% better performance
                pnl_percent=trade.pnl_percent * 1.2,
                duration_hours=trade.duration_hours,
                strategy_id="strategy_2",
                confidence=trade.confidence,
                regime=trade.regime
            )
            tracker2.record_trade(modified_trade)
        
        comparison = tracker1.compare_performance(tracker2)
        
        assert comparison['strategy_1']['id'] == "strategy_1"
        assert comparison['strategy_2']['id'] == "strategy_2"
        assert 'comparison' in comparison
        assert 'winner' in comparison
        
        # Check comparison metrics
        comp = comparison['comparison']
        assert 'return_difference' in comp
        assert 'sharpe_difference' in comp
        assert 'win_rate_difference' in comp
    
    def test_get_performance_summary(self, tracker, sample_trades):
        """Test getting comprehensive performance summary"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        summary = tracker.get_performance_summary()
        
        assert summary['strategy_id'] == "test_strategy_123"
        assert 'current_metrics' in summary
        assert 'daily_metrics' in summary
        assert 'regime_performance' in summary
        assert 'running_stats' in summary
        assert 'balance_info' in summary
        assert 'trade_count' in summary
        
        # Check balance info
        balance_info = summary['balance_info']
        assert 'current_balance' in balance_info
        assert 'peak_balance' in balance_info
        assert 'max_drawdown' in balance_info
    
    def test_get_trade_history(self, tracker, sample_trades):
        """Test getting trade history with filters"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        # Test getting all trades
        all_trades = tracker.get_trade_history()
        assert len(all_trades) == 10
        
        # Test with limit
        limited_trades = tracker.get_trade_history(limit=5)
        assert len(limited_trades) == 5
        
        # Test with date filter
        mid_date = sample_trades[5].timestamp
        recent_trades = tracker.get_trade_history(start_date=mid_date)
        assert len(recent_trades) <= 5  # Should be 5 or fewer
        
        # Verify trades are sorted by timestamp (most recent first)
        timestamps = [t.timestamp for t in all_trades]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_reset_performance(self, tracker, sample_trades):
        """Test resetting performance data"""
        # Add some trades
        for trade in sample_trades[:3]:
            tracker.record_trade(trade)
        
        assert len(tracker.trades) == 3
        assert tracker.trade_count == 3
        assert tracker.current_balance != 0.0
        
        # Reset
        tracker.reset_performance()
        
        assert len(tracker.trades) == 0
        assert tracker.trade_count == 0
        assert tracker.current_balance == 0.0
        assert tracker.initial_balance == 0.0
        assert tracker.peak_balance == 0.0
        assert tracker.running_stats['winning_trades'] == 0
        assert len(tracker.regime_performance) == 0
    
    def test_metrics_caching(self, tracker, sample_trades):
        """Test performance metrics caching"""
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        # First call should calculate and cache
        metrics1 = tracker.get_performance_metrics(PerformancePeriod.ALL_TIME)
        
        # Second call should use cache
        metrics2 = tracker.get_performance_metrics(PerformancePeriod.ALL_TIME)
        
        # Should be the same object (from cache)
        assert metrics1.total_return == metrics2.total_return
        assert metrics1.sharpe_ratio == metrics2.sharpe_ratio
        
        # Adding a new trade should clear cache
        new_trade = TradeResult(
            datetime.now(), "BTCUSDT", "long", 50000, 51000, 0.1,
            100.0, 2.0, 24.0, "test", 0.8
        )
        tracker.record_trade(new_trade)
        
        # Next call should recalculate
        metrics3 = tracker.get_performance_metrics(PerformancePeriod.ALL_TIME)
        assert metrics3.total_trades == metrics1.total_trades + 1
    
    def test_sharpe_ratio_calculation(self, tracker):
        """Test Sharpe ratio calculation"""
        # Create trades with known returns
        returns = [0.02, -0.01, 0.03, -0.015, 0.025, 0.01, -0.005, 0.02]
        
        for i, ret in enumerate(returns):
            trade = TradeResult(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                exit_price=50000.0 * (1 + ret),
                quantity=0.1,
                pnl=ret * 5000,  # Assuming $5000 position
                pnl_percent=ret * 100,
                duration_hours=24.0,
                strategy_id="test",
                confidence=0.8
            )
            tracker.record_trade(trade)
        
        metrics = tracker.get_performance_metrics()
        
        # Sharpe ratio should be calculated
        assert metrics.sharpe_ratio != 0.0
        # With positive average return and some volatility, should be positive
        assert metrics.sharpe_ratio > 0.0
    
    def test_max_drawdown_calculation(self, tracker):
        """Test maximum drawdown calculation"""
        # Create a sequence that goes up then down
        pnl_sequence = [100, 150, 200, 50, -100, -50, 0, 100]
        
        for i, pnl in enumerate(pnl_sequence):
            trade = TradeResult(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                side="long" if pnl > 0 else "short",
                entry_price=50000.0,
                exit_price=50000.0 + pnl,
                quantity=0.1,
                pnl=pnl,
                pnl_percent=(pnl / 5000) * 100,
                duration_hours=24.0,
                strategy_id="test",
                confidence=0.8
            )
            tracker.record_trade(trade)
        
        # Peak should be at 450 (100+150+200), then drops to 0 (450-450)
        # So max drawdown should be 100%
        assert tracker.max_drawdown > 0.0
        assert tracker.peak_balance > 0.0
    
    def test_consecutive_streaks(self, tracker):
        """Test consecutive win/loss streak tracking"""
        # Create alternating wins and losses with some streaks
        pnl_sequence = [100, 150, 200, -50, -75, 80, 120, 90, -30, -60]
        
        for i, pnl in enumerate(pnl_sequence):
            trade = TradeResult(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                side="long" if pnl > 0 else "short",
                entry_price=50000.0,
                exit_price=50000.0 + pnl,
                quantity=0.1,
                pnl=pnl,
                pnl_percent=(pnl / 5000) * 100,
                duration_hours=24.0,
                strategy_id="test",
                confidence=0.8
            )
            tracker.record_trade(trade)
        
        # Should have tracked streaks
        assert tracker.running_stats['max_win_streak'] >= 3  # First 3 wins
        assert tracker.running_stats['max_loss_streak'] >= 2  # 2 losses in middle
    
    def test_period_filtering(self, tracker):
        """Test filtering trades by different periods"""
        now = datetime.now()
        
        # Create trades across different time periods
        trades = [
            # Today
            TradeResult(now, "BTCUSDT", "long", 50000, 51000, 0.1, 100, 2.0, 24, "test", 0.8),
            # Yesterday
            TradeResult(now - timedelta(days=1), "BTCUSDT", "long", 50000, 51000, 0.1, 150, 3.0, 24, "test", 0.8),
            # Last week
            TradeResult(now - timedelta(days=7), "BTCUSDT", "long", 50000, 51000, 0.1, 200, 4.0, 24, "test", 0.8),
            # Last month
            TradeResult(now - timedelta(days=30), "BTCUSDT", "long", 50000, 51000, 0.1, 80, 1.6, 24, "test", 0.8),
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
        
        # Test daily metrics (should include today's trades)
        daily_metrics = tracker.get_performance_metrics(PerformancePeriod.DAILY)
        assert daily_metrics.total_trades >= 1
        
        # Test weekly metrics (should include this week's trades)
        weekly_metrics = tracker.get_performance_metrics(PerformancePeriod.WEEKLY)
        assert weekly_metrics.total_trades >= daily_metrics.total_trades
        
        # Test all-time metrics (should include all trades)
        all_time_metrics = tracker.get_performance_metrics(PerformancePeriod.ALL_TIME)
        assert all_time_metrics.total_trades == 4
    
    def test_storage_backend_integration(self, sample_trades):
        """Test integration with storage backend"""
        mock_storage = Mock()
        tracker = PerformanceTracker("test_strategy", storage_backend=mock_storage)
        
        # Record a trade
        trade = sample_trades[0]
        tracker.record_trade(trade)
        
        # Verify storage backend was called
        mock_storage.save_trade.assert_called_once_with(trade)
        
        # Test storage failure handling
        mock_storage.save_trade.side_effect = Exception("Storage error")
        
        # Should not raise exception, just log error
        tracker.record_trade(sample_trades[1])
        assert len(tracker.trades) == 2  # Trade should still be recorded locally


if __name__ == "__main__":
    pytest.main([__file__])