"""
Unit tests for PerformanceMonitor

Tests the sophisticated performance monitoring with degradation detection,
multi-timeframe analysis, and statistical significance testing.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd

from src.strategies.components.performance_monitor import (
    DegradationSeverity,
    PerformanceDegradationConfig,
    PerformanceMonitor,
    TimeFrame,
)
from src.strategies.components.performance_tracker import (
    PerformanceMetrics,
    PerformancePeriod,
    PerformanceTracker,
    TradeResult,
)
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = PerformanceDegradationConfig(
            min_trades_for_evaluation=20,
            min_days_for_evaluation=10,
            short_term_days=7,
            medium_term_days=30,
            long_term_days=90,
        )
        self.monitor = PerformanceMonitor(self.config)

        # Create mock performance tracker with sufficient data
        self.performance_tracker = PerformanceTracker("test_strategy")
        self._populate_performance_tracker()

        # Create mock regime context
        self.regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=20,
            strength=0.7,
        )

    def _populate_performance_tracker(self):
        """Populate performance tracker with test data"""
        base_time = datetime.now() - timedelta(days=100)

        # Add 50 trades over 100 days with declining performance
        for i in range(50):
            timestamp = base_time + timedelta(days=i * 2)

            # Simulate declining performance over time
            performance_factor = 1.0 - (i / 100)  # Gradual decline

            # Create winning and losing trades
            if i % 3 == 0:  # Losing trade
                pnl = -100 * performance_factor
                pnl_percent = -0.02 * performance_factor
            else:  # Winning trade
                pnl = 150 * performance_factor
                pnl_percent = 0.03 * performance_factor

            trade = TradeResult(
                timestamp=timestamp,
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                exit_price=50000.0 + pnl,
                quantity=0.001,
                pnl=pnl,
                pnl_percent=pnl_percent,
                duration_hours=24.0,
                strategy_id="test_strategy",
                confidence=0.7,
                regime="bull_low_vol",
            )

            self.performance_tracker.record_trade(trade)

    def test_initialization(self):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor()
        self.assertIsNotNone(monitor.config)
        self.assertEqual(len(monitor.performance_baselines), 0)
        self.assertEqual(len(monitor.regime_performance_history), 0)
        self.assertEqual(len(monitor.degradation_history), 0)

    def test_meets_minimum_requirements_sufficient_data(self):
        """Test minimum requirements check with sufficient data"""
        result = self.monitor._meets_minimum_requirements(self.performance_tracker)
        self.assertTrue(result)

    def test_meets_minimum_requirements_insufficient_trades(self):
        """Test minimum requirements check with insufficient trades"""
        empty_tracker = PerformanceTracker("empty_strategy")
        result = self.monitor._meets_minimum_requirements(empty_tracker)
        self.assertFalse(result)

    def test_meets_minimum_requirements_insufficient_time(self):
        """Test minimum requirements check with insufficient time period"""
        recent_tracker = PerformanceTracker("recent_strategy")

        # Add trades only from yesterday
        yesterday = datetime.now() - timedelta(days=1)
        for i in range(25):  # Enough trades but too recent
            trade = TradeResult(
                timestamp=yesterday + timedelta(hours=i),
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                exit_price=50100.0,
                quantity=0.001,
                pnl=100.0,
                pnl_percent=0.002,
                duration_hours=1.0,
                strategy_id="recent_strategy",
                confidence=0.7,
            )
            recent_tracker.record_trade(trade)

        result = self.monitor._meets_minimum_requirements(recent_tracker)
        self.assertFalse(result)

    def test_is_underperforming_true(self):
        """Test underperformance detection when strategy is underperforming"""
        # Create current metrics with poor performance
        current = PerformanceMetrics(
            total_return=100.0,
            total_return_pct=0.05,
            annualized_return=0.18,
            volatility=0.15,
            sharpe_ratio=0.3,
            sortino_ratio=0.4,
            calmar_ratio=1.2,
            max_drawdown=0.25,
            var_95=-0.03,
            total_trades=30,
            winning_trades=12,
            losing_trades=18,
            win_rate=0.4,
            avg_win=0.025,
            avg_loss=-0.015,
            profit_factor=1.2,
            expectancy=0.002,
            avg_trade_duration=24.0,
            trades_per_day=1.0,
            hit_rate=0.4,
            max_drawdown_duration=5.0,
            current_drawdown=0.1,
            drawdown_recovery_time=2.0,
            best_trade=0.05,
            worst_trade=-0.04,
            consecutive_wins=3,
            consecutive_losses=4,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            period_type=PerformancePeriod.MONTHLY,
        )

        # Create baseline metrics with good performance
        baseline = PerformanceMetrics(
            total_return=500.0,
            total_return_pct=0.25,
            annualized_return=0.91,
            volatility=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=4.5,
            max_drawdown=0.1,
            var_95=-0.02,
            total_trades=50,
            winning_trades=35,
            losing_trades=15,
            win_rate=0.7,
            avg_win=0.03,
            avg_loss=-0.01,
            profit_factor=2.1,
            expectancy=0.005,
            avg_trade_duration=20.0,
            trades_per_day=1.2,
            hit_rate=0.7,
            max_drawdown_duration=3.0,
            current_drawdown=0.05,
            drawdown_recovery_time=1.0,
            best_trade=0.06,
            worst_trade=-0.02,
            consecutive_wins=8,
            consecutive_losses=2,
            period_start=datetime.now() - timedelta(days=60),
            period_end=datetime.now() - timedelta(days=30),
            period_type=PerformancePeriod.MONTHLY,
        )

        result = self.monitor._is_underperforming(current, baseline)
        self.assertTrue(result)

    def test_is_underperforming_false(self):
        """Test underperformance detection when strategy is performing well"""
        # Create current metrics with good performance
        current = PerformanceMetrics(
            total_return=600.0,
            total_return_pct=0.30,
            annualized_return=1.1,
            volatility=0.10,
            sharpe_ratio=2.0,
            sortino_ratio=2.5,
            calmar_ratio=5.5,
            max_drawdown=0.08,
            var_95=-0.015,
            total_trades=40,
            winning_trades=32,
            losing_trades=8,
            win_rate=0.8,
            avg_win=0.035,
            avg_loss=-0.008,
            profit_factor=2.8,
            expectancy=0.0075,
            avg_trade_duration=18.0,
            trades_per_day=1.3,
            hit_rate=0.8,
            max_drawdown_duration=2.0,
            current_drawdown=0.02,
            drawdown_recovery_time=0.5,
            best_trade=0.07,
            worst_trade=-0.015,
            consecutive_wins=12,
            consecutive_losses=1,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            period_type=PerformancePeriod.MONTHLY,
        )

        # Create baseline metrics with moderate performance
        baseline = PerformanceMetrics(
            total_return=500.0,
            total_return_pct=0.25,
            annualized_return=0.91,
            volatility=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=4.5,
            max_drawdown=0.1,
            var_95=-0.02,
            total_trades=50,
            winning_trades=35,
            losing_trades=15,
            win_rate=0.7,
            avg_win=0.03,
            avg_loss=-0.01,
            profit_factor=2.1,
            expectancy=0.005,
            avg_trade_duration=20.0,
            trades_per_day=1.2,
            hit_rate=0.7,
            max_drawdown_duration=3.0,
            current_drawdown=0.05,
            drawdown_recovery_time=1.0,
            best_trade=0.06,
            worst_trade=-0.02,
            consecutive_wins=8,
            consecutive_losses=2,
            period_start=datetime.now() - timedelta(days=60),
            period_end=datetime.now() - timedelta(days=30),
            period_type=PerformancePeriod.MONTHLY,
        )

        result = self.monitor._is_underperforming(current, baseline)
        self.assertFalse(result)

    def test_calculate_statistical_significance(self):
        """Test statistical significance calculation"""
        current_metrics = self.performance_tracker.get_performance_metrics()
        baseline_metrics = PerformanceMetrics(
            total_return=1000.0,
            total_return_pct=0.5,
            annualized_return=1.8,
            volatility=0.08,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            calmar_ratio=7.5,
            max_drawdown=0.06,
            var_95=-0.01,
            total_trades=50,
            winning_trades=40,
            losing_trades=10,
            win_rate=0.8,
            avg_win=0.04,
            avg_loss=-0.005,
            profit_factor=3.2,
            expectancy=0.01,
            avg_trade_duration=16.0,
            trades_per_day=1.5,
            hit_rate=0.8,
            max_drawdown_duration=1.5,
            current_drawdown=0.01,
            drawdown_recovery_time=0.3,
            best_trade=0.08,
            worst_trade=-0.01,
            consecutive_wins=15,
            consecutive_losses=1,
            period_start=datetime.now() - timedelta(days=90),
            period_end=datetime.now() - timedelta(days=60),
            period_type=PerformancePeriod.QUARTERLY,
        )

        significance = self.monitor._calculate_statistical_significance(
            current_metrics, baseline_metrics, self.performance_tracker, 30
        )

        self.assertIsInstance(significance, float)
        self.assertGreaterEqual(significance, 0.0)
        self.assertLessEqual(significance, 1.0)

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation"""
        current_metrics = self.performance_tracker.get_performance_metrics()

        interval = self.monitor._calculate_confidence_interval(
            current_metrics, self.performance_tracker, 30
        )

        self.assertIsInstance(interval, tuple)
        self.assertEqual(len(interval), 2)
        self.assertLessEqual(interval[0], interval[1])  # Lower bound <= upper bound

    def test_determine_timeframe_severity(self):
        """Test degradation severity determination"""
        # Create severely degraded current metrics
        current = PerformanceMetrics(
            total_return=50.0,
            total_return_pct=0.025,
            annualized_return=0.09,
            volatility=0.20,
            sharpe_ratio=0.1,
            sortino_ratio=0.15,
            calmar_ratio=0.45,
            max_drawdown=0.35,
            var_95=-0.05,
            total_trades=25,
            winning_trades=8,
            losing_trades=17,
            win_rate=0.32,
            avg_win=0.02,
            avg_loss=-0.018,
            profit_factor=0.9,
            expectancy=-0.001,
            avg_trade_duration=30.0,
            trades_per_day=0.8,
            hit_rate=0.32,
            max_drawdown_duration=8.0,
            current_drawdown=0.2,
            drawdown_recovery_time=5.0,
            best_trade=0.04,
            worst_trade=-0.06,
            consecutive_wins=2,
            consecutive_losses=6,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            period_type=PerformancePeriod.MONTHLY,
        )

        # Create good baseline metrics
        baseline = PerformanceMetrics(
            total_return=500.0,
            total_return_pct=0.25,
            annualized_return=0.91,
            volatility=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=4.5,
            max_drawdown=0.1,
            var_95=-0.02,
            total_trades=50,
            winning_trades=35,
            losing_trades=15,
            win_rate=0.7,
            avg_win=0.03,
            avg_loss=-0.01,
            profit_factor=2.1,
            expectancy=0.005,
            avg_trade_duration=20.0,
            trades_per_day=1.2,
            hit_rate=0.7,
            max_drawdown_duration=3.0,
            current_drawdown=0.05,
            drawdown_recovery_time=1.0,
            best_trade=0.06,
            worst_trade=-0.02,
            consecutive_wins=8,
            consecutive_losses=2,
            period_start=datetime.now() - timedelta(days=60),
            period_end=datetime.now() - timedelta(days=30),
            period_type=PerformancePeriod.MONTHLY,
        )

        severity = self.monitor._determine_timeframe_severity(current, baseline)

        # Should detect severe or critical degradation
        self.assertIn(severity, [DegradationSeverity.SEVERE, DegradationSeverity.CRITICAL])

    def test_update_performance_baseline(self):
        """Test performance baseline update"""
        strategy_id = "test_strategy"

        # Update baseline
        self.monitor.update_performance_baseline(strategy_id, self.performance_tracker)

        # Check that baselines were created
        self.assertIn(strategy_id, self.monitor.performance_baselines)

        # Check that all timeframes have baselines
        for timeframe in TimeFrame:
            if timeframe in self.monitor.performance_baselines[strategy_id]:
                baseline = self.monitor.performance_baselines[strategy_id][timeframe]
                self.assertIsInstance(baseline, PerformanceMetrics)

    def test_update_performance_baseline_with_regime(self):
        """Test performance baseline update with regime context"""
        strategy_id = "test_strategy"
        regime = "bull_low_vol"

        # Update baseline with regime
        self.monitor.update_performance_baseline(strategy_id, self.performance_tracker, regime)

        # Check that regime-specific history was updated
        self.assertIn(strategy_id, self.monitor.regime_performance_history)
        self.assertIn(regime, self.monitor.regime_performance_history[strategy_id])
        self.assertGreater(len(self.monitor.regime_performance_history[strategy_id][regime]), 0)

    def test_get_regime_performance_confidence_unknown_strategy(self):
        """Test regime performance confidence for unknown strategy"""
        confidence = self.monitor.get_regime_performance_confidence(
            "unknown_strategy", "bull_low_vol", self.performance_tracker
        )

        self.assertEqual(confidence, 0.5)  # Neutral confidence

    def test_get_regime_performance_confidence_unknown_regime(self):
        """Test regime performance confidence for unknown regime"""
        strategy_id = "test_strategy"

        confidence = self.monitor.get_regime_performance_confidence(
            strategy_id, "unknown_regime", self.performance_tracker
        )

        self.assertEqual(confidence, 0.3)  # Low confidence

    def test_get_regime_performance_confidence_with_history(self):
        """Test regime performance confidence with historical data"""
        strategy_id = "test_strategy"
        regime = "bull_low_vol"

        # Add some regime performance history
        self.monitor.update_performance_baseline(strategy_id, self.performance_tracker, regime)

        confidence = self.monitor.get_regime_performance_confidence(
            strategy_id, regime, self.performance_tracker
        )

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_should_switch_strategy_insufficient_data(self):
        """Test strategy switch decision with insufficient data"""
        empty_tracker = PerformanceTracker("empty_strategy")
        market_data = pd.DataFrame()

        decision = self.monitor.should_switch_strategy("empty_strategy", empty_tracker, market_data)

        self.assertFalse(decision.should_switch)
        self.assertEqual(decision.reason, "Insufficient data for evaluation")
        self.assertEqual(decision.confidence, 0.0)

    def test_should_switch_strategy_regime_transition(self):
        """Test strategy switch decision during regime transition"""
        market_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        # Create unstable regime (low confidence, short duration)
        unstable_regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.HIGH,
            confidence=0.5,  # Low confidence
            duration=5,  # Short duration
            strength=0.6,
        )

        # Update baseline first
        self.monitor.update_performance_baseline("test_strategy", self.performance_tracker)

        decision = self.monitor.should_switch_strategy(
            "test_strategy", self.performance_tracker, market_data, unstable_regime
        )

        self.assertFalse(decision.should_switch)
        self.assertEqual(decision.reason, "In regime transition period")

    @patch(
        "src.strategies.components.performance_monitor.PerformanceMonitor._analyze_multiple_timeframes"
    )
    def test_should_switch_strategy_not_underperforming_all_timeframes(self, mock_analyze):
        """Test strategy switch decision when not underperforming across all timeframes"""
        # Mock timeframe analysis with mixed results
        mock_analyze.return_value = [
            Mock(underperforming=True, statistical_significance=0.8),
            Mock(underperforming=False, statistical_significance=0.3),  # Not underperforming
            Mock(underperforming=True, statistical_significance=0.9),
        ]

        market_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        decision = self.monitor.should_switch_strategy(
            "test_strategy", self.performance_tracker, market_data, self.regime
        )

        self.assertFalse(decision.should_switch)
        self.assertEqual(decision.reason, "Not underperforming across all timeframes")

    def test_get_degradation_history(self):
        """Test degradation history retrieval"""
        # Add some degradation history
        self.monitor.degradation_history.extend(
            [
                (datetime.now() - timedelta(days=5), "strategy1", DegradationSeverity.MODERATE),
                (datetime.now() - timedelta(days=10), "strategy2", DegradationSeverity.SEVERE),
                (
                    datetime.now() - timedelta(days=40),
                    "strategy1",
                    DegradationSeverity.MINOR,
                ),  # Too old
            ]
        )

        # Get recent history (30 days)
        history = self.monitor.get_degradation_history(days=30)

        self.assertEqual(len(history), 2)  # Should exclude the 40-day old entry

        # Get history for specific strategy
        strategy_history = self.monitor.get_degradation_history("strategy1", days=30)

        self.assertEqual(len(strategy_history), 1)
        self.assertEqual(strategy_history[0]["strategy_id"], "strategy1")

    def test_reset_baselines_specific_strategy(self):
        """Test resetting baselines for specific strategy"""
        # Add some baselines
        self.monitor.update_performance_baseline("strategy1", self.performance_tracker)
        self.monitor.update_performance_baseline("strategy2", self.performance_tracker)

        # Reset specific strategy
        self.monitor.reset_baselines("strategy1")

        # Check that only strategy1 was reset
        self.assertNotIn("strategy1", self.monitor.performance_baselines)
        self.assertIn("strategy2", self.monitor.performance_baselines)

    def test_reset_baselines_all_strategies(self):
        """Test resetting all baselines"""
        # Add some baselines
        self.monitor.update_performance_baseline("strategy1", self.performance_tracker)
        self.monitor.update_performance_baseline("strategy2", self.performance_tracker)

        # Reset all
        self.monitor.reset_baselines()

        # Check that all were reset
        self.assertEqual(len(self.monitor.performance_baselines), 0)
        self.assertEqual(len(self.monitor.regime_performance_history), 0)

    def test_get_timeframe_days(self):
        """Test timeframe days calculation"""
        self.assertEqual(self.monitor._get_timeframe_days(TimeFrame.SHORT_TERM), 7)
        self.assertEqual(self.monitor._get_timeframe_days(TimeFrame.MEDIUM_TERM), 30)
        self.assertEqual(self.monitor._get_timeframe_days(TimeFrame.LONG_TERM), 90)


if __name__ == "__main__":
    unittest.main()
