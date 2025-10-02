"""
Unit tests for StrategySelector

Tests the multi-criteria strategy selection algorithm with regime-specific
performance weighting, risk-adjusted selection, and correlation analysis.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from src.strategies.components.performance_tracker import (
    PerformanceMetrics,
    PerformancePeriod,
    PerformanceTracker,
    TradeResult,
)
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.strategy_selector import (
    SelectionConfig,
    SelectionCriteria,
    StrategyScore,
    StrategySelector,
)


class TestStrategySelector(unittest.TestCase):
    """Test cases for StrategySelector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SelectionConfig(
            min_trades_for_consideration=10,
            min_days_active=5
        )
        self.selector = StrategySelector(self.config)
        
        # Create mock strategies with different performance characteristics
        self.strategies = self._create_mock_strategies()
        
        # Create mock regime context
        self.regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=20,
            strength=0.7
        )
    
    def _create_mock_strategies(self) -> dict:
        """Create mock strategies with different performance profiles"""
        strategies = {}
        
        # Strategy 1: High Sharpe, moderate returns
        strategy1 = PerformanceTracker("high_sharpe_strategy")
        self._populate_tracker(strategy1, sharpe_target=2.0, return_target=0.15, drawdown_target=0.08)
        strategies["high_sharpe_strategy"] = strategy1
        
        # Strategy 2: High returns, higher risk
        strategy2 = PerformanceTracker("high_return_strategy")
        self._populate_tracker(strategy2, sharpe_target=1.2, return_target=0.25, drawdown_target=0.15)
        strategies["high_return_strategy"] = strategy2
        
        # Strategy 3: Conservative, low drawdown
        strategy3 = PerformanceTracker("conservative_strategy")
        self._populate_tracker(strategy3, sharpe_target=1.5, return_target=0.10, drawdown_target=0.05)
        strategies["conservative_strategy"] = strategy3
        
        # Strategy 4: Poor performance (should be filtered out or ranked low)
        strategy4 = PerformanceTracker("poor_strategy")
        self._populate_tracker(strategy4, sharpe_target=0.3, return_target=0.02, drawdown_target=0.25)
        strategies["poor_strategy"] = strategy4
        
        return strategies
    
    def _populate_tracker(self, tracker: PerformanceTracker, sharpe_target: float,
                         return_target: float, drawdown_target: float):
        """Populate a performance tracker with synthetic data"""
        base_time = datetime.now() - timedelta(days=60)
        
        # Calculate trade parameters to achieve target metrics
        num_trades = 30
        win_rate = 0.6  # 60% win rate
        winning_trades = int(num_trades * win_rate)
        num_trades - winning_trades
        
        # Calculate average win/loss to achieve target return
        avg_return_per_trade = return_target / num_trades
        avg_win = avg_return_per_trade / win_rate * 1.5  # Wins are larger
        avg_loss = -avg_return_per_trade / (1 - win_rate) * 0.5  # Losses are smaller
        
        trades = []
        cumulative_return = 0.0
        peak_return = 0.0
        max_drawdown = 0.0
        
        for i in range(num_trades):
            timestamp = base_time + timedelta(days=i * 2)
            
            # Determine if this is a winning or losing trade
            is_win = i < winning_trades
            
            if is_win:
                pnl_percent = avg_win * (0.8 + 0.4 * np.random.random())  # Add some variance
            else:
                pnl_percent = avg_loss * (0.8 + 0.4 * np.random.random())
            
            # Create trade
            trade = TradeResult(
                timestamp=timestamp,
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                exit_price=50000.0 * (1 + pnl_percent),
                quantity=0.001,
                pnl=50000.0 * 0.001 * pnl_percent,
                pnl_percent=pnl_percent,
                duration_hours=24.0,
                strategy_id=tracker.strategy_id,
                confidence=0.7,
                regime="trend_up_low_vol"
            )
            
            trades.append(trade)
            
            # Track drawdown
            cumulative_return += pnl_percent
            if cumulative_return > peak_return:
                peak_return = cumulative_return
            
            current_drawdown = (peak_return - cumulative_return) / max(0.01, peak_return)
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Add trades to tracker
        for trade in trades:
            tracker.record_trade(trade)
    
    def test_initialization(self):
        """Test StrategySelector initialization"""
        selector = StrategySelector()
        self.assertIsNotNone(selector.config)
        self.assertEqual(len(selector.performance_cache), 0)
        self.assertEqual(len(selector.correlation_matrix), 0)
    
    def test_config_validation_valid_weights(self):
        """Test configuration validation with valid weights"""
        config = SelectionConfig(
            sharpe_weight=0.3,
            return_weight=0.2,
            drawdown_weight=0.2,
            win_rate_weight=0.1,
            volatility_weight=0.1,
            regime_weight=0.1
        )
        # Should not raise an exception
        self.assertAlmostEqual(
            config.sharpe_weight + config.return_weight + config.drawdown_weight +
            config.win_rate_weight + config.volatility_weight + config.regime_weight,
            1.0, places=2
        )
    
    def test_config_validation_invalid_weights(self):
        """Test configuration validation with invalid weights"""
        with self.assertRaises(ValueError):
            SelectionConfig(
                sharpe_weight=0.5,  # Weights sum to more than 1.0
                return_weight=0.3,
                drawdown_weight=0.3,
                win_rate_weight=0.1,
                volatility_weight=0.1,
                regime_weight=0.1
            )
    
    def test_filter_eligible_strategies_sufficient_data(self):
        """Test strategy filtering with sufficient data"""
        eligible = self.selector._filter_eligible_strategies(self.strategies, [])
        
        # All strategies should be eligible (they have enough trades and time)
        self.assertEqual(len(eligible), len(self.strategies))
        self.assertIn("high_sharpe_strategy", eligible)
        self.assertIn("high_return_strategy", eligible)
        self.assertIn("conservative_strategy", eligible)
        self.assertIn("poor_strategy", eligible)
    
    def test_filter_eligible_strategies_insufficient_trades(self):
        """Test strategy filtering with insufficient trades"""
        # Create strategy with insufficient trades
        insufficient_strategy = PerformanceTracker("insufficient_strategy")
        for i in range(5):  # Only 5 trades (less than minimum 10)
            trade = TradeResult(
                timestamp=datetime.now() - timedelta(days=i),
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                exit_price=50100.0,
                quantity=0.001,
                pnl=100.0,
                pnl_percent=0.002,
                duration_hours=24.0,
                strategy_id="insufficient_strategy",
                confidence=0.7
            )
            insufficient_strategy.record_trade(trade)
        
        strategies_with_insufficient = self.strategies.copy()
        strategies_with_insufficient["insufficient_strategy"] = insufficient_strategy
        
        eligible = self.selector._filter_eligible_strategies(strategies_with_insufficient, [])
        
        # Should exclude the insufficient strategy
        self.assertNotIn("insufficient_strategy", eligible)
        self.assertEqual(len(eligible), len(self.strategies))
    
    def test_filter_eligible_strategies_excluded(self):
        """Test strategy filtering with exclusion list"""
        exclude_list = ["high_sharpe_strategy", "poor_strategy"]
        eligible = self.selector._filter_eligible_strategies(self.strategies, exclude_list)
        
        # Should exclude specified strategies
        self.assertNotIn("high_sharpe_strategy", eligible)
        self.assertNotIn("poor_strategy", eligible)
        self.assertIn("high_return_strategy", eligible)
        self.assertIn("conservative_strategy", eligible)
        self.assertEqual(len(eligible), 2)
    
    def test_normalize_score_higher_is_better(self):
        """Test score normalization for higher-is-better metrics"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test minimum value
        score_min = self.selector._normalize_score(1.0, values, higher_is_better=True)
        self.assertAlmostEqual(score_min, 0.0, places=2)
        
        # Test maximum value
        score_max = self.selector._normalize_score(5.0, values, higher_is_better=True)
        self.assertAlmostEqual(score_max, 1.0, places=2)
        
        # Test middle value
        score_mid = self.selector._normalize_score(3.0, values, higher_is_better=True)
        self.assertAlmostEqual(score_mid, 0.5, places=2)
    
    def test_normalize_score_lower_is_better(self):
        """Test score normalization for lower-is-better metrics"""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test minimum value (should get highest score)
        score_min = self.selector._normalize_score(0.1, values, higher_is_better=False)
        self.assertAlmostEqual(score_min, 1.0, places=2)
        
        # Test maximum value (should get lowest score)
        score_max = self.selector._normalize_score(0.5, values, higher_is_better=False)
        self.assertAlmostEqual(score_max, 0.0, places=2)
    
    def test_normalize_volatility_score_optimal_range(self):
        """Test volatility score normalization with optimal range"""
        volatilities = [0.05, 0.10, 0.15, 0.20, 0.30]
        
        # Test value in optimal range
        score_optimal = self.selector._normalize_volatility_score(0.12, volatilities)
        self.assertEqual(score_optimal, 1.0)
        
        # Test value too low
        score_low = self.selector._normalize_volatility_score(0.05, volatilities)
        self.assertLess(score_low, 1.0)
        
        # Test value too high
        score_high = self.selector._normalize_volatility_score(0.30, volatilities)
        self.assertLess(score_high, 1.0)
    
    def test_calculate_criteria_scores(self):
        """Test calculation of criteria scores"""
        # Get performance metrics for all strategies
        all_metrics = {}
        for strategy_id, tracker in self.strategies.items():
            all_metrics[strategy_id] = tracker.get_performance_metrics()
        
        # Test criteria scores for high Sharpe strategy
        high_sharpe_metrics = all_metrics["high_sharpe_strategy"]
        criteria_scores = self.selector._calculate_criteria_scores(high_sharpe_metrics, all_metrics)
        
        # Should have scores for all criteria
        expected_criteria = [
            SelectionCriteria.SHARPE_RATIO,
            SelectionCriteria.TOTAL_RETURN,
            SelectionCriteria.MAX_DRAWDOWN,
            SelectionCriteria.WIN_RATE,
            SelectionCriteria.VOLATILITY
        ]
        
        for criteria in expected_criteria:
            self.assertIn(criteria, criteria_scores)
            self.assertGreaterEqual(criteria_scores[criteria], 0.0)
            self.assertLessEqual(criteria_scores[criteria], 1.0)
        
        # High Sharpe strategy should score well on Sharpe ratio
        self.assertGreater(criteria_scores[SelectionCriteria.SHARPE_RATIO], 0.7)
    
    def test_calculate_risk_adjusted_score(self):
        """Test risk-adjusted score calculation"""
        # Create metrics with good risk profile
        good_metrics = PerformanceMetrics(
            total_return=500.0, total_return_pct=0.25, annualized_return=0.91,
            volatility=0.12, sharpe_ratio=1.8, sortino_ratio=2.2, calmar_ratio=4.5,
            max_drawdown=0.08, var_95=-0.02, total_trades=50, winning_trades=35,
            losing_trades=15, win_rate=0.7, avg_win=0.03, avg_loss=-0.01,
            profit_factor=2.1, expectancy=0.005, avg_trade_duration=20.0,
            trades_per_day=1.2, hit_rate=0.7, max_drawdown_duration=3.0,
            current_drawdown=0.05, drawdown_recovery_time=1.0, best_trade=0.06,
            worst_trade=-0.02, consecutive_wins=8, consecutive_losses=2,
            period_start=datetime.now() - timedelta(days=60),
            period_end=datetime.now(), period_type=PerformancePeriod.ALL_TIME
        )
        
        risk_score = self.selector._calculate_risk_adjusted_score(good_metrics)
        self.assertGreater(risk_score, 0.5)  # Should be good score
        
        # Create metrics with poor risk profile
        poor_metrics = PerformanceMetrics(
            total_return=100.0, total_return_pct=0.05, annualized_return=0.18,
            volatility=0.25, sharpe_ratio=0.2, sortino_ratio=0.3, calmar_ratio=0.6,
            max_drawdown=0.35, var_95=-0.08, total_trades=30, winning_trades=10,
            losing_trades=20, win_rate=0.33, avg_win=0.02, avg_loss=-0.015,
            profit_factor=0.8, expectancy=-0.002, avg_trade_duration=30.0,
            trades_per_day=0.8, hit_rate=0.33, max_drawdown_duration=10.0,
            current_drawdown=0.2, drawdown_recovery_time=8.0, best_trade=0.04,
            worst_trade=-0.06, consecutive_wins=2, consecutive_losses=8,
            period_start=datetime.now() - timedelta(days=60),
            period_end=datetime.now(), period_type=PerformancePeriod.ALL_TIME
        )
        
        poor_risk_score = self.selector._calculate_risk_adjusted_score(poor_metrics)
        self.assertLess(poor_risk_score, 0.3)  # Should be poor score
    
    def test_select_best_strategy_no_strategies(self):
        """Test strategy selection with no available strategies"""
        result = self.selector.select_best_strategy({})
        self.assertIsNone(result)
    
    def test_select_best_strategy_single_strategy(self):
        """Test strategy selection with single eligible strategy"""
        single_strategy = {"high_sharpe_strategy": self.strategies["high_sharpe_strategy"]}
        result = self.selector.select_best_strategy(single_strategy)
        self.assertEqual(result, "high_sharpe_strategy")
    
    def test_select_best_strategy_multiple_strategies(self):
        """Test strategy selection with multiple strategies"""
        result = self.selector.select_best_strategy(self.strategies, self.regime)
        
        # Should return a strategy ID
        self.assertIsNotNone(result)
        self.assertIn(result, self.strategies.keys())
        
        # Should not select the poor strategy
        self.assertNotEqual(result, "poor_strategy")
    
    def test_rank_strategies(self):
        """Test strategy ranking"""
        rankings = self.selector.rank_strategies(self.strategies, self.regime)
        
        # Should return all eligible strategies
        self.assertGreater(len(rankings), 0)
        self.assertLessEqual(len(rankings), len(self.strategies))
        
        # Should be sorted by score (descending)
        for i in range(1, len(rankings)):
            self.assertGreaterEqual(rankings[i-1].total_score, rankings[i].total_score)
        
        # Each ranking should have required fields
        for ranking in rankings:
            self.assertIsInstance(ranking, StrategyScore)
            self.assertIsNotNone(ranking.strategy_id)
            self.assertGreaterEqual(ranking.total_score, 0.0)
            self.assertLessEqual(ranking.total_score, 1.0)
    
    def test_compare_strategies(self):
        """Test strategy comparison"""
        strategy_ids = ["high_sharpe_strategy", "high_return_strategy", "conservative_strategy"]
        
        comparison = self.selector.compare_strategies(strategy_ids, self.strategies, self.regime)
        
        # Should have required fields
        self.assertIn('strategies', comparison)
        self.assertIn('correlation_matrix', comparison)
        self.assertIn('regime_context', comparison)
        self.assertIn('best_strategy', comparison)
        self.assertIn('comparison_timestamp', comparison)
        
        # Should have data for all compared strategies
        self.assertEqual(len(comparison['strategies']), 3)
        
        # Best strategy should be one of the compared strategies
        self.assertIn(comparison['best_strategy'], strategy_ids)
    
    def test_compare_strategies_insufficient_strategies(self):
        """Test strategy comparison with insufficient strategies"""
        with self.assertRaises(ValueError):
            self.selector.compare_strategies(["single_strategy"], self.strategies)
    
    def test_get_regime_specific_ranking(self):
        """Test regime-specific strategy ranking"""
        regime_type = "trend_up_low_vol"
        
        rankings = self.selector.get_regime_specific_ranking(self.strategies, regime_type)
        
        # Should return rankings for strategies with regime data
        self.assertGreater(len(rankings), 0)
        
        # Each ranking should be a tuple of (strategy_id, score)
        for strategy_id, score in rankings:
            self.assertIsInstance(strategy_id, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Should be sorted by score (descending)
        for i in range(1, len(rankings)):
            self.assertGreaterEqual(rankings[i-1][1], rankings[i][1])
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Populate cache
        for strategy_id, tracker in self.strategies.items():
            self.selector._get_cached_performance_metrics(strategy_id, tracker)
        
        # Verify cache is populated
        self.assertGreater(len(self.selector.performance_cache), 0)
        
        # Clear cache
        self.selector.clear_cache()
        
        # Verify cache is cleared
        self.assertEqual(len(self.selector.performance_cache), 0)
        self.assertEqual(len(self.selector.cache_expiry), 0)
        self.assertEqual(len(self.selector.correlation_matrix), 0)
    
    def test_cached_performance_metrics(self):
        """Test performance metrics caching"""
        strategy_id = "high_sharpe_strategy"
        tracker = self.strategies[strategy_id]
        
        # First call should calculate and cache
        metrics1 = self.selector._get_cached_performance_metrics(strategy_id, tracker)
        self.assertIn(strategy_id, self.selector.performance_cache)
        
        # Second call should return cached result
        metrics2 = self.selector._get_cached_performance_metrics(strategy_id, tracker)
        self.assertEqual(metrics1, metrics2)
    
    @patch('src.strategies.components.strategy_selector.pearsonr')
    def test_calculate_pairwise_correlation(self, mock_pearsonr):
        """Test pairwise correlation calculation"""
        mock_pearsonr.return_value = (0.75, 0.01)  # correlation, p-value
        
        tracker1 = self.strategies["high_sharpe_strategy"]
        tracker2 = self.strategies["high_return_strategy"]
        
        correlation = self.selector._calculate_pairwise_correlation(tracker1, tracker2)
        
        self.assertEqual(correlation, 0.75)
        mock_pearsonr.assert_called_once()
    
    def test_calculate_correlation_penalty(self):
        """Test correlation penalty calculation"""
        # Mock correlation matrix with high correlation
        correlation_matrix = {
            ("strategy1", "strategy2"): 0.8,  # High correlation
            ("strategy1", "strategy3"): 0.4,  # Moderate correlation
            ("strategy2", "strategy3"): 0.6   # Moderate correlation
        }
        
        penalty = self.selector._calculate_correlation_penalty("strategy1", correlation_matrix)
        
        # Should have penalty for high correlation
        self.assertGreater(penalty, 0.0)
        self.assertLessEqual(penalty, 1.0)


if __name__ == '__main__':
    unittest.main()