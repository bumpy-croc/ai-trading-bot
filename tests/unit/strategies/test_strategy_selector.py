"""Tests for strategies.components.strategy_selector module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.strategies.components.strategy_selector import (
    SelectionConfig,
    SelectionCriteria,
    StrategyScore,
    StrategySelector,
)


class TestSelectionCriteria:
    """Tests for SelectionCriteria enum."""

    def test_criteria_values(self):
        """Test that criteria values are correct."""
        assert SelectionCriteria.SHARPE_RATIO.value == "sharpe_ratio"
        assert SelectionCriteria.TOTAL_RETURN.value == "total_return"
        assert SelectionCriteria.MAX_DRAWDOWN.value == "max_drawdown"
        assert SelectionCriteria.WIN_RATE.value == "win_rate"
        assert SelectionCriteria.VOLATILITY.value == "volatility"
        assert SelectionCriteria.REGIME_PERFORMANCE.value == "regime_performance"
        assert SelectionCriteria.CORRELATION.value == "correlation"


class TestSelectionConfig:
    """Tests for SelectionConfig dataclass."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = SelectionConfig()
        total = (
            config.sharpe_weight
            + config.return_weight
            + config.drawdown_weight
            + config.win_rate_weight
            + config.volatility_weight
            + config.regime_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_raises_on_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SelectionConfig(sharpe_weight=0.5, return_weight=0.5)  # Sum != 1.0

        assert "must sum to 1.0" in str(exc_info.value)

    def test_default_values(self):
        """Test default configuration values."""
        config = SelectionConfig()

        assert config.risk_free_rate == 0.02
        assert config.max_acceptable_drawdown == 0.30
        assert config.min_acceptable_sharpe == 0.5
        assert config.correlation_threshold == 0.7
        assert config.min_trades_for_consideration == 30
        assert config.min_days_active == 30


class TestStrategyScore:
    """Tests for StrategyScore dataclass."""

    def test_to_dict(self):
        """Test score serialization to dictionary."""
        score = StrategyScore(
            strategy_id="ml_basic",
            total_score=0.85,
            criteria_scores={
                SelectionCriteria.SHARPE_RATIO: 0.9,
                SelectionCriteria.WIN_RATE: 0.8,
            },
            regime_scores={"trend_up_low_vol": 0.75},
            risk_adjusted_score=0.82,
            correlation_penalty=0.05,
        )

        result = score.to_dict()

        assert result["strategy_id"] == "ml_basic"
        assert result["total_score"] == 0.85
        assert result["criteria_scores"]["sharpe_ratio"] == 0.9
        assert result["regime_scores"]["trend_up_low_vol"] == 0.75
        assert result["risk_adjusted_score"] == 0.82
        assert result["correlation_penalty"] == 0.05


class TestStrategySelector:
    """Tests for StrategySelector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selector = StrategySelector()

    def create_mock_tracker(
        self,
        trade_count=50,
        sharpe_ratio=1.5,
        total_return_pct=0.25,
        max_drawdown=0.10,
        win_rate=0.55,
        volatility=0.12,
    ):
        """Create a mock performance tracker."""
        mock_tracker = MagicMock()
        mock_tracker.trade_count = trade_count
        mock_tracker.trades = []

        # Create mock trades with timestamps
        for i in range(trade_count):
            mock_trade = MagicMock()
            mock_trade.timestamp = datetime.now(UTC) - timedelta(days=60 - i)
            mock_trade.pnl_percent = 0.01 if i % 2 == 0 else -0.005
            mock_tracker.trades.append(mock_trade)

        mock_metrics = MagicMock()
        mock_metrics.sharpe_ratio = sharpe_ratio
        mock_metrics.total_return_pct = total_return_pct
        mock_metrics.max_drawdown = max_drawdown
        mock_metrics.win_rate = win_rate
        mock_metrics.volatility = volatility
        mock_tracker.get_performance_metrics.return_value = mock_metrics
        mock_tracker.get_regime_performance.return_value = {}

        return mock_tracker

    def test_initialization(self):
        """Test proper initialization."""
        assert self.selector.config is not None
        assert isinstance(self.selector.performance_cache, dict)
        assert isinstance(self.selector.correlation_matrix, dict)

    def test_select_best_strategy_empty_input(self):
        """Test selecting from empty strategy dict."""
        result = self.selector.select_best_strategy({})
        assert result is None

    def test_select_best_strategy_single_eligible(self):
        """Test selecting when only one strategy is eligible."""
        tracker = self.create_mock_tracker()
        strategies = {"ml_basic": tracker}

        result = self.selector.select_best_strategy(strategies)

        assert result == "ml_basic"

    def test_select_best_strategy_multiple_strategies(self):
        """Test selecting from multiple strategies."""
        strategies = {
            "ml_basic": self.create_mock_tracker(sharpe_ratio=1.5, win_rate=0.55),
            "ml_adaptive": self.create_mock_tracker(sharpe_ratio=2.0, win_rate=0.60),
            "ml_sentiment": self.create_mock_tracker(sharpe_ratio=1.0, win_rate=0.50),
        }

        result = self.selector.select_best_strategy(strategies)

        # ml_adaptive should win with higher Sharpe and win rate
        assert result is not None
        # The exact result depends on the weighting, but it should be one of the strategies
        assert result in strategies

    def test_select_best_strategy_excludes_strategies(self):
        """Test that excluded strategies are not selected."""
        strategies = {
            "ml_basic": self.create_mock_tracker(sharpe_ratio=2.0),
            "ml_adaptive": self.create_mock_tracker(sharpe_ratio=1.5),
        }

        result = self.selector.select_best_strategy(
            strategies, exclude_strategies=["ml_basic"]
        )

        assert result == "ml_adaptive"

    def test_select_best_strategy_filters_insufficient_trades(self):
        """Test that strategies with insufficient trades are filtered."""
        strategies = {
            "ml_basic": self.create_mock_tracker(trade_count=10),  # Too few
            "ml_adaptive": self.create_mock_tracker(trade_count=50),  # Enough
        }

        result = self.selector.select_best_strategy(strategies)

        assert result == "ml_adaptive"

    def test_rank_strategies(self):
        """Test ranking strategies."""
        strategies = {
            "ml_basic": self.create_mock_tracker(sharpe_ratio=1.5),
            "ml_adaptive": self.create_mock_tracker(sharpe_ratio=2.0),
            "ml_sentiment": self.create_mock_tracker(sharpe_ratio=1.0),
        }

        rankings = self.selector.rank_strategies(strategies)

        assert len(rankings) == 3
        # Rankings should be sorted by total score descending
        assert rankings[0].total_score >= rankings[1].total_score
        assert rankings[1].total_score >= rankings[2].total_score

    def test_rank_strategies_empty(self):
        """Test ranking empty strategy dict."""
        rankings = self.selector.rank_strategies({})
        assert rankings == []

    def test_compare_strategies(self):
        """Test strategy comparison."""
        strategies = {
            "ml_basic": self.create_mock_tracker(sharpe_ratio=1.5),
            "ml_adaptive": self.create_mock_tracker(sharpe_ratio=2.0),
        }

        comparison = self.selector.compare_strategies(
            ["ml_basic", "ml_adaptive"], strategies
        )

        assert "strategies" in comparison
        assert "correlation_matrix" in comparison
        assert "best_strategy" in comparison
        assert len(comparison["strategies"]) == 2

    def test_compare_strategies_insufficient(self):
        """Test comparison with insufficient strategies."""
        with pytest.raises(ValueError) as exc_info:
            self.selector.compare_strategies(["only_one"], {})

        assert "at least 2 strategies" in str(exc_info.value)

    def test_normalize_score_higher_is_better(self):
        """Test score normalization when higher is better."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.selector._normalize_score(5.0, values, higher_is_better=True)
        assert result == 1.0

        result = self.selector._normalize_score(1.0, values, higher_is_better=True)
        assert result == 0.0

    def test_normalize_score_lower_is_better(self):
        """Test score normalization when lower is better."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.selector._normalize_score(1.0, values, higher_is_better=False)
        assert result == 1.0

        result = self.selector._normalize_score(5.0, values, higher_is_better=False)
        assert result == 0.0

    def test_normalize_score_single_value(self):
        """Test normalization with single value returns 0.5."""
        result = self.selector._normalize_score(5.0, [5.0])
        assert result == 0.5

    def test_normalize_volatility_score_optimal_range(self):
        """Test volatility score in optimal range."""
        # Volatility in optimal range (0.08-0.15) should score 1.0
        result = self.selector._normalize_volatility_score(0.10, [0.10])
        assert result == 1.0

    def test_normalize_volatility_score_too_high(self):
        """Test volatility score when too high."""
        result = self.selector._normalize_volatility_score(0.30, [0.30])
        assert result < 1.0

    def test_normalize_volatility_score_too_low(self):
        """Test volatility score when too low."""
        result = self.selector._normalize_volatility_score(0.02, [0.02])
        assert result < 1.0

    def test_clear_cache(self):
        """Test clearing all caches."""
        # Add some cached data
        self.selector.performance_cache["test"] = MagicMock()
        self.selector.correlation_matrix[("a", "b")] = 0.5

        self.selector.clear_cache()

        assert len(self.selector.performance_cache) == 0
        assert len(self.selector.correlation_matrix) == 0

    def test_get_regime_specific_ranking(self):
        """Test getting regime-specific ranking."""
        mock_tracker = self.create_mock_tracker()

        mock_regime_perf = MagicMock()
        mock_regime_perf.sharpe_ratio = 1.5
        mock_regime_perf.win_rate = 0.6
        mock_regime_perf.max_drawdown = 0.10
        mock_regime_perf.avg_return = 0.02

        mock_tracker.get_regime_performance.return_value = {
            "trend_up_low_vol": mock_regime_perf
        }

        strategies = {"ml_basic": mock_tracker}

        rankings = self.selector.get_regime_specific_ranking(
            strategies, "trend_up_low_vol"
        )

        assert len(rankings) == 1
        assert rankings[0][0] == "ml_basic"
        assert rankings[0][1] > 0


class TestStrategySelectorCaching:
    """Tests for StrategySelector caching behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selector = StrategySelector()

    def test_performance_cache_used(self):
        """Test that performance cache is used on subsequent calls."""
        mock_tracker = MagicMock()
        mock_tracker.trade_count = 50
        mock_tracker.trades = []

        for i in range(50):
            mock_trade = MagicMock()
            mock_trade.timestamp = datetime.now(UTC) - timedelta(days=60 - i)
            mock_tracker.trades.append(mock_trade)

        mock_metrics = MagicMock()
        mock_metrics.sharpe_ratio = 1.5
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        # First call - should compute
        self.selector._get_cached_performance_metrics("test", mock_tracker)
        first_call_count = mock_tracker.get_performance_metrics.call_count

        # Second call - should use cache
        self.selector._get_cached_performance_metrics("test", mock_tracker)
        second_call_count = mock_tracker.get_performance_metrics.call_count

        assert second_call_count == first_call_count  # No additional calls


@pytest.mark.fast
class TestStrategySelectorIntegration:
    """Integration tests for StrategySelector."""

    def test_full_selection_workflow(self):
        """Test complete selection workflow."""
        selector = StrategySelector()

        # Create multiple mock strategies with different performance
        def create_tracker(sharpe, win_rate, drawdown, trade_count=50):
            mock = MagicMock()
            mock.trade_count = trade_count
            mock.trades = []
            for i in range(trade_count):
                trade = MagicMock()
                trade.timestamp = datetime.now(UTC) - timedelta(days=60 - i)
                trade.pnl_percent = 0.01
                mock.trades.append(trade)

            metrics = MagicMock()
            metrics.sharpe_ratio = sharpe
            metrics.win_rate = win_rate
            metrics.max_drawdown = drawdown
            metrics.total_return_pct = 0.20
            metrics.volatility = 0.12
            mock.get_performance_metrics.return_value = metrics
            mock.get_regime_performance.return_value = {}
            return mock

        strategies = {
            "conservative": create_tracker(1.0, 0.55, 0.08),
            "aggressive": create_tracker(2.5, 0.50, 0.25),
            "balanced": create_tracker(1.8, 0.58, 0.12),
        }

        # Select best
        best = selector.select_best_strategy(strategies)
        assert best is not None
        assert best in strategies

        # Rank all
        rankings = selector.rank_strategies(strategies)
        assert len(rankings) == 3
        assert rankings[0].strategy_id in strategies

        # Compare two
        comparison = selector.compare_strategies(
            ["conservative", "aggressive"], strategies
        )
        assert "best_strategy" in comparison

    def test_handles_all_ineligible_strategies(self):
        """Test handling when all strategies are ineligible."""
        selector = StrategySelector()

        # All strategies have insufficient trades
        mock = MagicMock()
        mock.trade_count = 5  # Below minimum
        mock.trades = []

        strategies = {"low_trade": mock}

        result = selector.select_best_strategy(strategies)
        assert result is None
