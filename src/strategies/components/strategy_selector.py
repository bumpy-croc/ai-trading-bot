"""
Strategy Selection Algorithm

This module implements a multi-criteria strategy selection algorithm with
regime-specific performance weighting, risk-adjusted selection, and
correlation analysis to avoid selecting similar strategies.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from .performance_tracker import PerformanceMetrics, PerformanceTracker
from .regime_context import RegimeContext


class SelectionCriteria(Enum):
    """Criteria for strategy selection"""

    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    VOLATILITY = "volatility"
    REGIME_PERFORMANCE = "regime_performance"
    CORRELATION = "correlation"


@dataclass
class StrategyScore:
    """Score breakdown for a strategy"""

    strategy_id: str
    total_score: float
    criteria_scores: dict[SelectionCriteria, float]
    regime_scores: dict[str, float]
    risk_adjusted_score: float
    correlation_penalty: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "strategy_id": self.strategy_id,
            "total_score": self.total_score,
            "criteria_scores": {k.value: v for k, v in self.criteria_scores.items()},
            "regime_scores": self.regime_scores,
            "risk_adjusted_score": self.risk_adjusted_score,
            "correlation_penalty": self.correlation_penalty,
        }


@dataclass
class SelectionConfig:
    """Configuration for strategy selection algorithm"""

    # Criteria weights (must sum to 1.0)
    sharpe_weight: float = 0.25
    return_weight: float = 0.20
    drawdown_weight: float = 0.20
    win_rate_weight: float = 0.15
    volatility_weight: float = 0.10
    regime_weight: float = 0.10

    # Risk adjustment parameters
    risk_free_rate: float = 0.02  # Annual risk-free rate
    max_acceptable_drawdown: float = 0.30  # 30%
    min_acceptable_sharpe: float = 0.5

    # Correlation parameters
    correlation_threshold: float = 0.7  # High correlation threshold
    correlation_penalty_factor: float = 0.2  # Penalty for high correlation

    # Regime-specific parameters
    regime_lookback_days: int = 90
    regime_confidence_threshold: float = 0.6

    # Performance requirements
    min_trades_for_consideration: int = 30
    min_days_active: int = 30

    def __post_init__(self):
        """Validate configuration after initialization"""
        total_weight = (
            self.sharpe_weight
            + self.return_weight
            + self.drawdown_weight
            + self.win_rate_weight
            + self.volatility_weight
            + self.regime_weight
        )

        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")


class StrategySelector:
    """
    Multi-criteria strategy selection algorithm

    This class implements sophisticated strategy selection using multiple
    performance criteria, regime-specific weighting, risk adjustment,
    and correlation analysis.
    """

    def __init__(self, config: SelectionConfig | None = None):
        """
        Initialize strategy selector

        Args:
            config: Configuration for selection algorithm
        """
        self.config = config or SelectionConfig()
        self.logger = logging.getLogger("StrategySelector")

        # Strategy performance cache
        self.performance_cache: dict[str, PerformanceMetrics] = {}
        self.cache_expiry: dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

        # Correlation matrix cache
        self.correlation_matrix: dict[tuple[str, str], float] = {}
        self.correlation_cache_expiry = datetime.min
        self.correlation_strategy_set: frozenset[str] | None = (
            None  # Track which strategies are cached
        )

        # Thread safety locks
        self._cache_lock = threading.RLock()
        self._correlation_lock = threading.RLock()

        self.logger.info("StrategySelector initialized")

    def select_best_strategy(
        self,
        available_strategies: dict[str, PerformanceTracker],
        current_regime: RegimeContext | None = None,
        exclude_strategies: list[str] | None = None,
    ) -> str | None:
        """
        Select the best strategy based on multi-criteria analysis

        Args:
            available_strategies: Dictionary mapping strategy IDs to performance trackers
            current_regime: Current market regime context
            exclude_strategies: List of strategy IDs to exclude from selection

        Returns:
            Strategy ID of the best strategy, or None if no suitable strategy found
        """
        if not available_strategies:
            self.logger.warning("No strategies available for selection")
            return None

        exclude_strategies = exclude_strategies or []

        # Filter strategies based on minimum requirements
        eligible_strategies = self._filter_eligible_strategies(
            available_strategies, exclude_strategies
        )

        if not eligible_strategies:
            self.logger.warning("No eligible strategies found")
            return None

        if len(eligible_strategies) == 1:
            strategy_id = list(eligible_strategies.keys())[0]
            self.logger.info(f"Only one eligible strategy: {strategy_id}")
            return strategy_id

        # Calculate scores for all eligible strategies
        strategy_scores = self._calculate_strategy_scores(eligible_strategies, current_regime)

        # Sort by total score (descending)
        sorted_strategies = sorted(strategy_scores, key=lambda s: s.total_score, reverse=True)

        best_strategy = sorted_strategies[0]

        self.logger.info(
            f"Selected best strategy: {best_strategy.strategy_id} "
            f"(score: {best_strategy.total_score:.3f})"
        )

        return best_strategy.strategy_id

    def rank_strategies(
        self,
        available_strategies: dict[str, PerformanceTracker],
        current_regime: RegimeContext | None = None,
        exclude_strategies: list[str] | None = None,
    ) -> list[StrategyScore]:
        """
        Rank all strategies by their selection scores

        Args:
            available_strategies: Dictionary mapping strategy IDs to performance trackers
            current_regime: Current market regime context
            exclude_strategies: List of strategy IDs to exclude from ranking

        Returns:
            List of StrategyScore objects sorted by total score (descending)
        """
        exclude_strategies = exclude_strategies or []

        # Filter strategies based on minimum requirements
        eligible_strategies = self._filter_eligible_strategies(
            available_strategies, exclude_strategies
        )

        if not eligible_strategies:
            return []

        # Calculate scores for all eligible strategies
        strategy_scores = self._calculate_strategy_scores(eligible_strategies, current_regime)

        # Sort by total score (descending)
        return sorted(strategy_scores, key=lambda s: s.total_score, reverse=True)

    def compare_strategies(
        self,
        strategy_ids: list[str],
        performance_trackers: dict[str, PerformanceTracker],
        current_regime: RegimeContext | None = None,
    ) -> dict[str, Any]:
        """
        Compare specific strategies with detailed analysis

        Args:
            strategy_ids: List of strategy IDs to compare
            performance_trackers: Dictionary mapping strategy IDs to performance trackers
            current_regime: Current market regime context

        Returns:
            Detailed comparison results
        """
        if len(strategy_ids) < 2:
            raise ValueError("Need at least 2 strategies to compare")

        # Get performance trackers for specified strategies
        strategies_to_compare = {
            sid: tracker for sid, tracker in performance_trackers.items() if sid in strategy_ids
        }

        if len(strategies_to_compare) < 2:
            raise ValueError("Not enough valid strategies found for comparison")

        # Calculate scores
        strategy_scores = self._calculate_strategy_scores(strategies_to_compare, current_regime)

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(strategies_to_compare)

        # Prepare comparison results
        comparison = {
            "strategies": [score.to_dict() for score in strategy_scores],
            "correlation_matrix": correlation_matrix,
            "regime_context": {
                "trend": current_regime.trend.value if current_regime else None,
                "volatility": current_regime.volatility.value if current_regime else None,
                "confidence": current_regime.confidence if current_regime else None,
            },
            "best_strategy": max(strategy_scores, key=lambda s: s.total_score).strategy_id,
            "comparison_timestamp": datetime.now(UTC).isoformat(),
        }

        return comparison

    def get_regime_specific_ranking(
        self, available_strategies: dict[str, PerformanceTracker], regime_type: str
    ) -> list[tuple[str, float]]:
        """
        Get strategy ranking for a specific regime type

        Args:
            available_strategies: Dictionary mapping strategy IDs to performance trackers
            regime_type: Regime type to analyze (e.g., "trend_up_low_vol")

        Returns:
            List of (strategy_id, regime_score) tuples sorted by regime score
        """
        regime_scores = []

        for strategy_id, tracker in available_strategies.items():
            # Get regime-specific performance
            regime_performance = tracker.get_regime_performance(regime_type)

            if regime_type in regime_performance:
                perf = regime_performance[regime_type]

                # Calculate regime-specific score
                score = self._calculate_regime_score(perf)
                regime_scores.append((strategy_id, score))

        # Sort by score (descending)
        return sorted(regime_scores, key=lambda x: x[1], reverse=True)

    def _filter_eligible_strategies(
        self, available_strategies: dict[str, PerformanceTracker], exclude_strategies: list[str]
    ) -> dict[str, PerformanceTracker]:
        """Filter strategies based on minimum requirements"""
        eligible = {}

        for strategy_id, tracker in available_strategies.items():
            if strategy_id in exclude_strategies:
                continue

            # Check minimum trade count
            if tracker.trade_count < self.config.min_trades_for_consideration:
                self.logger.debug(
                    f"Strategy {strategy_id} excluded: insufficient trades "
                    f"({tracker.trade_count} < {self.config.min_trades_for_consideration})"
                )
                continue

            # Check minimum active days
            if tracker.trades:
                oldest_trade = min(tracker.trades, key=lambda t: t.timestamp)
                days_active = (datetime.now(UTC) - oldest_trade.timestamp).days

                if days_active < self.config.min_days_active:
                    self.logger.debug(
                        f"Strategy {strategy_id} excluded: insufficient active days "
                        f"({days_active} < {self.config.min_days_active})"
                    )
                    continue

            eligible[strategy_id] = tracker

        return eligible

    def _calculate_strategy_scores(
        self, strategies: dict[str, PerformanceTracker], current_regime: RegimeContext | None
    ) -> list[StrategyScore]:
        """Calculate comprehensive scores for all strategies"""
        strategy_scores = []

        # Get performance metrics for all strategies
        all_metrics = {}
        for strategy_id, tracker in strategies.items():
            metrics = self._get_cached_performance_metrics(strategy_id, tracker)
            all_metrics[strategy_id] = metrics

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(strategies)

        for strategy_id, tracker in strategies.items():
            metrics = all_metrics[strategy_id]

            # Calculate individual criteria scores
            criteria_scores = self._calculate_criteria_scores(metrics, all_metrics)

            # Calculate regime-specific scores
            regime_scores = self._calculate_regime_scores(tracker, current_regime)

            # Calculate risk-adjusted score
            risk_adjusted_score = self._calculate_risk_adjusted_score(metrics)

            # Calculate correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(
                strategy_id, correlation_matrix
            )

            # Calculate total score
            total_score = self._calculate_total_score(
                criteria_scores,
                regime_scores,
                risk_adjusted_score,
                correlation_penalty,
                current_regime,
            )

            strategy_score = StrategyScore(
                strategy_id=strategy_id,
                total_score=total_score,
                criteria_scores=criteria_scores,
                regime_scores=regime_scores,
                risk_adjusted_score=risk_adjusted_score,
                correlation_penalty=correlation_penalty,
            )

            strategy_scores.append(strategy_score)

        return strategy_scores

    def _calculate_criteria_scores(
        self, metrics: PerformanceMetrics, all_metrics: dict[str, PerformanceMetrics]
    ) -> dict[SelectionCriteria, float]:
        """Calculate normalized scores for each selection criteria"""
        scores = {}

        # Get all values for normalization
        all_sharpe = [m.sharpe_ratio for m in all_metrics.values()]
        all_returns = [m.total_return_pct for m in all_metrics.values()]
        all_drawdowns = [m.max_drawdown for m in all_metrics.values()]
        all_win_rates = [m.win_rate for m in all_metrics.values()]
        all_volatilities = [m.volatility for m in all_metrics.values()]

        # Sharpe ratio (higher is better)
        scores[SelectionCriteria.SHARPE_RATIO] = self._normalize_score(
            metrics.sharpe_ratio, all_sharpe, higher_is_better=True
        )

        # Total return (higher is better)
        scores[SelectionCriteria.TOTAL_RETURN] = self._normalize_score(
            metrics.total_return_pct, all_returns, higher_is_better=True
        )

        # Max drawdown (lower is better)
        scores[SelectionCriteria.MAX_DRAWDOWN] = self._normalize_score(
            metrics.max_drawdown, all_drawdowns, higher_is_better=False
        )

        # Win rate (higher is better)
        scores[SelectionCriteria.WIN_RATE] = self._normalize_score(
            metrics.win_rate, all_win_rates, higher_is_better=True
        )

        # Volatility (lower is better, but not too low)
        scores[SelectionCriteria.VOLATILITY] = self._normalize_volatility_score(
            metrics.volatility, all_volatilities
        )

        return scores

    def _calculate_regime_scores(
        self, tracker: PerformanceTracker, current_regime: RegimeContext | None
    ) -> dict[str, float]:
        """Calculate regime-specific performance scores"""
        regime_scores = {}

        # Get regime performance data
        regime_performance = tracker.get_regime_performance()

        for regime_type, perf in regime_performance.items():
            regime_scores[regime_type] = self._calculate_regime_score(perf)

        # If current regime is specified, boost its weight
        if current_regime:
            current_regime_key = f"{current_regime.trend.value}_{current_regime.volatility.value}"
            if current_regime_key in regime_scores:
                # Boost current regime score based on regime confidence
                boost_factor = 1.0 + (current_regime.confidence * 0.5)
                regime_scores[current_regime_key] *= boost_factor

        return regime_scores

    def _calculate_regime_score(self, regime_perf) -> float:
        """Calculate score for regime-specific performance"""
        import math

        # Validate inputs are finite to prevent NaN/inf propagation
        # Return 0.0 for invalid metrics rather than propagating corrupt values
        if not all(
            math.isfinite(getattr(regime_perf, attr, 0.0))
            for attr in ["sharpe_ratio", "win_rate", "max_drawdown", "avg_return"]
        ):
            return 0.0

        # Weighted combination of regime performance metrics
        sharpe_score = min(1.0, max(0.0, regime_perf.sharpe_ratio / 2.0))
        win_rate_score = min(1.0, max(0.0, (regime_perf.win_rate - 0.3) / 0.4))
        drawdown_score = min(1.0, max(0.0, 1.0 - (regime_perf.max_drawdown / 0.3)))
        return_score = min(1.0, max(0.0, regime_perf.avg_return / 0.05))

        # Weighted combination
        score = (
            sharpe_score * 0.3 + win_rate_score * 0.25 + drawdown_score * 0.25 + return_score * 0.2
        )

        return score

    def _calculate_risk_adjusted_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate risk-adjusted performance score"""
        # Penalize strategies that exceed risk thresholds
        drawdown_penalty = 0.0
        if metrics.max_drawdown > self.config.max_acceptable_drawdown:
            drawdown_penalty = (metrics.max_drawdown - self.config.max_acceptable_drawdown) * 2.0

        sharpe_penalty = 0.0
        if metrics.sharpe_ratio < self.config.min_acceptable_sharpe:
            sharpe_penalty = (self.config.min_acceptable_sharpe - metrics.sharpe_ratio) * 0.5

        # Base score from Sharpe ratio
        base_score = min(1.0, max(0.0, metrics.sharpe_ratio / 3.0))

        # Apply penalties
        risk_adjusted_score = max(0.0, base_score - drawdown_penalty - sharpe_penalty)

        return risk_adjusted_score

    def _calculate_correlation_penalty(
        self, strategy_id: str, correlation_matrix: dict[tuple[str, str], float]
    ) -> float:
        """Calculate penalty for high correlation with other strategies"""
        correlations = []
        processed_pairs = set()  # Track which pairs we've already processed

        for (sid1, sid2), correlation in correlation_matrix.items():
            if sid1 == strategy_id or sid2 == strategy_id:
                if sid1 != sid2:  # Don't include self-correlation
                    # Create a canonical pair representation to avoid double-counting
                    # Use lexicographic ordering to ensure consistent pair representation
                    pair = tuple(sorted([sid1, sid2]))
                    if pair not in processed_pairs:
                        correlations.append(abs(correlation))
                        processed_pairs.add(pair)

        if not correlations:
            return 0.0

        # Calculate penalty based on highest correlations
        max_correlation = max(correlations)
        avg_correlation = statistics.mean(correlations)

        penalty = 0.0

        # Penalty for high maximum correlation
        if max_correlation > self.config.correlation_threshold:
            penalty += (
                max_correlation - self.config.correlation_threshold
            ) * self.config.correlation_penalty_factor

        # Additional penalty for high average correlation
        if avg_correlation > 0.5:
            penalty += (avg_correlation - 0.5) * self.config.correlation_penalty_factor * 0.5

        return min(1.0, penalty)  # Cap penalty at 1.0

    def _calculate_total_score(
        self,
        criteria_scores: dict[SelectionCriteria, float],
        regime_scores: dict[str, float],
        risk_adjusted_score: float,
        correlation_penalty: float,
        current_regime: RegimeContext | None,
    ) -> float:
        """Calculate total weighted score for strategy"""
        # Base score from criteria
        base_score = (
            criteria_scores[SelectionCriteria.SHARPE_RATIO] * self.config.sharpe_weight
            + criteria_scores[SelectionCriteria.TOTAL_RETURN] * self.config.return_weight
            + criteria_scores[SelectionCriteria.MAX_DRAWDOWN] * self.config.drawdown_weight
            + criteria_scores[SelectionCriteria.WIN_RATE] * self.config.win_rate_weight
            + criteria_scores[SelectionCriteria.VOLATILITY] * self.config.volatility_weight
        )

        # Add regime score
        regime_score = 0.0
        if regime_scores and current_regime:
            current_regime_key = f"{current_regime.trend.value}_{current_regime.volatility.value}"
            regime_score = regime_scores.get(current_regime_key, 0.0)
        elif regime_scores:
            # Use average regime score if no current regime specified
            regime_score = statistics.mean(regime_scores.values())

        base_score += regime_score * self.config.regime_weight

        # Apply risk adjustment
        total_score = base_score * (0.7 + 0.3 * risk_adjusted_score)

        # Apply correlation penalty
        total_score *= 1.0 - correlation_penalty

        return max(0.0, min(1.0, total_score))

    def _calculate_correlation_matrix(
        self, strategies: dict[str, PerformanceTracker]
    ) -> dict[tuple[str, str], float]:
        """Calculate correlation matrix between strategies (thread-safe with double-checked locking)"""
        strategy_ids = list(strategies.keys())
        current_strategy_set = frozenset(strategy_ids)

        # Calculate cache version based on strategy data freshness
        cache_version = self._calculate_strategy_cache_version(strategies)

        # First check without holding lock during computation (double-checked locking)
        with self._correlation_lock:
            # Check if cache is still valid AND strategy set hasn't changed AND cache version matches
            if (
                datetime.now(UTC) < self.correlation_cache_expiry
                and self.correlation_strategy_set == current_strategy_set
                and hasattr(self, "correlation_cache_version")
                and self.correlation_cache_version == cache_version
            ):
                return self.correlation_matrix

            # Check if another thread is already computing for this strategy set
            if (
                hasattr(self, "_computing_strategy_set")
                and self._computing_strategy_set == current_strategy_set
            ):
                # Another thread is computing, wait for it to complete
                while (
                    hasattr(self, "_computing_strategy_set")
                    and self._computing_strategy_set == current_strategy_set
                ):
                    self._correlation_lock.release()
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    self._correlation_lock.acquire()

                # Re-check cache after waiting
                if (
                    datetime.now(UTC) < self.correlation_cache_expiry
                    and self.correlation_strategy_set == current_strategy_set
                ):
                    return self.correlation_matrix

            # Mark that we're computing to prevent other threads from computing
            self._computing_strategy_set = current_strategy_set

        # Calculate correlation matrix outside lock to avoid holding it during computation
        # Pre-compute all daily returns for vectorized correlation calculation
        cutoff_date = datetime.now(UTC) - timedelta(days=90)
        strategy_returns: dict[str, dict] = {}

        for sid in strategy_ids:
            trades = [t for t in strategies[sid].trades if t.timestamp >= cutoff_date]
            returns_by_date = defaultdict(list)
            for trade in trades:
                date_key = trade.timestamp.date()
                returns_by_date[date_key].append(trade.pnl_percent)

            # Convert to daily returns
            daily_returns = {date: sum(returns) for date, returns in returns_by_date.items()}
            strategy_returns[sid] = daily_returns

        # Calculate correlations using vectorized operations where possible
        correlation_matrix = {}

        for i, sid1 in enumerate(strategy_ids):
            for j, sid2 in enumerate(strategy_ids):
                if i < j:  # Only calculate upper triangle
                    correlation = self._calculate_pairwise_correlation_from_daily_returns(
                        strategy_returns[sid1], strategy_returns[sid2]
                    )
                    correlation_matrix[(sid1, sid2)] = correlation
                    correlation_matrix[(sid2, sid1)] = correlation
                elif i == j:
                    correlation_matrix[(sid1, sid2)] = 1.0

        with self._correlation_lock:
            # Double-check: another thread might have computed while we were calculating
            if (
                datetime.now(UTC) < self.correlation_cache_expiry
                and self.correlation_strategy_set == current_strategy_set
            ):
                # Another thread already updated the cache, use that
                delattr(self, "_computing_strategy_set")  # Clear computing flag
                return self.correlation_matrix

            # Update cache with strategy set tracking and version
            self.correlation_matrix = correlation_matrix
            self.correlation_cache_expiry = datetime.now(UTC) + timedelta(hours=1)
            self.correlation_strategy_set = current_strategy_set
            self.correlation_cache_version = cache_version

            # Clear computing flag
            if hasattr(self, "_computing_strategy_set"):
                delattr(self, "_computing_strategy_set")

        return correlation_matrix

    def _calculate_strategy_cache_version(self, strategies: dict[str, PerformanceTracker]) -> str:
        """Calculate cache version based on strategy data freshness"""
        version_parts = []
        for strategy_id in sorted(strategies.keys()):
            tracker = strategies[strategy_id]
            # Use the timestamp of the most recent trade as version component
            if tracker.trades:
                latest_trade_time = max(trade.timestamp for trade in tracker.trades)
                version_parts.append(f"{strategy_id}:{latest_trade_time.timestamp()}")
            else:
                version_parts.append(f"{strategy_id}:0")

        # Also include the number of trades to detect when strategies change significantly
        total_trades = sum(len(tracker.trades) for tracker in strategies.values())
        version_parts.append(f"total_trades:{total_trades}")

        return "|".join(version_parts)

    def _calculate_pairwise_correlation_from_daily_returns(
        self, daily_returns1: dict, daily_returns2: dict
    ) -> float:
        """Calculate correlation from pre-computed daily returns (optimized)"""
        # Find common dates
        common_dates = set(daily_returns1.keys()) & set(daily_returns2.keys())

        if len(common_dates) < 10:
            return 0.0  # Insufficient overlapping data

        # Extract returns for common dates (vectorized)
        returns1 = np.array([daily_returns1[date] for date in sorted(common_dates)])
        returns2 = np.array([daily_returns2[date] for date in sorted(common_dates)])

        # Calculate Pearson correlation using numpy (faster)
        try:
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            return 0.0

    def _normalize_score(
        self, value: float, all_values: list[float], higher_is_better: bool = True
    ) -> float:
        """Normalize a score to 0-1 range"""
        if not all_values or len(all_values) == 1:
            return 0.5

        min_val = min(all_values)
        max_val = max(all_values)

        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)

        if not higher_is_better:
            normalized = 1.0 - normalized

        return max(0.0, min(1.0, normalized))

    def _normalize_volatility_score(
        self, volatility: float, all_volatilities: list[float]
    ) -> float:
        """Normalize volatility score (optimal range, not too high or too low)"""
        if not all_volatilities:
            return 0.5

        # Target volatility range (e.g., 0.1 to 0.2 for crypto)
        target_min = 0.08
        target_max = 0.15

        if target_min <= volatility <= target_max:
            return 1.0
        elif volatility < target_min:
            # Penalize too low volatility (might indicate insufficient opportunities)
            return max(0.0, volatility / target_min)
        else:
            # Penalize too high volatility
            penalty = (volatility - target_max) / target_max
            return max(0.0, 1.0 - penalty)

    def _get_cached_performance_metrics(
        self, strategy_id: str, tracker: PerformanceTracker
    ) -> PerformanceMetrics:
        """Get performance metrics with caching (thread-safe)"""
        with self._cache_lock:
            # Check cache
            if (
                strategy_id in self.performance_cache
                and strategy_id in self.cache_expiry
                and datetime.now(UTC) < self.cache_expiry[strategy_id]
            ):
                return self.performance_cache[strategy_id]

            # Calculate fresh metrics (outside lock to avoid holding it during computation)

        metrics = tracker.get_performance_metrics()

        with self._cache_lock:
            # Update cache
            self.performance_cache[strategy_id] = metrics
            self.cache_expiry[strategy_id] = datetime.now(UTC) + self.cache_duration

        return metrics

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.performance_cache.clear()
        self.cache_expiry.clear()
        self.correlation_matrix.clear()
        self.correlation_cache_expiry = datetime.min

        self.logger.info("Strategy selector cache cleared")
