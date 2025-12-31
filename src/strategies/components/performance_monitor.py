"""
Performance Monitor with Degradation Detection

This module implements sophisticated performance monitoring with multi-timeframe
analysis, statistical significance testing, and regime-aware evaluation for
automatic strategy switching decisions.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config.constants import DEFAULT_MAX_DRAWDOWN

from .performance_tracker import PerformanceMetrics, PerformanceTracker
from .regime_context import RegimeContext


class DegradationSeverity(Enum):
    """Severity levels for performance degradation"""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class TimeFrame(Enum):
    """Time frames for performance analysis"""

    SHORT_TERM = "short_term"  # 7 days
    MEDIUM_TERM = "medium_term"  # 30 days
    LONG_TERM = "long_term"  # 90 days


@dataclass
class PerformanceDegradationConfig:
    """Configuration for performance degradation detection"""

    # Statistical requirements
    min_trades_for_evaluation: int = 50
    min_days_for_evaluation: int = 30

    # Multi-timeframe thresholds
    short_term_days: int = 7
    medium_term_days: int = 30
    long_term_days: int = 90

    # Performance thresholds (all must be met for degradation)
    max_drawdown_threshold: float = DEFAULT_MAX_DRAWDOWN  # 20% max drawdown
    sharpe_ratio_threshold: float = 0.5  # Minimum acceptable Sharpe
    win_rate_threshold: float = 0.35  # Minimum win rate

    # Statistical significance
    confidence_level: float = 0.95  # 95% confidence for underperformance

    # Regime-specific considerations
    regime_specific_evaluation: bool = True
    min_regime_duration_days: int = 14  # Don't switch during regime transitions


@dataclass
class TimeFrameAnalysis:
    """Analysis results for a specific time frame"""

    timeframe: TimeFrame
    period_days: int
    trade_count: int
    current_performance: PerformanceMetrics
    historical_performance: PerformanceMetrics
    underperforming: bool
    statistical_significance: float
    confidence_interval: tuple[float, float]
    degradation_severity: DegradationSeverity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timeframe": self.timeframe.value,
            "period_days": self.period_days,
            "trade_count": self.trade_count,
            "current_performance": self.current_performance.to_dict(),
            "historical_performance": self.historical_performance.to_dict(),
            "underperforming": self.underperforming,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": self.confidence_interval,
            "degradation_severity": self.degradation_severity.value,
        }


@dataclass
class SwitchDecision:
    """Decision result for strategy switching"""

    should_switch: bool
    reason: str
    confidence: float
    recommended_strategy: str | None = None
    degradation_severity: DegradationSeverity = DegradationSeverity.NONE
    timeframe_results: list[TimeFrameAnalysis] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "should_switch": self.should_switch,
            "reason": self.reason,
            "confidence": self.confidence,
            "recommended_strategy": self.recommended_strategy,
            "degradation_severity": self.degradation_severity.value,
            "timeframe_results": (
                [r.to_dict() for r in self.timeframe_results] if self.timeframe_results else None
            ),
        }


class PerformanceMonitor:
    """
    Sophisticated performance monitoring with degradation detection

    This class implements multi-timeframe performance analysis, statistical
    significance testing, and regime-aware performance evaluation to make
    intelligent strategy switching decisions.
    """

    def __init__(self, config: PerformanceDegradationConfig | None = None):
        """
        Initialize performance monitor

        Args:
            config: Configuration for degradation detection
        """
        self.config = config or PerformanceDegradationConfig()
        self.logger = logging.getLogger("PerformanceMonitor")

        # Historical performance baselines
        self.performance_baselines: dict[str, dict[TimeFrame, PerformanceMetrics]] = defaultdict(
            dict
        )

        # Regime-specific performance tracking
        self.regime_performance_history: dict[str, dict[str, list[PerformanceMetrics]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        # Performance degradation history
        self.degradation_history: list[tuple[datetime, str, DegradationSeverity]] = []

        self.logger.info("PerformanceMonitor initialized")

    def should_switch_strategy(
        self,
        current_strategy_id: str,
        performance_tracker: PerformanceTracker,
        market_data: pd.DataFrame,
        current_regime: RegimeContext | None = None,
    ) -> SwitchDecision:
        """
        Determine if strategy should be switched based on performance degradation

        Args:
            current_strategy_id: ID of currently active strategy
            performance_tracker: Performance tracker for current strategy
            market_data: Recent market data for regime analysis
            current_regime: Current market regime context

        Returns:
            Switch decision with detailed analysis
        """
        self.logger.info(f"Evaluating strategy switch for {current_strategy_id}")

        # 1. Check minimum requirements
        if not self._meets_minimum_requirements(performance_tracker):
            return SwitchDecision(
                should_switch=False, reason="Insufficient data for evaluation", confidence=0.0
            )

        # 2. Multi-timeframe analysis
        timeframe_results = self._analyze_multiple_timeframes(
            current_strategy_id, performance_tracker, current_regime
        )

        # 3. Check if underperforming across all timeframes
        underperforming_timeframes = [r for r in timeframe_results if r.underperforming]

        if len(underperforming_timeframes) < len(timeframe_results):
            return SwitchDecision(
                should_switch=False,
                reason="Not underperforming across all timeframes",
                confidence=0.3,
                timeframe_results=timeframe_results,
            )

        # 4. Statistical significance test
        if not self._is_statistically_significant(timeframe_results):
            return SwitchDecision(
                should_switch=False,
                reason="Underperformance not statistically significant",
                confidence=0.4,
                timeframe_results=timeframe_results,
            )

        # 5. Regime context analysis
        if self._is_regime_transition_period(market_data, current_regime):
            return SwitchDecision(
                should_switch=False,
                reason="In regime transition period",
                confidence=0.2,
                timeframe_results=timeframe_results,
            )

        # 6. Determine degradation severity
        severity = self._calculate_degradation_severity(timeframe_results)

        # 7. Calculate switch confidence
        switch_confidence = self._calculate_switch_confidence(timeframe_results, severity)

        # Record degradation
        self.degradation_history.append((datetime.now(UTC), current_strategy_id, severity))

        return SwitchDecision(
            should_switch=True,
            reason=f"Multi-criteria performance degradation detected ({severity.value})",
            confidence=switch_confidence,
            degradation_severity=severity,
            timeframe_results=timeframe_results,
        )

    def update_performance_baseline(
        self,
        strategy_id: str,
        performance_tracker: PerformanceTracker,
        regime: str | None = None,
    ) -> None:
        """
        Update performance baseline for a strategy

        Args:
            strategy_id: Strategy identifier
            performance_tracker: Performance tracker with historical data
            regime: Optional regime context for regime-specific baselines
        """
        # Update baselines for each timeframe
        for timeframe in TimeFrame:
            days = self._get_timeframe_days(timeframe)
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days)

            # Get performance metrics for this timeframe
            metrics = performance_tracker.get_performance_metrics(
                start_date=start_date, end_date=end_date
            )

            if metrics.total_trades >= self.config.min_trades_for_evaluation // 3:
                self.performance_baselines[strategy_id][timeframe] = metrics

                # Update regime-specific baseline if regime provided
                if regime:
                    self.regime_performance_history[strategy_id][regime].append(metrics)
                    # Keep only recent regime performance (last 10 periods)
                    if len(self.regime_performance_history[strategy_id][regime]) > 10:
                        self.regime_performance_history[strategy_id][regime] = (
                            self.regime_performance_history[strategy_id][regime][-10:]
                        )

        self.logger.debug(f"Updated performance baseline for {strategy_id}")

    def get_regime_performance_confidence(
        self, strategy_id: str, current_regime: str, performance_tracker: PerformanceTracker
    ) -> float:
        """
        Get confidence in strategy performance for current regime

        Args:
            strategy_id: Strategy identifier
            current_regime: Current market regime
            performance_tracker: Performance tracker

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if strategy_id not in self.regime_performance_history:
            return 0.5  # Neutral confidence for unknown strategy

        regime_history = self.regime_performance_history[strategy_id].get(current_regime, [])

        if not regime_history:
            return 0.3  # Low confidence for unknown regime

        # Calculate confidence based on historical regime performance
        recent_performance = regime_history[-3:]  # Last 3 periods

        if len(recent_performance) < 2:
            return 0.4  # Low confidence with insufficient data

        # Calculate average performance metrics
        avg_sharpe = statistics.mean([p.sharpe_ratio for p in recent_performance])
        avg_win_rate = statistics.mean([p.win_rate for p in recent_performance])
        avg_drawdown = statistics.mean([p.max_drawdown for p in recent_performance])

        # Score based on performance thresholds
        sharpe_score = min(1.0, max(0.0, avg_sharpe / 2.0))  # Normalize to 0-1
        win_rate_score = min(1.0, max(0.0, (avg_win_rate - 0.3) / 0.4))  # 30-70% range
        drawdown_score = min(1.0, max(0.0, 1.0 - (avg_drawdown / 0.3)))  # Penalize high drawdown

        # Weighted combination
        confidence = sharpe_score * 0.4 + win_rate_score * 0.3 + drawdown_score * 0.3

        return confidence

    def _meets_minimum_requirements(self, performance_tracker: PerformanceTracker) -> bool:
        """Check if strategy meets minimum requirements for evaluation"""
        # Check trade count
        if performance_tracker.trade_count < self.config.min_trades_for_evaluation:
            self.logger.debug(
                f"Insufficient trades: {performance_tracker.trade_count} < {self.config.min_trades_for_evaluation}"
            )
            return False

        # Check time period
        if not performance_tracker.trades:
            return False

        oldest_trade = min(performance_tracker.trades, key=lambda t: t.timestamp)
        days_active = (datetime.now(UTC) - oldest_trade.timestamp).days

        if days_active < self.config.min_days_for_evaluation:
            self.logger.debug(
                f"Insufficient time period: {days_active} < {self.config.min_days_for_evaluation}"
            )
            return False

        return True

    def _analyze_multiple_timeframes(
        self,
        strategy_id: str,
        performance_tracker: PerformanceTracker,
        current_regime: RegimeContext | None,
    ) -> list[TimeFrameAnalysis]:
        """Analyze performance across multiple timeframes"""
        results = []

        for timeframe in TimeFrame:
            days = self._get_timeframe_days(timeframe)
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days)

            # Get current performance for this timeframe
            current_metrics = performance_tracker.get_performance_metrics(
                start_date=start_date, end_date=end_date
            )

            # Get historical baseline
            historical_metrics = self.performance_baselines.get(strategy_id, {}).get(timeframe)

            if not historical_metrics:
                # Use overall performance as baseline if no specific baseline exists
                historical_metrics = performance_tracker.get_performance_metrics()

            # Determine if underperforming
            underperforming = self._is_underperforming(current_metrics, historical_metrics)

            # Calculate statistical significance
            significance = self._calculate_statistical_significance(
                current_metrics, historical_metrics, performance_tracker, days
            )

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                current_metrics, performance_tracker, days
            )

            # Determine degradation severity for this timeframe
            severity = self._determine_timeframe_severity(current_metrics, historical_metrics)

            analysis = TimeFrameAnalysis(
                timeframe=timeframe,
                period_days=days,
                trade_count=current_metrics.total_trades,
                current_performance=current_metrics,
                historical_performance=historical_metrics,
                underperforming=underperforming,
                statistical_significance=significance,
                confidence_interval=confidence_interval,
                degradation_severity=severity,
            )

            results.append(analysis)

        return results

    def _is_underperforming(
        self, current: PerformanceMetrics, baseline: PerformanceMetrics
    ) -> bool:
        """Determine if current performance is worse than baseline"""
        # Multiple criteria must be met for underperformance
        criteria_failed = 0

        # Sharpe ratio degradation
        if current.sharpe_ratio < baseline.sharpe_ratio * 0.8:  # 20% degradation
            criteria_failed += 1

        # Win rate degradation
        if current.win_rate < baseline.win_rate * 0.9:  # 10% degradation
            criteria_failed += 1

        # Drawdown increase
        if current.max_drawdown > baseline.max_drawdown * 1.5:  # 50% increase
            criteria_failed += 1

        # Return degradation
        if current.total_return_pct < baseline.total_return_pct * 0.7:  # 30% degradation
            criteria_failed += 1

        # Require at least 2 criteria to be failed
        return criteria_failed >= 2

    def _calculate_statistical_significance(
        self,
        current: PerformanceMetrics,
        baseline: PerformanceMetrics,
        performance_tracker: PerformanceTracker,
        days: int,
    ) -> float:
        """Calculate statistical significance of performance difference"""
        # Get recent trades for the timeframe
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        recent_trades = [t for t in performance_tracker.trades if t.timestamp >= cutoff_date]

        if len(recent_trades) < 10:
            return 0.0  # Insufficient data

        # Get returns for statistical test
        recent_returns = [t.pnl_percent for t in recent_trades]

        # Calculate baseline mean from historical individual trade returns
        # Get historical trades excluding the recent period
        historical_end = cutoff_date
        historical_start = historical_end - timedelta(days=days * 2)  # Use 2x period for baseline
        historical_trades = [
            t
            for t in performance_tracker.trades
            if historical_start <= t.timestamp < historical_end
        ]

        if len(historical_trades) < 10:
            # Fallback: use all historical trades if insufficient data
            historical_trades = [t for t in performance_tracker.trades if t.timestamp < cutoff_date]

        if len(historical_trades) < 10:
            return 0.0  # Insufficient baseline data

        # Calculate mean from individual trade returns (not aggregated metrics)
        baseline_mean = statistics.mean([t.pnl_percent for t in historical_trades])

        try:
            t_stat, p_value = stats.ttest_1samp(recent_returns, baseline_mean)

            # Return significance level (1 - p_value for underperformance)
            if t_stat < 0:  # Underperforming
                return 1.0 - p_value
            else:
                return 0.0  # Not underperforming

        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return 0.0

    def _calculate_confidence_interval(
        self, metrics: PerformanceMetrics, performance_tracker: PerformanceTracker, days: int
    ) -> tuple[float, float]:
        """Calculate confidence interval for performance metrics"""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        recent_trades = [t for t in performance_tracker.trades if t.timestamp >= cutoff_date]

        if len(recent_trades) < 5:
            return (0.0, 0.0)

        returns = [t.pnl_percent for t in recent_trades]

        try:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0

            # 95% confidence interval
            margin_of_error = 1.96 * (std_return / np.sqrt(len(returns)))

            return (mean_return - margin_of_error, mean_return + margin_of_error)

        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)

    def _determine_timeframe_severity(
        self, current: PerformanceMetrics, baseline: PerformanceMetrics
    ) -> DegradationSeverity:
        """Determine degradation severity for a timeframe"""
        # Calculate degradation scores
        sharpe_degradation = max(
            0, (baseline.sharpe_ratio - current.sharpe_ratio) / max(0.1, baseline.sharpe_ratio)
        )
        drawdown_increase = max(
            0, (current.max_drawdown - baseline.max_drawdown) / max(0.01, baseline.max_drawdown)
        )
        return_degradation = max(
            0,
            (baseline.total_return_pct - current.total_return_pct)
            / max(0.01, abs(baseline.total_return_pct)),
        )

        # Weighted severity score
        severity_score = (
            sharpe_degradation * 0.4 + drawdown_increase * 0.3 + return_degradation * 0.3
        )

        if severity_score < 0.1:
            return DegradationSeverity.NONE
        elif severity_score < 0.3:
            return DegradationSeverity.MINOR
        elif severity_score < 0.6:
            return DegradationSeverity.MODERATE
        elif severity_score < 1.0:
            return DegradationSeverity.SEVERE
        else:
            return DegradationSeverity.CRITICAL

    def _is_statistically_significant(self, timeframe_results: list[TimeFrameAnalysis]) -> bool:
        """Check if underperformance is statistically significant"""
        significant_timeframes = [
            r
            for r in timeframe_results
            if r.statistical_significance >= self.config.confidence_level
        ]

        # Require at least 2 out of 3 timeframes to be statistically significant
        return len(significant_timeframes) >= 2

    def _is_regime_transition_period(
        self, market_data: pd.DataFrame, current_regime: RegimeContext | None
    ) -> bool:
        """Check if we're in a regime transition period"""
        if not current_regime or market_data.empty:
            return False

        # Check regime stability based on confidence and duration
        if current_regime.confidence < 0.7:
            return True

        if current_regime.duration < self.config.min_regime_duration_days:
            return True

        return False

    def _calculate_degradation_severity(
        self, timeframe_results: list[TimeFrameAnalysis]
    ) -> DegradationSeverity:
        """Calculate overall degradation severity across timeframes"""
        severities = [r.degradation_severity for r in timeframe_results]

        # Count severity levels
        severity_counts = {severity: severities.count(severity) for severity in DegradationSeverity}

        # Determine overall severity based on worst case and frequency
        if severity_counts.get(DegradationSeverity.CRITICAL, 0) >= 1:
            return DegradationSeverity.CRITICAL
        elif severity_counts.get(DegradationSeverity.SEVERE, 0) >= 2:
            return DegradationSeverity.SEVERE
        elif severity_counts.get(DegradationSeverity.MODERATE, 0) >= 2:
            return DegradationSeverity.MODERATE
        elif severity_counts.get(DegradationSeverity.MINOR, 0) >= 2:
            return DegradationSeverity.MINOR
        else:
            return DegradationSeverity.NONE

    def _calculate_switch_confidence(
        self, timeframe_results: list[TimeFrameAnalysis], severity: DegradationSeverity
    ) -> float:
        """Calculate confidence in switch decision"""
        base_confidence = 0.5

        # Adjust for statistical significance
        avg_significance = statistics.mean([r.statistical_significance for r in timeframe_results])
        significance_boost = min(0.3, avg_significance)

        # Adjust for severity
        severity_multipliers = {
            DegradationSeverity.NONE: 0.0,
            DegradationSeverity.MINOR: 0.1,
            DegradationSeverity.MODERATE: 0.2,
            DegradationSeverity.SEVERE: 0.3,
            DegradationSeverity.CRITICAL: 0.4,
        }
        severity_boost = severity_multipliers.get(severity, 0.0)

        # Adjust for consistency across timeframes
        underperforming_count = sum(1 for r in timeframe_results if r.underperforming)
        consistency_boost = (underperforming_count / len(timeframe_results)) * 0.2

        final_confidence = base_confidence + significance_boost + severity_boost + consistency_boost

        return min(1.0, max(0.0, final_confidence))

    def _get_timeframe_days(self, timeframe: TimeFrame) -> int:
        """Get number of days for a timeframe"""
        if timeframe == TimeFrame.SHORT_TERM:
            return self.config.short_term_days
        elif timeframe == TimeFrame.MEDIUM_TERM:
            return self.config.medium_term_days
        elif timeframe == TimeFrame.LONG_TERM:
            return self.config.long_term_days
        else:
            return 30  # Default

    def get_degradation_history(
        self, strategy_id: str | None = None, days: int = 30
    ) -> list[dict[str, Any]]:
        """Get degradation history for analysis"""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        filtered_history = [
            {"timestamp": timestamp.isoformat(), "strategy_id": sid, "severity": severity.value}
            for timestamp, sid, severity in self.degradation_history
            if timestamp >= cutoff_date and (strategy_id is None or sid == strategy_id)
        ]

        return filtered_history

    def reset_baselines(self, strategy_id: str | None = None) -> None:
        """Reset performance baselines"""
        if strategy_id:
            if strategy_id in self.performance_baselines:
                del self.performance_baselines[strategy_id]
            if strategy_id in self.regime_performance_history:
                del self.regime_performance_history[strategy_id]
        else:
            self.performance_baselines.clear()
            self.regime_performance_history.clear()

        self.logger.info(f"Reset performance baselines for {strategy_id or 'all strategies'}")
