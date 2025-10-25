"""
Performance Tracking System

This module implements comprehensive performance tracking for strategies,
including real-time metrics, historical data storage, comparison utilities,
and visualization/reporting capabilities.
"""

import logging
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np


class PerformancePeriod(Enum):
    """Performance measurement periods"""

    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


class MetricType(Enum):
    """Types of performance metrics"""

    RETURN = "return"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    DRAWDOWN = "drawdown"
    TRADE_STATS = "trade_stats"
    REGIME_SPECIFIC = "regime_specific"


@dataclass
class TradeResult:
    """Individual trade result for performance tracking"""

    timestamp: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    duration_hours: float
    strategy_id: str
    confidence: float
    regime: Optional[str] = None
    exit_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeResult":
        """Create from dictionary"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""

    # Return metrics
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Efficiency metrics
    avg_trade_duration: float
    trades_per_day: float
    hit_rate: float

    # Drawdown analysis
    max_drawdown_duration: float
    current_drawdown: float
    drawdown_recovery_time: float

    # Additional metrics
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int

    # Time period
    period_start: datetime
    period_end: datetime
    period_type: PerformancePeriod

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["period_start"] = self.period_start.isoformat()
        data["period_end"] = self.period_end.isoformat()
        data["period_type"] = self.period_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary"""
        data = data.copy()
        data["period_start"] = datetime.fromisoformat(data["period_start"])
        data["period_end"] = datetime.fromisoformat(data["period_end"])
        data["period_type"] = PerformancePeriod(data["period_type"])
        return cls(**data)


@dataclass
class RegimePerformance:
    """Performance metrics specific to market regimes"""

    regime_type: str
    trade_count: int
    win_rate: float
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class PerformanceTracker:
    """
    Real-time strategy performance tracking system

    This class tracks strategy performance in real-time, maintains historical
    data, and provides comprehensive analysis and comparison capabilities.
    """

    def __init__(
        self, strategy_id: str, max_history: int = 10000, storage_backend: Optional[Any] = None
    ):
        """
        Initialize performance tracker

        Args:
            strategy_id: Strategy identifier
            max_history: Maximum number of trades to keep in memory
            storage_backend: Optional storage backend for persistence
        """
        self.strategy_id = strategy_id
        self.max_history = max_history
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(f"PerformanceTracker.{strategy_id}")

        # Trade history
        self.trades: deque[TradeResult] = deque(maxlen=max_history)
        self.trade_count = 0

        # Real-time metrics
        self.current_balance = 0.0
        self.initial_balance = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Performance cache
        self._metrics_cache: dict[str, PerformanceMetrics] = {}
        self._cache_expiry: dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=5)  # Cache for 5 minutes

        # Regime-specific tracking
        self.regime_performance: dict[str, list[TradeResult]] = defaultdict(list)

        # Running statistics
        self.running_stats = {
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "current_streak_type": None,  # 'win' or 'loss'
            "current_streak_count": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

        self.logger.info(f"PerformanceTracker initialized for strategy {strategy_id}")

    def record_trade(self, trade: TradeResult) -> None:
        """
        Record a completed trade

        Args:
            trade: Trade result to record
        """
        # Add to trade history
        self.trades.append(trade)
        self.trade_count += 1

        # Update balance tracking
        self.current_balance += trade.pnl
        if self.initial_balance == 0.0:
            self.initial_balance = self.current_balance - trade.pnl

        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0.0
        else:
            if self.peak_balance > 0:
                self.current_drawdown = (
                    self.peak_balance - self.current_balance
                ) / self.peak_balance
            else:
                self.current_drawdown = 0.0
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Update running statistics
        self._update_running_stats(trade)

        # Add to regime-specific tracking
        if trade.regime:
            self.regime_performance[trade.regime].append(trade)

        # Clear metrics cache
        self._clear_metrics_cache()

        # Persist if backend available
        if self.storage_backend:
            try:
                self.storage_backend.save_trade(trade)
            except Exception as e:
                self.logger.error(f"Failed to persist trade: {e}")

        self.logger.debug(
            f"Recorded trade: PnL={trade.pnl:.2f}, Balance={self.current_balance:.2f}"
        )

    def get_performance_metrics(
        self,
        period: PerformancePeriod = PerformancePeriod.ALL_TIME,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """
        Get performance metrics for specified period

        Args:
            period: Performance period type
            start_date: Optional start date for custom period
            end_date: Optional end date for custom period

        Returns:
            Performance metrics
        """
        # Check cache first
        cache_key = f"{period.value}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            return self._metrics_cache[cache_key]

        # Filter trades by period
        filtered_trades = self._filter_trades_by_period(period, start_date, end_date)

        if not filtered_trades:
            # Return empty metrics
            now = datetime.now()
            return PerformanceMetrics(
                total_return=0.0,
                total_return_pct=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                avg_trade_duration=0.0,
                trades_per_day=0.0,
                hit_rate=0.0,
                max_drawdown_duration=0.0,
                current_drawdown=0.0,
                drawdown_recovery_time=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                period_start=now,
                period_end=now,
                period_type=period,
            )

        # Calculate metrics
        metrics = self._calculate_metrics(filtered_trades, period)

        # Cache results
        self._metrics_cache[cache_key] = metrics
        self._cache_expiry[cache_key] = datetime.now() + self._cache_duration

        return metrics

    def get_regime_performance(self, regime: Optional[str] = None) -> dict[str, RegimePerformance]:
        """
        Get performance metrics by market regime

        Args:
            regime: Optional specific regime to analyze

        Returns:
            Dictionary of regime performance metrics
        """
        regime_metrics = {}

        regimes_to_analyze = [regime] if regime else list(self.regime_performance.keys())

        for regime_type in regimes_to_analyze:
            if regime_type not in self.regime_performance:
                continue

            trades = self.regime_performance[regime_type]
            if not trades:
                continue

            # Calculate regime-specific metrics
            pnl_values = [t.pnl_percent for t in trades]
            winning_trades = [t for t in trades if t.pnl > 0]

            regime_metrics[regime_type] = RegimePerformance(
                regime_type=regime_type,
                trade_count=len(trades),
                win_rate=len(winning_trades) / len(trades) if trades else 0.0,
                avg_return=statistics.mean(pnl_values) if pnl_values else 0.0,
                volatility=statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0.0,
                sharpe_ratio=self._calculate_sharpe_ratio(pnl_values),
                max_drawdown=self._calculate_max_drawdown([t.pnl for t in trades]),
            )

        return regime_metrics

    def compare_performance(
        self,
        other_tracker: "PerformanceTracker",
        period: PerformancePeriod = PerformancePeriod.ALL_TIME,
    ) -> dict[str, Any]:
        """
        Compare performance with another tracker

        Args:
            other_tracker: Another performance tracker to compare with
            period: Period for comparison

        Returns:
            Comparison results dictionary
        """
        self_metrics = self.get_performance_metrics(period)
        other_metrics = other_tracker.get_performance_metrics(period)

        comparison = {
            "strategy_1": {"id": self.strategy_id, "metrics": self_metrics.to_dict()},
            "strategy_2": {"id": other_tracker.strategy_id, "metrics": other_metrics.to_dict()},
            "comparison": {
                "return_difference": self_metrics.total_return_pct - other_metrics.total_return_pct,
                "sharpe_difference": self_metrics.sharpe_ratio - other_metrics.sharpe_ratio,
                "drawdown_difference": self_metrics.max_drawdown - other_metrics.max_drawdown,
                "win_rate_difference": self_metrics.win_rate - other_metrics.win_rate,
                "trade_count_difference": self_metrics.total_trades - other_metrics.total_trades,
                "volatility_difference": self_metrics.volatility - other_metrics.volatility,
            },
            "winner": self._determine_winner(self_metrics, other_metrics),
            "comparison_date": datetime.now().isoformat(),
        }

        return comparison

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get comprehensive performance summary

        Returns:
            Performance summary dictionary
        """
        current_metrics = self.get_performance_metrics(PerformancePeriod.ALL_TIME)
        daily_metrics = self.get_performance_metrics(PerformancePeriod.DAILY)
        regime_performance = self.get_regime_performance()

        return {
            "strategy_id": self.strategy_id,
            "current_metrics": current_metrics.to_dict(),
            "daily_metrics": daily_metrics.to_dict(),
            "regime_performance": {k: v.to_dict() for k, v in regime_performance.items()},
            "running_stats": self.running_stats.copy(),
            "balance_info": {
                "current_balance": self.current_balance,
                "initial_balance": self.initial_balance,
                "peak_balance": self.peak_balance,
                "current_drawdown": self.current_drawdown,
                "max_drawdown": self.max_drawdown,
            },
            "trade_count": self.trade_count,
            "last_updated": datetime.now().isoformat(),
        }

    def get_trade_history(
        self,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[TradeResult]:
        """
        Get trade history with optional filtering

        Args:
            limit: Maximum number of trades to return
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of trade results
        """
        trades = list(self.trades)

        # Apply date filters
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]

        # Sort by timestamp (most recent first)
        trades.sort(key=lambda t: t.timestamp, reverse=True)

        # Apply limit
        if limit:
            trades = trades[:limit]

        return trades

    def reset_performance(self) -> None:
        """Reset all performance tracking data"""
        self.trades.clear()
        self.trade_count = 0
        self.current_balance = 0.0
        self.initial_balance = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.regime_performance.clear()
        self.running_stats = {
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "current_streak_type": None,
            "current_streak_count": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
        self._clear_metrics_cache()
        self.logger.info("Performance tracking data reset")

    def _update_running_stats(self, trade: TradeResult) -> None:
        """Update running statistics with new trade"""
        # Update totals
        self.running_stats["total_pnl"] += trade.pnl
        self.running_stats["total_pnl_pct"] += trade.pnl_percent

        # Update win/loss counts
        if trade.pnl > 0:
            self.running_stats["winning_trades"] += 1

            # Update streaks
            if self.running_stats["current_streak_type"] == "win":
                self.running_stats["current_streak_count"] += 1
                self.running_stats["consecutive_wins"] = self.running_stats["current_streak_count"]
            else:
                self.running_stats["current_streak_type"] = "win"
                self.running_stats["current_streak_count"] = 1
                self.running_stats["consecutive_wins"] = 1

            self.running_stats["max_win_streak"] = max(
                self.running_stats["max_win_streak"], self.running_stats["consecutive_wins"]
            )

        else:
            self.running_stats["losing_trades"] += 1

            # Update streaks
            if self.running_stats["current_streak_type"] == "loss":
                self.running_stats["current_streak_count"] += 1
                self.running_stats["consecutive_losses"] = self.running_stats[
                    "current_streak_count"
                ]
            else:
                self.running_stats["current_streak_type"] = "loss"
                self.running_stats["current_streak_count"] = 1
                self.running_stats["consecutive_losses"] = 1

            self.running_stats["max_loss_streak"] = max(
                self.running_stats["max_loss_streak"], self.running_stats["consecutive_losses"]
            )

        # Update best/worst trades
        self.running_stats["best_trade"] = max(self.running_stats["best_trade"], trade.pnl_percent)
        self.running_stats["worst_trade"] = min(
            self.running_stats["worst_trade"], trade.pnl_percent
        )

    def _filter_trades_by_period(
        self,
        period: PerformancePeriod,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[TradeResult]:
        """Filter trades by time period"""
        trades = list(self.trades)

        if period == PerformancePeriod.ALL_TIME:
            if start_date or end_date:
                # Custom period
                if start_date:
                    trades = [t for t in trades if t.timestamp >= start_date]
                if end_date:
                    trades = [t for t in trades if t.timestamp <= end_date]
            return trades

        # Calculate period boundaries
        now = datetime.now()

        if period == PerformancePeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.WEEKLY:
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif period == PerformancePeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.QUARTERLY:
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(
                month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0
            )
        elif period == PerformancePeriod.YEARLY:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return trades

        return [t for t in trades if t.timestamp >= start]

    def _calculate_metrics(
        self, trades: list[TradeResult], period: PerformancePeriod
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not trades:
            now = datetime.now()
            return PerformanceMetrics(
                total_return=0.0,
                total_return_pct=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                avg_trade_duration=0.0,
                trades_per_day=0.0,
                hit_rate=0.0,
                max_drawdown_duration=0.0,
                current_drawdown=0.0,
                drawdown_recovery_time=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                period_start=now,
                period_end=now,
                period_type=period,
            )

        # Basic statistics
        pnl_values = [t.pnl for t in trades]
        pnl_pct_values = [t.pnl_percent for t in trades]
        durations = [t.duration_hours for t in trades]

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        # Return metrics
        total_return = sum(pnl_values)
        total_return_pct = sum(pnl_pct_values)

        # Calculate annualized return
        period_days = (trades[-1].timestamp - trades[0].timestamp).days or 1
        annualized_return = (total_return_pct / period_days) * 365 if period_days > 0 else 0.0

        # Risk metrics
        volatility = statistics.stdev(pnl_pct_values) if len(pnl_pct_values) > 1 else 0.0
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_pct_values)
        sortino_ratio = self._calculate_sortino_ratio(pnl_pct_values)
        max_drawdown = self._calculate_max_drawdown(pnl_values)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        var_95 = np.percentile(pnl_pct_values, 5) if pnl_pct_values else 0.0

        # Trade statistics
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        avg_win = (
            statistics.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0.0
        )
        avg_loss = statistics.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0.0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf") if gross_profit > 0 else 0.0
        )

        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Efficiency metrics
        avg_trade_duration = statistics.mean(durations) if durations else 0.0
        trades_per_day = len(trades) / period_days if period_days > 0 else 0.0
        hit_rate = win_rate  # Same as win rate for now

        # Streak analysis
        consecutive_wins, consecutive_losses = self._calculate_streaks(trades)

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_trade_duration=avg_trade_duration,
            trades_per_day=trades_per_day,
            hit_rate=hit_rate,
            max_drawdown_duration=0.0,  # Would need more complex calculation
            current_drawdown=self.current_drawdown,
            drawdown_recovery_time=0.0,  # Would need more complex calculation
            best_trade=max(pnl_pct_values) if pnl_pct_values else 0.0,
            worst_trade=min(pnl_pct_values) if pnl_pct_values else 0.0,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            period_start=trades[0].timestamp,
            period_end=trades[-1].timestamp,
            period_type=period,
        )

    def _calculate_sharpe_ratio(self, returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        # Annualize the Sharpe ratio
        return (mean_return - risk_free_rate / 365) / std_return * np.sqrt(365)

    def _calculate_sortino_ratio(self, returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns:
            return 0.0

        mean_return = statistics.mean(returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float("inf") if mean_return > risk_free_rate / 365 else 0.0

        if len(downside_returns) == 1:
            return float("inf") if mean_return > risk_free_rate / 365 else 0.0

        downside_deviation = statistics.stdev(downside_returns)

        if downside_deviation == 0:
            return 0.0

        return (mean_return - risk_free_rate / 365) / downside_deviation * np.sqrt(365)

    def _calculate_max_drawdown(self, pnl_values: list[float]) -> float:
        """Calculate maximum drawdown"""
        if not pnl_values:
            return 0.0

        cumulative = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / np.maximum(running_max, 1)  # Avoid division by zero

        return float(np.max(drawdown))

    def _calculate_streaks(self, trades: list[TradeResult]) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _determine_winner(self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics) -> str:
        """Determine which strategy performed better"""
        score1 = 0
        score2 = 0

        # Compare key metrics
        if metrics1.total_return_pct > metrics2.total_return_pct:
            score1 += 1
        else:
            score2 += 1

        if metrics1.sharpe_ratio > metrics2.sharpe_ratio:
            score1 += 1
        else:
            score2 += 1

        if metrics1.max_drawdown < metrics2.max_drawdown:
            score1 += 1
        else:
            score2 += 1

        if metrics1.win_rate > metrics2.win_rate:
            score1 += 1
        else:
            score2 += 1

        if score1 > score2:
            return "strategy_1"
        elif score2 > score1:
            return "strategy_2"
        else:
            return "tie"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached metrics are still valid"""
        if cache_key not in self._metrics_cache:
            return False

        if cache_key not in self._cache_expiry:
            return False

        return datetime.now() < self._cache_expiry[cache_key]

    def _clear_metrics_cache(self) -> None:
        """Clear all cached metrics"""
        self._metrics_cache.clear()
        self._cache_expiry.clear()
