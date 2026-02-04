import logging
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd

from src.config.constants import (
    DEFAULT_DRAWDOWN_STOP_TIGHTENING_INCREMENT,
    DEFAULT_DRAWDOWN_THRESHOLDS,
    DEFAULT_EPSILON,
    DEFAULT_CORRELATED_EXPOSURE_MIN_FACTOR,
    DEFAULT_GOOD_PERF_DAILY_RISK_FACTOR,
    DEFAULT_GOOD_PERF_POSITION_FACTOR,
    DEFAULT_GOOD_PERF_STOP_TIGHTENING,
    DEFAULT_HIGH_VOL_STOP_TIGHTENING,
    DEFAULT_HIGH_VOLATILITY_THRESHOLD,
    DEFAULT_LARGE_SINGLE_POSITION_SIZE_FACTOR,
    DEFAULT_LARGE_SINGLE_POSITION_STOP_MULTIPLIER,
    DEFAULT_LARGE_SINGLE_POSITION_THRESHOLD,
    DEFAULT_LOW_DIVERSIFICATION_EXPOSURE_THRESHOLD,
    DEFAULT_LOW_DIVERSIFICATION_POSITION_COUNT,
    DEFAULT_LOW_DIVERSIFICATION_SIZE_FACTOR,
    DEFAULT_LOW_VOL_STOP_TIGHTENING,
    DEFAULT_LOW_VOLATILITY_THRESHOLD,
    DEFAULT_MAX_CORRELATED_POSITIONS_FOR_CAP,
    DEFAULT_MIN_TRADES_FOR_DYNAMIC_ADJUSTMENT,
    DEFAULT_PERFORMANCE_GOOD_THRESHOLD,
    DEFAULT_PERFORMANCE_POOR_THRESHOLD,
    DEFAULT_PERFORMANCE_PROFIT_FACTOR_DIVISOR,
    DEFAULT_PERFORMANCE_PROFIT_FACTOR_WEIGHT,
    DEFAULT_PERFORMANCE_WIN_RATE_WEIGHT,
    DEFAULT_PERFORMANCE_WINDOW_DAYS,
    DEFAULT_POOR_PERF_DAILY_RISK_FACTOR,
    DEFAULT_POOR_PERF_POSITION_FACTOR,
    DEFAULT_POOR_PERF_STOP_TIGHTENING,
    DEFAULT_PROFIT_FACTOR_FALLBACK,
    DEFAULT_RECOVERY_SCALING_FACTOR,
    DEFAULT_RECOVERY_THRESHOLDS,
    DEFAULT_RISK_REDUCTION_FACTORS,
    DEFAULT_VOLATILITY_ADJUSTMENT_ENABLED,
    DEFAULT_VOLATILITY_FALLBACK,
    DEFAULT_VOLATILITY_RISK_MULTIPLIERS,
    DEFAULT_VOLATILITY_WINDOW_DAYS,
    DEFAULT_WIN_RATE_FALLBACK,
)

if TYPE_CHECKING:
    from database.manager import DatabaseManager

from src.risk.risk_manager import RiskParameters

logger = logging.getLogger(__name__)


@dataclass
class DynamicRiskConfig:
    """Configuration for dynamic risk management"""

    # Core settings
    enabled: bool = True
    performance_window_days: int = DEFAULT_PERFORMANCE_WINDOW_DAYS

    # Drawdown thresholds and adjustments (use centralized constants)
    drawdown_thresholds: list[float] = None
    risk_reduction_factors: list[float] = None

    # Recovery thresholds
    recovery_thresholds: list[float] = None

    # Volatility adjustments (use centralized constants)
    volatility_adjustment_enabled: bool = DEFAULT_VOLATILITY_ADJUSTMENT_ENABLED
    volatility_window_days: int = DEFAULT_VOLATILITY_WINDOW_DAYS
    high_volatility_threshold: float = DEFAULT_HIGH_VOLATILITY_THRESHOLD
    low_volatility_threshold: float = DEFAULT_LOW_VOLATILITY_THRESHOLD
    volatility_risk_multipliers: tuple[float, float] = DEFAULT_VOLATILITY_RISK_MULTIPLIERS

    def __post_init__(self):
        """Set defaults and validate configuration"""
        if self.drawdown_thresholds is None:
            self.drawdown_thresholds = list(DEFAULT_DRAWDOWN_THRESHOLDS)
        if self.risk_reduction_factors is None:
            self.risk_reduction_factors = list(DEFAULT_RISK_REDUCTION_FACTORS)
        if self.recovery_thresholds is None:
            self.recovery_thresholds = list(DEFAULT_RECOVERY_THRESHOLDS)

        # Validation
        if len(self.drawdown_thresholds) != len(self.risk_reduction_factors):
            raise ValueError("drawdown_thresholds and risk_reduction_factors must have same length")

        if not all(0 < factor <= 1 for factor in self.risk_reduction_factors):
            raise ValueError("risk_reduction_factors must be between 0 and 1")

        if not all(threshold > 0 for threshold in self.drawdown_thresholds):
            raise ValueError("drawdown_thresholds must be positive")


@dataclass
class RiskAdjustments:
    """Container for calculated risk adjustments."""

    position_size_factor: float = 1.0  # Multiplier for position sizing
    # Multiplier for stop-loss distance (ATR). >1 widens stops, <1 tightens.
    stop_loss_tightening: float = 1.0
    daily_risk_factor: float = 1.0  # Multiplier for daily risk limits

    # Metadata
    primary_reason: str = "normal"
    adjustment_details: dict[str, Any] = None

    def __post_init__(self):
        if self.adjustment_details is None:
            self.adjustment_details = {}


class DynamicRiskManager:
    """
    Manages dynamic risk adjustments based on performance and market conditions.

    This class works alongside the existing RiskManager to provide adaptive
    risk management that responds to changing performance and market conditions.
    """

    def __init__(
        self,
        config: DynamicRiskConfig | None = None,
        db_manager: Optional["DatabaseManager"] = None,
        max_correlated_risk: Optional[float] = None,
        risk_parameters: RiskParameters | None = None,
        positions_provider: Callable[[], dict[str, dict[str, Any]]] | None = None,
    ):
        self.config = config or DynamicRiskConfig()
        self.db_manager = db_manager
        self._positions_provider = positions_provider
        if max_correlated_risk is not None:
            self.max_correlated_risk = max_correlated_risk
        elif risk_parameters is not None:
            self.max_correlated_risk = risk_parameters.max_correlated_risk
        else:
            self.max_correlated_risk = 0.10  # 10% default

        # Cache for performance calculations with thread safety
        self._performance_cache: dict[str, Any] = {}
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minutes
        self._cache_lock = threading.Lock()  # Prevents race conditions in cache access
        self._computing = False  # Flag to prevent duplicate calculations

    def calculate_dynamic_risk_adjustments(
        self,
        current_balance: float,
        peak_balance: float,
        session_id: int | None = None,
        previous_peak_balance: float | None = None,
    ) -> RiskAdjustments:
        """
        Calculate dynamic risk adjustments based on current performance metrics.

        Args:
            current_balance: Current account balance
            peak_balance: Peak account balance (for drawdown calculation)
            session_id: Trading session ID for database queries
            previous_peak_balance: Previous peak balance for recovery calculation

        Returns:
            RiskAdjustments object with calculated adjustment factors
        """
        if not self.config.enabled:
            return RiskAdjustments(primary_reason="disabled")

        # Validate balance inputs to prevent NaN/Infinity propagation
        if not math.isfinite(current_balance) or current_balance < 0:
            logger.warning(f"Invalid current_balance: {current_balance}")
            return RiskAdjustments(primary_reason="invalid_balance")
        if not math.isfinite(peak_balance) or peak_balance <= 0:
            logger.warning(f"Invalid peak_balance: {peak_balance}")
            return RiskAdjustments(primary_reason="invalid_balance")

        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown(current_balance, peak_balance)

        # Check for recovery if we have previous peak data
        recovery_return = 0.0
        # Validate previous_peak_balance is numeric, positive, and finite before division
        if (
            previous_peak_balance
            and isinstance(previous_peak_balance, int | float)
            and previous_peak_balance > 0
            and math.isfinite(previous_peak_balance)
        ):
            recovery_return = (current_balance - previous_peak_balance) / previous_peak_balance

        # Get performance metrics
        performance_metrics = self._get_performance_metrics(session_id)

        # Calculate adjustments based on different factors
        drawdown_adjustment = self._calculate_drawdown_adjustment(current_drawdown, recovery_return)
        performance_adjustment = self._calculate_performance_adjustment(performance_metrics)
        volatility_adjustment = self._calculate_volatility_adjustment(performance_metrics)
        correlation_adjustment = self._calculate_correlation_adjustment(
            session_id
        )  # Placeholder for future implementation

        # Combine adjustments (take the most conservative)
        final_position_factor = min(
            drawdown_adjustment.position_size_factor,
            performance_adjustment.position_size_factor,
            volatility_adjustment.position_size_factor,
            correlation_adjustment.position_size_factor,
        )

        final_stop_loss_factor = max(
            drawdown_adjustment.stop_loss_tightening,
            performance_adjustment.stop_loss_tightening,
            volatility_adjustment.stop_loss_tightening,
            correlation_adjustment.stop_loss_tightening,
        )

        final_daily_risk_factor = min(
            drawdown_adjustment.daily_risk_factor,
            performance_adjustment.daily_risk_factor,
            volatility_adjustment.daily_risk_factor,
            correlation_adjustment.daily_risk_factor,
        )

        # Determine primary reason
        primary_reason = self._determine_primary_reason(
            drawdown_adjustment,
            performance_adjustment,
            volatility_adjustment,
            correlation_adjustment,
        )

        return RiskAdjustments(
            position_size_factor=final_position_factor,
            stop_loss_tightening=final_stop_loss_factor,
            daily_risk_factor=final_daily_risk_factor,
            primary_reason=primary_reason,
            adjustment_details={
                "current_drawdown": current_drawdown,
                "recovery_return": recovery_return,
                "drawdown_adjustment": drawdown_adjustment,
                "performance_adjustment": performance_adjustment,
                "volatility_adjustment": volatility_adjustment,
                "correlation_adjustment": correlation_adjustment,
                "performance_metrics": performance_metrics,
            },
        )

    def apply_risk_adjustments(
        self, risk_parameters: RiskParameters, adjustments: RiskAdjustments
    ) -> RiskParameters:
        """
        Apply dynamic adjustments to risk parameters.

        Args:
                risk_parameters: Original risk parameters
                adjustments: Calculated adjustments to apply

        Returns:
                New RiskParameters with adjustments applied
        """
        # Validate adjustment factors at boundary to prevent NaN/Infinity corruption
        position_factor = adjustments.position_size_factor
        if not math.isfinite(position_factor) or position_factor < 0:
            logger.warning(f"Invalid position_size_factor: {position_factor}, using 1.0")
            position_factor = 1.0

        stop_factor = adjustments.stop_loss_tightening
        if not math.isfinite(stop_factor) or stop_factor < 0:
            logger.warning(f"Invalid stop_loss_tightening: {stop_factor}, using 1.0")
            stop_factor = 1.0

        daily_factor = adjustments.daily_risk_factor
        if not math.isfinite(daily_factor) or daily_factor < 0:
            logger.warning(f"Invalid daily_risk_factor: {daily_factor}, using 1.0")
            daily_factor = 1.0

        # Validate risk_parameters fields to prevent NaN propagation through multiplication
        for field_name in [
            "base_risk_per_trade",
            "max_risk_per_trade",
            "max_position_size",
            "max_daily_risk",
            "max_correlated_risk",
            "position_size_atr_multiplier",
        ]:
            value = getattr(risk_parameters, field_name)
            if not math.isfinite(value) or value < 0:
                logger.warning(
                    f"Invalid risk_parameters.{field_name}: {value}, using original value"
                )
                # Return unmodified parameters if any field is invalid
                return risk_parameters

        # Create a copy to avoid modifying the original
        adjusted_params = RiskParameters(
            base_risk_per_trade=risk_parameters.base_risk_per_trade * position_factor,
            max_risk_per_trade=risk_parameters.max_risk_per_trade * position_factor,
            max_position_size=risk_parameters.max_position_size * position_factor,
            max_daily_risk=risk_parameters.max_daily_risk * daily_factor,
            max_correlated_risk=risk_parameters.max_correlated_risk * position_factor,
            max_drawdown=risk_parameters.max_drawdown,  # Don't adjust max drawdown threshold
            position_size_atr_multiplier=risk_parameters.position_size_atr_multiplier * stop_factor,
            default_take_profit_pct=risk_parameters.default_take_profit_pct,
            atr_period=risk_parameters.atr_period,
        )

        return adjusted_params

    def _calculate_current_drawdown(self, current_balance: float, peak_balance: float) -> float:
        """Calculate current drawdown percentage with epsilon protection for precision"""
        if peak_balance <= DEFAULT_EPSILON:
            return 0.0
        return max(0.0, (peak_balance - current_balance) / peak_balance)

    def _get_performance_metrics(self, session_id: int | None) -> dict[str, Any]:
        """Get cached performance metrics or calculate new ones with thread safety."""
        now = datetime.now(UTC)

        # Check cache validity with lock to prevent race conditions
        with self._cache_lock:
            if (
                self._cache_timestamp
                and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
                and self._performance_cache
            ):
                # Return copy to prevent external mutations
                return self._performance_cache.copy()
            # If another thread is computing, wait and return stale cache if available
            if self._computing:
                return self._performance_cache.copy() if self._performance_cache else {}
            # Mark that we're computing to prevent other threads from duplicating work
            self._computing = True

        # Calculate new metrics outside lock to minimize lock duration
        metrics = {}

        if self.db_manager and session_id:
            try:
                # Get recent performance data
                start_date = now - timedelta(days=self.config.performance_window_days)
                db_metrics = self.db_manager.get_dynamic_risk_performance_metrics(
                    start_date=start_date, session_id=session_id
                )

                metrics.update(db_metrics)

                # Estimate volatility from recent account equity if available
                try:
                    with self.db_manager.get_session() as session:
                        from src.database.models import AccountHistory

                        history = (
                            session.query(AccountHistory)
                            .filter(AccountHistory.session_id == session_id)
                            .filter(AccountHistory.timestamp >= start_date)
                            .order_by(AccountHistory.timestamp.asc())
                            .all()
                        )
                        closes: list[float] = []
                        for rec in history:
                            try:
                                closes.append(float(rec.equity))
                            except (ValueError, TypeError, AttributeError):
                                # Skip invalid equity values that cannot be converted to float
                                continue
                        # Need at least 3 prices to calculate 2 log returns (N prices -> N-1 returns)
                        if len(closes) >= 3:
                            series = pd.Series(closes)
                            # Volatility calculations require positive prices for log returns
                            # Non-positive values from data gaps would corrupt the volatility metric
                            series_positive = series[series > 0]
                            if len(series_positive) >= 3:
                                log_returns = np.log(series_positive).diff().dropna()
                                # Require at least 2 returns for reliable std (ddof=1 requires 2+ samples)
                                if len(log_returns) >= 2:
                                    vol = float(log_returns.std())
                                    if math.isfinite(vol):
                                        metrics["estimated_volatility"] = vol
                                    else:
                                        logger.debug("Volatility std() returned non-finite value")
                except Exception as vol_err:
                    logger.warning(f"Volatility estimation failed, using fallback: {vol_err}")

            except Exception as e:
                logger.warning(f"Failed to get performance metrics from database: {e}")

        # Cache the results with lock to prevent race conditions
        with self._cache_lock:
            self._performance_cache = metrics
            self._cache_timestamp = now
            self._computing = False

        return metrics

    def _calculate_drawdown_adjustment(
        self, current_drawdown: float, recovery_return: float = 0.0
    ) -> RiskAdjustments:
        """Calculate adjustments based on current drawdown level and recovery"""
        # Check for recovery first (de-throttling)
        if recovery_return > 0:
            for threshold in sorted(self.config.recovery_thresholds, reverse=True):
                if recovery_return >= threshold:
                    # Recovery allows aggressive position scaling up to 2x when performance is strong.
                    # This is intentional: asymmetric response (conservative on drawdown, aggressive
                    # on recovery) to capitalize on momentum after losses.
                    # Clamp recovery_factor to [0.1, 2.0] to prevent extreme adjustments
                    raw_recovery_factor = (
                        1.0 + (recovery_return - threshold) * DEFAULT_RECOVERY_SCALING_FACTOR
                    )
                    recovery_factor = max(0.1, min(raw_recovery_factor, 2.0))
                    return RiskAdjustments(
                        position_size_factor=min(1.0, recovery_factor),
                        stop_loss_tightening=max(1.0, 1.0 / recovery_factor),
                        daily_risk_factor=min(1.0, recovery_factor),
                        primary_reason=f"recovery_{recovery_return:.1%}",
                    )

        # Standard drawdown logic
        if current_drawdown <= 0:
            return RiskAdjustments(primary_reason="normal")

        # Find the appropriate threshold
        position_factor = 1.0
        stop_loss_factor = 1.0
        daily_risk_factor = 1.0

        for i, threshold in enumerate(self.config.drawdown_thresholds):
            if current_drawdown >= threshold:
                position_factor = self.config.risk_reduction_factors[i]
                daily_risk_factor = self.config.risk_reduction_factors[i]
                # Widen stop distance progressively based on drawdown level
                stop_loss_factor = 1.0 + (DEFAULT_DRAWDOWN_STOP_TIGHTENING_INCREMENT * i)

        reason = f"drawdown_{current_drawdown:.1%}"

        return RiskAdjustments(
            position_size_factor=position_factor,
            stop_loss_tightening=stop_loss_factor,
            daily_risk_factor=daily_risk_factor,
            primary_reason=reason,
        )

    def _calculate_performance_adjustment(
        self, performance_metrics: dict[str, Any]
    ) -> RiskAdjustments:
        """Calculate adjustments based on recent performance"""
        win_rate = performance_metrics.get("win_rate", DEFAULT_WIN_RATE_FALLBACK)
        profit_factor = performance_metrics.get("profit_factor", DEFAULT_PROFIT_FACTOR_FALLBACK)
        total_trades = performance_metrics.get("total_trades", 0)

        # Need minimum trades for reliable adjustment
        if total_trades < DEFAULT_MIN_TRADES_FOR_DYNAMIC_ADJUSTMENT:
            return RiskAdjustments(primary_reason="insufficient_data")

        # Calculate performance score using weighted factors
        normalized_profit = min(profit_factor / DEFAULT_PERFORMANCE_PROFIT_FACTOR_DIVISOR, 1.0)
        performance_score = (win_rate * DEFAULT_PERFORMANCE_WIN_RATE_WEIGHT) + (
            normalized_profit * DEFAULT_PERFORMANCE_PROFIT_FACTOR_WEIGHT
        )

        if performance_score < DEFAULT_PERFORMANCE_POOR_THRESHOLD:
            return RiskAdjustments(
                position_size_factor=DEFAULT_POOR_PERF_POSITION_FACTOR,
                stop_loss_tightening=DEFAULT_POOR_PERF_STOP_TIGHTENING,
                daily_risk_factor=DEFAULT_POOR_PERF_DAILY_RISK_FACTOR,
                primary_reason="poor_performance",
            )
        elif performance_score > DEFAULT_PERFORMANCE_GOOD_THRESHOLD:
            return RiskAdjustments(
                position_size_factor=DEFAULT_GOOD_PERF_POSITION_FACTOR,
                stop_loss_tightening=DEFAULT_GOOD_PERF_STOP_TIGHTENING,
                daily_risk_factor=DEFAULT_GOOD_PERF_DAILY_RISK_FACTOR,
                primary_reason="good_performance",
            )
        else:
            return RiskAdjustments(primary_reason="normal_performance")

    def _calculate_volatility_adjustment(
        self, performance_metrics: dict[str, Any]
    ) -> RiskAdjustments:
        """Calculate adjustments based on market volatility"""
        if not self.config.volatility_adjustment_enabled:
            return RiskAdjustments(primary_reason="volatility_disabled")

        # Prefer estimated volatility computed from equity history if present
        estimated_volatility = float(
            performance_metrics.get("estimated_volatility", DEFAULT_VOLATILITY_FALLBACK)
        )

        if estimated_volatility > self.config.high_volatility_threshold:
            return RiskAdjustments(
                position_size_factor=self.config.volatility_risk_multipliers[0],
                stop_loss_tightening=DEFAULT_HIGH_VOL_STOP_TIGHTENING,
                daily_risk_factor=self.config.volatility_risk_multipliers[0],
                primary_reason="high_volatility",
            )
        elif estimated_volatility < self.config.low_volatility_threshold:
            return RiskAdjustments(
                position_size_factor=self.config.volatility_risk_multipliers[1],
                stop_loss_tightening=DEFAULT_LOW_VOL_STOP_TIGHTENING,
                daily_risk_factor=self.config.volatility_risk_multipliers[1],
                primary_reason="low_volatility",
            )
        else:
            return RiskAdjustments(primary_reason="normal_volatility")

    def _calculate_correlation_adjustment(self, session_id: int | None) -> RiskAdjustments:
        """
        Calculate adjustments based on position correlation and concentration risk.

        Analyzes current open positions to detect excessive concentration risk:
        - Total exposure across all positions
        - Number of concurrent positions (diversification)
        - Concentration in single positions

        Note: Symbol-level correlation is handled by CorrelationEngine at the engine level.
        This provides portfolio-level concentration risk management.
        """
        if self.db_manager is None and self._positions_provider is None:
            return RiskAdjustments(primary_reason="correlation_no_data")

        try:
            # Get active positions from database when available; otherwise use in-memory snapshot
            if self.db_manager:
                positions = self.db_manager.get_active_positions(session_id=session_id)
            else:
                positions_snapshot = self._positions_provider() if self._positions_provider else {}
                positions = [
                    {"symbol": symbol, **(pos or {})}
                    for symbol, pos in positions_snapshot.items()
                ]
            if not positions:
                return RiskAdjustments(primary_reason="correlation_no_positions")

            # Calculate total exposure and concentration metrics
            total_exposure = 0.0
            max_single_exposure = 0.0
            num_positions = len(positions)

            for pos in positions:
                # Extract size as fraction of balance using current_size after partial exits.
                size = pos.get("current_size")
                if size is None:
                    size = pos.get("size", 0.0)
                if size is None:
                    size = 0.0
                size = float(size)
                if not math.isfinite(size):
                    logger.warning(
                        "Invalid position size for %s: %s",
                        pos.get("symbol", "unknown"),
                        size,
                    )
                    size = 0.0
                # Normalize sizes that arrive as raw quantities (exchange sync).
                if size > 1.0:
                    entry_balance = pos.get("entry_balance")
                    entry_price = pos.get("entry_price")
                    quantity = pos.get("quantity")
                    if entry_balance and entry_price and quantity:
                        size = float(quantity) * float(entry_price) / float(entry_balance)

                total_exposure += abs(size)  # Count both long and short
                max_single_exposure = max(max_single_exposure, abs(size))

            # Apply concentration-based risk reduction
            position_size_factor = 1.0
            stop_loss_factor = 1.0
            primary_reason = "normal"
            details = {
                "total_exposure": round(total_exposure, 4),
                "num_positions": num_positions,
                "max_single_exposure": round(max_single_exposure, 4),
            }

            # Check 1: Correlated exposure cap (proxy via low position count).
            # Treat small baskets as likely correlated and enforce cap on total exposure.
            max_correlated_positions = DEFAULT_MAX_CORRELATED_POSITIONS_FOR_CAP
            if num_positions <= max_correlated_positions and total_exposure > self.max_correlated_risk:
                excess_ratio = total_exposure / self.max_correlated_risk
                position_size_factor = 1.0 / excess_ratio
                position_size_factor = max(
                    DEFAULT_CORRELATED_EXPOSURE_MIN_FACTOR,
                    min(1.0, position_size_factor),
                )
                primary_reason = "high_correlated_exposure"
                details["excess_ratio"] = round(excess_ratio, 2)
                details["correlated_positions_cap"] = max_correlated_positions

            # Check 2: Low diversification (too few positions with high exposure)
            elif (
                num_positions <= DEFAULT_LOW_DIVERSIFICATION_POSITION_COUNT
                and total_exposure > DEFAULT_LOW_DIVERSIFICATION_EXPOSURE_THRESHOLD
            ):
                # Reduce new position sizing to encourage diversification
                position_size_factor = DEFAULT_LOW_DIVERSIFICATION_SIZE_FACTOR
                primary_reason = "low_diversification"

            # Check 3: Single position too large
            elif max_single_exposure > DEFAULT_LARGE_SINGLE_POSITION_THRESHOLD:
                # Widen stop distance on new positions
                stop_loss_factor = DEFAULT_LARGE_SINGLE_POSITION_STOP_MULTIPLIER
                position_size_factor = DEFAULT_LARGE_SINGLE_POSITION_SIZE_FACTOR
                primary_reason = "large_single_position"

            return RiskAdjustments(
                position_size_factor=position_size_factor,
                stop_loss_tightening=stop_loss_factor,
                daily_risk_factor=position_size_factor,  # Also reduce daily risk allowance
                primary_reason=primary_reason,
                adjustment_details=details,
            )

        except Exception as exc:
            logger.warning(
                "Failed to calculate correlation adjustment: %s",
                exc,
                exc_info=True,
            )
            return RiskAdjustments(primary_reason="correlation_error")

    def _determine_primary_reason(
        self,
        drawdown_adj: RiskAdjustments,
        performance_adj: RiskAdjustments,
        volatility_adj: RiskAdjustments,
        correlation_adj: RiskAdjustments,
    ) -> str:
        """Determine the primary reason for risk adjustment"""
        # Priority: drawdown > performance > volatility > correlation
        if drawdown_adj.position_size_factor < 1.0:
            return drawdown_adj.primary_reason
        elif performance_adj.position_size_factor != 1.0:
            return performance_adj.primary_reason
        elif volatility_adj.position_size_factor != 1.0:
            return volatility_adj.primary_reason
        elif correlation_adj.position_size_factor != 1.0:
            return correlation_adj.primary_reason
        else:
            return "normal"
