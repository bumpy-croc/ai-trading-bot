"""Adaptive Trend Strategy - Component-Based Implementation

A trend-following strategy designed to beat buy-and-hold by capturing major
bull runs and avoiding prolonged bear markets. Uses EMA crossovers for trend
detection with regime-aware filtering.

Key principles:
1. Stay fully invested during confirmed uptrends to capture bull runs
2. Exit during confirmed downtrends to avoid major drawdowns (e.g., 2022 bear)
3. Use trailing stops to protect profits and lock in gains
4. Minimize trading frequency to reduce fee drag
5. Long-only - no shorting

The edge over buy-and-hold comes from avoiding major crashes, not from
predicting short-term moves. Even partial crash avoidance (e.g., dodging
50% of a 60% drawdown) compounds into significant outperformance over
multiple market cycles.
"""

from typing import Any

from src.strategies.components import (
    EnhancedRegimeDetector,
    Strategy,
)
from src.strategies.components.adaptive_trend_signal_generator import (
    AdaptiveTrendSignalGenerator,
)
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection


class TrendFollowingRiskManager(RiskManager):
    """Risk manager for trend-following strategies with large position sizing.

    Unlike conservative risk managers that cap positions at 10-15% of balance,
    this manager allows large allocations suited to a strategy that is
    either fully in or fully out of the market. The risk is managed at the
    strategy level (via trend detection) rather than at the position level.
    """

    def __init__(
        self,
        target_allocation: float = 0.90,
        stop_loss_pct: float = 0.15,
    ) -> None:
        """Initialize the trend-following risk manager.

        Args:
            target_allocation: Target position size as fraction of balance.
            stop_loss_pct: Stop loss percentage for positions.
        """
        super().__init__("trend_following_risk_manager")

        if not 0.01 <= target_allocation <= 0.99:
            raise ValueError(
                f"target_allocation must be between 0.01 and 0.99, got {target_allocation}"
            )
        if not 0.01 <= stop_loss_pct <= 0.50:
            raise ValueError(f"stop_loss_pct must be between 0.01 and 0.50, got {stop_loss_pct}")

        self.target_allocation = target_allocation
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        """Calculate position size as a fixed fraction of balance.

        Trend-following is binary: fully invested in confirmed uptrends,
        fully out in downtrends. Position size is fixed at target_allocation
        to maximize compounding during multi-month bull runs.

        Args:
            signal: Trading signal.
            balance: Available balance.
            regime: Optional regime context.

        Returns:
            Position size in base currency (notional).
        """
        self.validate_inputs(balance)

        if signal.direction == SignalDirection.HOLD:
            return 0.0

        # Fixed allocation: either fully in or fully out.
        # Confidence scaling is intentionally removed — the signal generator's
        # confirmation logic already filters weak signals.
        return balance * self.target_allocation

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> bool:
        """Determine exit based on stop loss.

        Args:
            position: Current position.
            current_data: Current market data.
            regime: Optional regime context.

        Returns:
            True if position should be exited.
        """
        # Convert percentage (-20 for -20% loss) to decimal ratio (0.20)
        loss_ratio = abs(position.get_pnl_percentage()) / 100
        return position.get_pnl_percentage() < 0 and loss_ratio >= self.stop_loss_pct

    def get_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        """Calculate stop loss level.

        Args:
            entry_price: Entry price.
            signal: Trading signal.
            regime: Optional regime context.

        Returns:
            Stop loss price.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        if signal.direction == SignalDirection.BUY:
            return entry_price * (1 - self.stop_loss_pct)
        if signal.direction == SignalDirection.SELL:
            return entry_price * (1 + self.stop_loss_pct)
        return entry_price

    def get_parameters(self) -> dict[str, Any]:
        """Get risk manager parameters."""
        params = super().get_parameters()
        params.update(
            {
                "target_allocation": self.target_allocation,
                "stop_loss_pct": self.stop_loss_pct,
            }
        )
        return params


class TrendFollowingPositionSizer(PositionSizer):
    """Position sizer that passes through the risk manager's allocation.

    Standard position sizers cap at 20% via apply_bounds_checking.
    This sizer allows the full allocation from the risk manager to pass
    through, capped only by the strategy-level and engine-level limits.
    """

    def __init__(self, max_fraction: float = 0.95) -> None:
        """Initialize the trend-following position sizer.

        Args:
            max_fraction: Maximum fraction of balance for a single position.
        """
        super().__init__("trend_following_sizer")
        self.max_fraction = max_fraction

    def calculate_size(
        self,
        signal: "Signal",
        balance: float,
        risk_amount: float,
        regime: RegimeContext | None = None,
    ) -> float:
        """Pass through the risk manager's allocation with minimal adjustment.

        Args:
            signal: Trading signal.
            balance: Available balance.
            risk_amount: Amount from the risk manager.
            regime: Optional regime context.

        Returns:
            Position size in base currency.
        """
        self.validate_inputs(balance, risk_amount)

        if signal.direction == SignalDirection.HOLD:
            return 0.0

        if risk_amount <= 0:
            return 0.0

        # Use the risk manager's allocation directly, bounded by max_fraction
        max_size = balance * self.max_fraction
        return min(risk_amount, max_size)

    def get_parameters(self) -> dict[str, Any]:
        """Get position sizer parameters."""
        params = super().get_parameters()
        params.update({"max_fraction": self.max_fraction})
        return params


def create_adaptive_trend_strategy(
    name: str = "AdaptiveTrend",
    trend_ema_period: int = 90,
    entry_confirmation_days: int = 2,
    exit_confirmation_days: int = 18,
    entry_buffer_pct: float = 0.005,
    exit_buffer_pct: float = 0.08,
    exit_ratio_threshold: float = 0.65,
    ema_slope_lookback: int = 35,
    target_allocation: float = 0.99,
    max_position_pct: float = 0.99,
    stop_loss_pct: float = 0.40,
    take_profit_pct: float = 10.0,
) -> Strategy:
    """Create an adaptive trend-following strategy.

    This strategy uses the price position relative to a long-period EMA
    to detect major trends. Positions are sized aggressively during
    confirmed uptrends. The strategy stays out during bear markets to
    preserve capital.

    Args:
        name: Strategy name.
        trend_ema_period: Period for the main trend EMA.
        entry_confirmation_days: Consecutive days above EMA to confirm entry.
        exit_confirmation_days: Consecutive days below EMA to confirm exit.
        entry_buffer_pct: Price must be this % above EMA for entry.
        exit_buffer_pct: Price must be this % below EMA for exit.
        exit_ratio_threshold: Fraction of exit window days that must be below
            threshold to trigger exit (0.65 = 65% of recent days).
        ema_slope_lookback: Days to measure EMA slope; entry blocked when declining.
        target_allocation: Target position size as fraction of balance.
        max_position_pct: Maximum position size cap in Strategy validation.
        stop_loss_pct: Initial stop loss percentage (kept wide for crypto).
        take_profit_pct: Take profit percentage (kept very high to let winners run).

    Returns:
        Configured Strategy instance.
    """
    signal_generator = AdaptiveTrendSignalGenerator(
        name=f"{name}_signals",
        trend_ema_period=trend_ema_period,
        entry_confirmation_days=entry_confirmation_days,
        exit_confirmation_days=exit_confirmation_days,
        entry_buffer_pct=entry_buffer_pct,
        exit_buffer_pct=exit_buffer_pct,
        exit_ratio_threshold=exit_ratio_threshold,
        ema_slope_lookback=ema_slope_lookback,
    )

    risk_manager = TrendFollowingRiskManager(
        target_allocation=target_allocation,
        stop_loss_pct=stop_loss_pct,
    )

    position_sizer = TrendFollowingPositionSizer(
        max_fraction=max_position_pct,
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    # Allow large positions through strategy-level validation
    strategy._max_position_pct = max_position_pct

    # Expose configuration for engine consumption
    strategy.stop_loss_pct = stop_loss_pct
    strategy.take_profit_pct = take_profit_pct
    strategy.base_fraction = target_allocation

    # Risk overrides consumed by the backtest/live engines
    # No trailing stop: trailing stops with tight distances cause premature exits
    # during normal bull market corrections (20-30% dips are common for BTC).
    # Risk is managed at the strategy level via EMA trend detection.
    strategy.set_risk_overrides(
        {
            "position_sizer": "trend_following",
            "base_fraction": target_allocation,
            "min_fraction": 0.50,
            "max_fraction": max_position_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
    )

    return strategy
