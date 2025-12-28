"""
Aggressive Trend-Following Strategy - High Risk, High Return

This strategy is designed to beat buy-and-hold returns through aggressive
trend following with leverage. It aims for maximum returns while accepting
high drawdown risk.

Strategy Characteristics:
- High leverage (2-3x effective through concentrated positions)
- Early trend entry (lower confidence threshold)
- Extended hold periods (ride trends to completion)
- Aggressive trailing stops (protect profits in strong trends)
- Dynamic position sizing based on trend strength

Risk Profile:
- Expected Max Drawdown: 60-80%
- Expected Sharpe Ratio: 0.5-1.0
- Target Returns: 2-3x buy-and-hold (in strong trending markets)
- Win Rate: 45-55%
- Profit Factor: 2.0-3.0

When to Use:
- Bull markets with strong trends
- High volatility environments
- When willing to accept high drawdown for high returns
- Phase 2 of beating buy-and-hold (aggressive alpha generation)

When NOT to Use:
- Ranging markets (high whipsaw risk)
- Low volatility environments
- Conservative risk tolerance
- After achieving target returns (switch to risk optimization)
"""

from __future__ import annotations

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    Strategy,
    VolatilityRiskManager,
)
from src.strategies.components.aggressive_trend_signal_generator import (
    AggressiveTrendSignalGenerator,
)


def create_aggressive_trend_strategy(
    name: str = "AggressiveTrend",
    # Signal generation parameters
    fast_ema: int = 8,
    medium_ema: int = 21,
    slow_ema: int = 50,
    adx_threshold: float = 25.0,
    min_confidence: float = 0.55,  # Lower for early entries
    # Position sizing parameters (AGGRESSIVE)
    base_fraction: float = 0.5,  # 50% base position
    max_fraction: float = 0.5,  # Allow up to 50% (system maximum)
    min_fraction: float = 0.3,  # Minimum 30% even in bad conditions
    # Risk management parameters (WIDE STOPS)
    base_stop_loss: float = 0.12,  # 12% stop loss (very wide)
    take_profit_pct: float = 0.50,  # 50% take profit target
    trailing_activation: float = 0.10,  # Activate trail at 10% profit
    trailing_distance: float = 0.05,  # 5% trailing distance
) -> Strategy:
    """
    Create aggressive trend-following strategy

    This strategy uses high leverage and wide stops to capture major trends
    while accepting high drawdown risk. Designed to beat buy-and-hold in
    strong trending markets.

    Args:
        name: Strategy name
        fast_ema: Fast EMA period for trend detection
        medium_ema: Medium EMA period for trend detection
        slow_ema: Slow EMA period for trend detection
        adx_threshold: Minimum ADX for trend confirmation
        min_confidence: Minimum confidence for entry (lower = earlier entries)
        base_fraction: Base position size as fraction of capital
        max_fraction: Maximum position size (system cap at 0.5)
        min_fraction: Minimum position size in adverse conditions
        base_stop_loss: Stop loss percentage (wider = more trend breathing room)
        take_profit_pct: Take profit percentage target
        trailing_activation: Profit level to activate trailing stop
        trailing_distance: Trailing stop distance from peak

    Returns:
        Configured Strategy instance optimized for aggressive trend following

    Example:
        >>> strategy = create_aggressive_trend_strategy(
        ...     base_fraction=0.5,  # 50% position size
        ...     base_stop_loss=0.15,  # 15% stop loss
        ... )
        >>> # Backtest with this strategy to measure vs buy-and-hold
        >>> backtester = Backtester(strategy=strategy, initial_balance=10_000)
        >>> results = backtester.run(symbol="BTCUSDT", timeframe="4h", days=1825)
        >>> print(f"vs Buy-Hold: {results['trading_vs_hold_difference']:+.2f}%")
    """
    # Create aggressive trend signal generator
    signal_generator = AggressiveTrendSignalGenerator(
        name=f"{name}_signals",
        fast_ema=fast_ema,
        medium_ema=medium_ema,
        slow_ema=slow_ema,
        adx_threshold=adx_threshold,
        min_confidence=min_confidence,
    )

    # Create volatility-based risk manager with WIDE stops
    # Wide stops allow trends to breathe and avoid premature exits
    risk_manager = VolatilityRiskManager(
        base_risk=base_stop_loss,  # 12-15% base stop
        atr_multiplier=2.5,  # Wider volatility adjustment
        min_risk=0.08,  # Minimum 8% stop (still wide)
        max_risk=0.20,  # Maximum 20% stop (very wide)
    )

    # Create confidence-weighted position sizer
    # Higher confidence = larger position size (up to max_fraction)
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=min(base_fraction, 0.5),  # Enforce system maximum
        min_confidence=0.3,  # Accept lower confidence for aggressive trading
    )

    # Create regime detector for context
    regime_detector = EnhancedRegimeDetector()

    # Compose strategy
    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    # Configure aggressive risk overrides
    # These are consumed by the backtesting/live engines
    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": base_fraction,
        "min_fraction": min_fraction,
        "max_fraction": max_fraction,
        "stop_loss_pct": base_stop_loss,
        "take_profit_pct": take_profit_pct,
        # Dynamic risk management
        "dynamic_risk": {
            "enabled": True,
            # Reduce position size on drawdowns (but stay aggressive)
            "drawdown_thresholds": [0.30, 0.45, 0.60],  # Deeper drawdowns before reducing
            "risk_reduction_factors": [0.90, 0.80, 0.70],  # Modest reductions
            "recovery_thresholds": [0.15, 0.30],  # Quick recovery to full size
        },
        # Partial exit operations (take profits along the way)
        "partial_operations": {
            "exit_targets": [0.15, 0.30, 0.50],  # Take profits at 15%, 30%, 50%
            "exit_sizes": [0.20, 0.25, 0.30],  # Exit 20%, 25%, 30% at each level
            "scale_in_thresholds": [0.03, 0.06],  # Add to position on pullbacks
            "scale_in_sizes": [0.30, 0.20],  # Add 30% and 20% more
            "max_scale_ins": 2,  # Maximum 2 scale-in additions
        },
        # Aggressive trailing stop (protect profits in strong trends)
        "trailing_stop": {
            "activation_threshold": trailing_activation,  # Activate at 10% profit
            "trailing_distance_pct": trailing_distance,  # Trail 5% below peak
            "breakeven_threshold": 0.15,  # Move to breakeven at 15% profit
            "breakeven_buffer": 0.02,  # 2% buffer above entry
        },
    }

    # Store parameters for testing/validation
    strategy.config = {
        "base_fraction": base_fraction,
        "max_fraction": max_fraction,
        "min_fraction": min_fraction,
        "base_stop_loss": base_stop_loss,
        "take_profit_pct": take_profit_pct,
        "fast_ema": fast_ema,
        "medium_ema": medium_ema,
        "slow_ema": slow_ema,
        "adx_threshold": adx_threshold,
    }

    return strategy


def create_ultra_aggressive_trend_strategy(
    name: str = "UltraAggressiveTrend",
) -> Strategy:
    """
    Create ULTRA aggressive variant with maximum risk/return profile

    This is the most aggressive configuration:
    - Maximum position sizes (50%)
    - Widest stops (20%)
    - Earliest entries (low confidence threshold)
    - Maximum leverage effect

    WARNING: Expected drawdown 70-90%. Only use if willing to accept
    severe drawdowns in exchange for maximum upside potential.

    Returns:
        Ultra-aggressive strategy configuration
    """
    return create_aggressive_trend_strategy(
        name=name,
        base_fraction=0.5,  # Maximum allowed
        max_fraction=0.5,  # Maximum allowed
        min_fraction=0.40,  # Stay large even in drawdowns
        base_stop_loss=0.18,  # Very wide 18% stop
        take_profit_pct=0.75,  # Extended 75% profit target
        trailing_activation=0.15,  # Later activation
        trailing_distance=0.08,  # Wider trail distance
        min_confidence=0.50,  # Very low confidence threshold (early entries)
        adx_threshold=20.0,  # Lower ADX threshold (catch trends earlier)
    )


def create_moderate_aggressive_trend_strategy(
    name: str = "ModerateAggressiveTrend",
) -> Strategy:
    """
    Create moderately aggressive variant (step down from full aggression)

    This configuration balances aggression with some risk control:
    - Moderate position sizes (35%)
    - Moderate stops (10%)
    - Standard entry threshold
    - Balanced risk/return

    Expected drawdown: 40-50%
    Expected returns: 1.5-2x buy-and-hold

    Returns:
        Moderately aggressive strategy configuration
    """
    return create_aggressive_trend_strategy(
        name=name,
        base_fraction=0.35,  # Moderate size
        max_fraction=0.45,  # Allow some upside
        min_fraction=0.20,  # Reduce more in drawdowns
        base_stop_loss=0.10,  # Tighter 10% stop
        take_profit_pct=0.35,  # More conservative profit target
        trailing_activation=0.08,  # Earlier activation
        trailing_distance=0.04,  # Tighter trail
        min_confidence=0.60,  # Higher confidence required
        adx_threshold=27.0,  # Higher ADX threshold (stronger trends only)
    )
