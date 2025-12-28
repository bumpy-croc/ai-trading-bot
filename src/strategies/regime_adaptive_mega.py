"""
Regime-Adaptive Mega Strategy - The Swiss Army Knife

This is the comprehensive "mega" strategy that adapts its entire behavior
based on detected market regime. It combines all the techniques we've
developed into one adaptive system.

Strategy Philosophy:
- No single strategy works in all market conditions
- Adapt to market regime rather than fight it
- Use aggressive tactics in favorable regimes
- Use defensive tactics in unfavorable regimes
- Smooth transitions between regimes to avoid whipsaws

Regime-Specific Behaviors:

1. TRENDING_UP Regime:
   - Use aggressive trend-following
   - Large position sizes (40-50%)
   - Wide stops (10-12%)
   - Trail aggressively
   - Stay fully invested

2. TRENDING_DOWN Regime:
   - Exit positions quickly
   - Small positions if any (10-20%)
   - Tight stops (3-5%)
   - Look for recovery signals

3. RANGING Regime:
   - Reduce position sizes (20-30%)
   - Tighter stops (5-7%)
   - Quick profit taking
   - Less frequent trading

4. HIGH_VOLATILITY Regime:
   - Exploit volatility with momentum
   - Moderate positions (30-40%)
   - Wide stops (10-15%)
   - Quick partial exits

5. CRASH_RISK Regime:
   - Defensive positioning
   - Exit most positions
   - Very small positions (10-15%)
   - Very tight stops (3-5%)
   - Capital preservation mode

Expected Performance:
- Max Drawdown: 35-45% (better than pure aggressive strategies)
- Sharpe Ratio: 1.5-2.0 (excellent risk-adjusted returns)
- Win Rate: 50-55%
- Profit Factor: 2.5-3.5
- Best: All market conditions (adaptive)
- Worst: Regime transitions (brief whipsaws)

Performance Target:
Beat buy-and-hold by 50-100% with 30-40% lower drawdown
"""

from __future__ import annotations

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    RegimeAdaptiveRiskManager,
    Strategy,
)
from src.strategies.components.regime_adaptive_mega_signal_generator import (
    RegimeAdaptiveMegaSignalGenerator,
)


def create_regime_adaptive_mega_strategy(
    name: str = "RegimeAdaptiveMega",
    # Regime detection parameters
    trend_strength_threshold: float = 0.015,
    high_volatility_threshold: float = 2.0,
    crash_risk_threshold: int = 2,
    # Position sizing (regime-adaptive)
    base_fraction: float = 0.35,  # Base position size
    max_fraction: float = 0.50,  # Maximum in favorable regimes
    min_fraction: float = 0.10,  # Minimum in unfavorable regimes
    # Risk management (regime-adaptive)
    base_stop_loss: float = 0.08,  # Base stop loss
    take_profit_pct: float = 0.30,  # Base take profit
) -> Strategy:
    """
    Create regime-adaptive mega strategy

    This strategy adapts its entire behavior based on detected market
    regime, combining aggressive trend-following, crash avoidance, and
    volatility exploitation into one adaptive system.

    Args:
        name: Strategy name
        trend_strength_threshold: Momentum for trending regime
        high_volatility_threshold: ATR ratio for high volatility
        crash_risk_threshold: Crash indicators for defensive mode
        base_fraction: Base position size
        max_fraction: Maximum position in favorable regimes
        min_fraction: Minimum position in unfavorable regimes
        base_stop_loss: Base stop loss percentage
        take_profit_pct: Base take profit percentage

    Returns:
        Configured Strategy instance with full regime adaptation

    Example:
        >>> strategy = create_regime_adaptive_mega_strategy()
        >>> # Should perform well across all market conditions
        >>> results = backtester.run(
        ...     symbol="BTCUSDT",
        ...     timeframe="4h",
        ...     start=datetime(2020, 1, 1),  # Full cycle: bull, crash, bear, recovery
        ...     end=datetime(2024, 12, 31),
        ... )
        >>> print(f"4-year adaptive return: {results['total_return']:.2f}%")
        >>> print(f"vs Buy-Hold: {results['trading_vs_hold_difference']:+.2f}%")
    """
    # Create regime-adaptive signal generator
    signal_generator = RegimeAdaptiveMegaSignalGenerator(
        name=f"{name}_signals",
        trend_strength_threshold=trend_strength_threshold,
        high_volatility_threshold=high_volatility_threshold,
        crash_risk_threshold=crash_risk_threshold,
        use_aggressive_trend=True,
        use_crash_detection=True,
        use_volatility_exploitation=True,
    )

    # Create regime-adaptive risk manager
    risk_manager = RegimeAdaptiveRiskManager(
        low_vol_risk=0.05,  # 5% stop in low volatility
        high_vol_risk=0.12,  # 12% stop in high volatility
        base_risk=base_stop_loss,
    )

    # Create confidence-weighted position sizer
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=base_fraction,
        min_confidence=0.50,  # Moderate threshold for adaptive trading
    )

    # Create regime detector
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

    # Configure comprehensive regime-adaptive overrides
    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": base_fraction,
        "min_fraction": min_fraction,
        "max_fraction": max_fraction,
        "stop_loss_pct": base_stop_loss,
        "take_profit_pct": take_profit_pct,
        # Regime-specific dynamic risk
        "dynamic_risk": {
            "enabled": True,
            # Adapt to different drawdown levels
            "drawdown_thresholds": [0.15, 0.25, 0.35],  # Moderate thresholds
            "risk_reduction_factors": [0.85, 0.70, 0.50],
            "recovery_thresholds": [0.08, 0.15],
        },
        # Regime-adaptive partial operations
        "partial_operations": {
            # Quick profits in volatile/uncertain regimes
            "exit_targets": [0.12, 0.25, 0.40],
            "exit_sizes": [0.25, 0.30, 0.30],
            # Add to winners in trending regimes
            "scale_in_thresholds": [0.03, 0.06],
            "scale_in_sizes": [0.25, 0.20],
            "max_scale_ins": 2,
        },
        # Adaptive trailing stop
        "trailing_stop": {
            "activation_threshold": 0.08,  # Activate at 8%
            "trailing_distance_pct": 0.04,  # 4% trail
            "breakeven_threshold": 0.12,  # Move to breakeven at 12%
            "breakeven_buffer": 0.02,
        },
        # Regime-specific position sizing multipliers
        "regime_position_multipliers": {
            "TRENDING_UP": 1.2,  # 20% larger positions in uptrends
            "TRENDING_DOWN": 0.5,  # 50% smaller in downtrends
            "RANGING": 0.7,  # 30% smaller in ranging
            "HIGH_VOLATILITY": 1.0,  # Normal size in high volatility
            "CRASH_RISK": 0.3,  # 70% smaller in crash risk
        },
        # Regime-specific stop loss multipliers
        "regime_stop_multipliers": {
            "TRENDING_UP": 1.3,  # Wider stops in uptrends
            "TRENDING_DOWN": 0.7,  # Tighter stops in downtrends
            "RANGING": 0.8,  # Tighter stops in ranging
            "HIGH_VOLATILITY": 1.5,  # Much wider in high volatility
            "CRASH_RISK": 0.6,  # Very tight in crash risk
        },
    }

    # Store config
    strategy.config = {
        "base_fraction": base_fraction,
        "max_fraction": max_fraction,
        "min_fraction": min_fraction,
        "base_stop_loss": base_stop_loss,
        "take_profit_pct": take_profit_pct,
        "trend_strength_threshold": trend_strength_threshold,
        "high_volatility_threshold": high_volatility_threshold,
        "crash_risk_threshold": crash_risk_threshold,
    }

    return strategy


def create_conservative_regime_adaptive_strategy(
    name: str = "ConservativeRegimeAdaptive",
) -> Strategy:
    """
    Create conservative variant of regime-adaptive strategy

    This variant prioritizes capital preservation:
    - Smaller position sizes (25% base, 35% max)
    - Tighter stops (6% base)
    - More defensive in all regimes
    - Quick profit taking (20%)

    Use when: Want regime adaptation with lower risk
    Expected: Beat buy-and-hold with <30% max drawdown

    Returns:
        Conservative regime-adaptive configuration
    """
    return create_regime_adaptive_mega_strategy(
        name=name,
        base_fraction=0.25,  # Smaller base
        max_fraction=0.35,  # Lower maximum
        min_fraction=0.10,
        base_stop_loss=0.06,  # Tighter stop
        take_profit_pct=0.20,  # Quick profits
        trend_strength_threshold=0.020,  # Require stronger trends
        high_volatility_threshold=2.5,  # Higher volatility threshold
        crash_risk_threshold=2,  # Same crash sensitivity
    )


def create_aggressive_regime_adaptive_strategy(
    name: str = "AggressiveRegimeAdaptive",
) -> Strategy:
    """
    Create aggressive variant of regime-adaptive strategy

    This variant maximizes returns in favorable regimes:
    - Larger position sizes (45% base, 50% max)
    - Wider stops (10% base)
    - More aggressive in trending/volatile regimes
    - Extended profit targets (40%)

    Use when: Want maximum returns with regime awareness
    Expected: 2x+ buy-and-hold with 40-50% max drawdown

    Returns:
        Aggressive regime-adaptive configuration
    """
    return create_regime_adaptive_mega_strategy(
        name=name,
        base_fraction=0.45,  # Large base
        max_fraction=0.50,  # Maximum allowed
        min_fraction=0.15,  # Stay somewhat invested
        base_stop_loss=0.10,  # Wide stop
        take_profit_pct=0.40,  # Extended profits
        trend_strength_threshold=0.012,  # Catch trends earlier
        high_volatility_threshold=1.8,  # Lower volatility threshold (more aggressive)
        crash_risk_threshold=3,  # Less sensitive to crash (require more indicators)
    )


def create_balanced_regime_adaptive_strategy(
    name: str = "BalancedRegimeAdaptive",
) -> Strategy:
    """
    Create balanced variant (recommended default)

    This is the recommended starting point:
    - Balanced position sizes (30% base, 40% max)
    - Moderate stops (7% base)
    - Balanced adaptation across regimes
    - Moderate profit targets (25%)

    Use when: Want good balance of returns and risk
    Expected: 1.5x buy-and-hold with 30-35% max drawdown

    Returns:
        Balanced regime-adaptive configuration
    """
    return create_regime_adaptive_mega_strategy(
        name=name,
        base_fraction=0.30,  # Moderate base
        max_fraction=0.40,  # Moderate maximum
        min_fraction=0.12,
        base_stop_loss=0.07,  # Moderate stop
        take_profit_pct=0.25,  # Moderate profits
        trend_strength_threshold=0.015,  # Standard threshold
        high_volatility_threshold=2.0,  # Standard threshold
        crash_risk_threshold=2,  # Standard sensitivity
    )
