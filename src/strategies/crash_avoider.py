"""
Crash Avoider Strategy - Capital Preservation Focus

This strategy focuses on avoiding major market crashes by exiting positions
early when crash indicators are detected. The goal is to preserve capital
during bear markets while staying invested during bull markets.

Key Insight:
Buy-and-hold suffers 60-85% drawdowns in crashes. If we can exit before
losing more than 20% and re-enter after stabilization, we beat buy-and-hold
by avoiding the catastrophic losses.

Historical Performance Target:
- 2018 Crash: Exit at -15%, avoid additional -70% decline
- COVID 2020: Exit at -20%, avoid additional -40% decline
- May 2021: Exit at -15%, avoid additional -39% decline
- 2022 LUNA/FTX: Exit at -20%, avoid additional -46% decline

Strategy Profile:
- Win Rate: 40-45% (many small losses from whipsaws)
- Profit Factor: 3.0-5.0 (huge wins from avoided crashes)
- Max Drawdown: 20-30% (vs buy-and-hold 60-85%)
- Sharpe Ratio: 1.5-2.5 (excellent risk-adjusted returns)
- Best Market: Bear markets and high volatility
- Worst Market: Steady bull trends (whipsaw risk)

Deployment Strategy:
Combine with aggressive trend strategy (60/40 split):
- 60% in aggressive trend (capture upside)
- 40% in crash avoider (protect downside)
- Result: Beat buy-and-hold returns with lower drawdown
"""

from __future__ import annotations

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    FixedRiskManager,
    Strategy,
)
from src.strategies.components.crash_detection_signal_generator import (
    CrashDetectionSignalGenerator,
)


def create_crash_avoider_strategy(
    name: str = "CrashAvoider",
    # Crash detection parameters
    rsi_overbought: float = 70.0,
    volume_spike_threshold: float = 3.0,
    atr_expansion_threshold: float = 2.0,
    waterfall_threshold: float = -0.15,
    min_crash_indicators: int = 2,
    # Position sizing (conservative - focus on preservation)
    base_fraction: float = 0.30,  # Moderate 30% position
    max_fraction: float = 0.40,  # Cap at 40%
    min_fraction: float = 0.10,  # Minimum 10% during crashes
    # Risk management (tight stops - exit quickly if wrong)
    base_stop_loss: float = 0.05,  # Tight 5% stop
    take_profit_pct: float = 0.20,  # Take 20% profits quickly
) -> Strategy:
    """
    Create crash avoider strategy focused on capital preservation

    This strategy monitors crash indicators and exits before major
    drawdowns. Optimized for avoiding 60-85% buy-and-hold crashes
    by exiting at 15-20% drawdowns.

    Args:
        name: Strategy name
        rsi_overbought: RSI level indicating potential top
        volume_spike_threshold: Volume panic threshold
        atr_expansion_threshold: Volatility expansion threshold
        waterfall_threshold: Rapid decline threshold
        min_crash_indicators: Indicators required for crash signal
        base_fraction: Base position size
        max_fraction: Maximum position size
        min_fraction: Minimum position during crash
        base_stop_loss: Stop loss percentage (tight)
        take_profit_pct: Take profit target (conservative)

    Returns:
        Configured Strategy instance optimized for crash avoidance

    Example:
        >>> strategy = create_crash_avoider_strategy()
        >>> # Backtest on historical crashes
        >>> results = backtester.run(
        ...     symbol="BTCUSDT",
        ...     start=datetime(2022, 1, 1),  # 2022 crash year
        ...     end=datetime(2022, 12, 31),
        ... )
        >>> print(f"vs Buy-Hold: {results['trading_vs_hold_difference']:+.2f}%")
        >>> # Expected: +40-50% (avoided the crash)
    """
    # Create crash detection signal generator
    signal_generator = CrashDetectionSignalGenerator(
        name=f"{name}_signals",
        rsi_overbought=rsi_overbought,
        volume_spike_threshold=volume_spike_threshold,
        atr_expansion_threshold=atr_expansion_threshold,
        waterfall_threshold=waterfall_threshold,
        min_indicators=min_crash_indicators,
    )

    # Create fixed risk manager with TIGHT stops
    # We want to exit quickly if crash signal is false alarm
    risk_manager = FixedRiskManager(
        base_risk=base_stop_loss,  # 5% tight stop
        min_risk=0.03,  # Minimum 3%
        max_risk=0.08,  # Maximum 8%
    )

    # Create conservative position sizer
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=base_fraction,
        min_confidence=0.6,  # Higher confidence required
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

    # Configure crash-focused risk overrides
    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": base_fraction,
        "min_fraction": min_fraction,
        "max_fraction": max_fraction,
        "stop_loss_pct": base_stop_loss,
        "take_profit_pct": take_profit_pct,
        # Aggressive dynamic risk during crashes
        "dynamic_risk": {
            "enabled": True,
            # Exit aggressively on any drawdown
            "drawdown_thresholds": [0.10, 0.15, 0.20],  # Shallow thresholds
            "risk_reduction_factors": [0.70, 0.50, 0.20],  # Aggressive reductions
            "recovery_thresholds": [0.05, 0.10],  # Quick recovery
        },
        # No partial operations - all in or all out approach
        "partial_operations": {
            "exit_targets": [0.10, 0.20],  # Quick profit taking
            "exit_sizes": [0.40, 0.40],  # Exit most of position
            "scale_in_thresholds": [],  # No scale-in (too risky in crashes)
            "scale_in_sizes": [],
            "max_scale_ins": 0,
        },
        # Tight trailing stop
        "trailing_stop": {
            "activation_threshold": 0.05,  # Activate at 5% profit
            "trailing_distance_pct": 0.03,  # Trail 3% below peak
            "breakeven_threshold": 0.08,  # Move to breakeven quickly
            "breakeven_buffer": 0.01,  # Small 1% buffer
        },
    }

    # Store config
    strategy.config = {
        "base_fraction": base_fraction,
        "max_fraction": max_fraction,
        "min_fraction": min_fraction,
        "base_stop_loss": base_stop_loss,
        "take_profit_pct": take_profit_pct,
        "rsi_overbought": rsi_overbought,
        "volume_spike_threshold": volume_spike_threshold,
        "atr_expansion_threshold": atr_expansion_threshold,
        "waterfall_threshold": waterfall_threshold,
        "min_crash_indicators": min_crash_indicators,
    }

    return strategy


def create_ultra_defensive_crash_avoider_strategy(
    name: str = "UltraDefensiveCrashAvoider",
) -> Strategy:
    """
    Create ultra-defensive variant (maximum crash protection)

    This variant is extremely conservative:
    - Very small positions (20% max)
    - Very tight stops (3%)
    - Single crash indicator triggers exit
    - Exits aggressively on any weakness

    Use when: Maximum capital preservation is priority
    Expected: Very low drawdown (<15%) but may underperform in bull markets

    Returns:
        Ultra-defensive crash avoider configuration
    """
    return create_crash_avoider_strategy(
        name=name,
        base_fraction=0.15,  # Very small positions
        max_fraction=0.20,  # Cap at 20%
        min_fraction=0.05,  # Minimal exposure
        base_stop_loss=0.03,  # Very tight 3% stop
        take_profit_pct=0.15,  # Quick 15% profits
        min_crash_indicators=1,  # Single indicator triggers exit
        volume_spike_threshold=2.5,  # Lower panic threshold
        atr_expansion_threshold=1.8,  # Lower volatility threshold
    )


def create_balanced_crash_avoider_strategy(
    name: str = "BalancedCrashAvoider",
) -> Strategy:
    """
    Create balanced variant (moderate protection + returns)

    This variant balances crash protection with return potential:
    - Moderate positions (40% max)
    - Moderate stops (7%)
    - Requires 3 indicators for crash signal
    - Stays invested during normal volatility

    Use when: Want crash protection but don't want to miss bull markets
    Expected: Moderate drawdown (25-35%) with good upside capture

    Returns:
        Balanced crash avoider configuration
    """
    return create_crash_avoider_strategy(
        name=name,
        base_fraction=0.35,  # Moderate positions
        max_fraction=0.45,  # Allow larger positions
        min_fraction=0.15,  # Stay somewhat invested
        base_stop_loss=0.07,  # Moderate 7% stop
        take_profit_pct=0.30,  # Extended 30% profits
        min_crash_indicators=3,  # Require strong crash signal
        volume_spike_threshold=3.5,  # Higher panic threshold
        atr_expansion_threshold=2.5,  # Higher volatility threshold
    )
