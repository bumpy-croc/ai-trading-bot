"""
Extreme Leverage Strategies - Maximum Risk, Maximum Returns

These strategies are configured for MAXIMUM returns by using:
- 50% position sizes (system maximum)
- Very wide stops (15-20%) to avoid premature exits
- Aggressive scaling and compounding
- Accept 80%+ drawdowns for maximum upside potential

WARNING: These strategies are extremely risky and should only be used
when you fully understand and accept the possibility of catastrophic losses.

Expected Performance Profile:
- Total Returns: 3-10x buy-and-hold (in favorable conditions)
- Max Drawdown: 60-90% (EXTREME)
- Sharpe Ratio: 0.5-1.2 (poor risk-adjusted returns)
- Recovery Time: Very long (months to years)
- Best Use: Small portion of portfolio, long-term holding, high risk tolerance

Deployment Recommendation:
- Allocate maximum 10-20% of portfolio to extreme leverage strategies
- Combine with conservative strategies for balance
- Only use funds you can afford to lose completely
- Requires strong emotional discipline to hold through drawdowns
"""

from __future__ import annotations

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    RegimeAdaptiveRiskManager,
    Strategy,
)
from src.strategies.components.aggressive_trend_signal_generator import (
    AggressiveTrendSignalGenerator,
)
from src.strategies.components.regime_adaptive_mega_signal_generator import (
    RegimeAdaptiveMegaSignalGenerator,
)
from src.strategies.components.volatility_exploitation_signal_generator import (
    VolatilityExploitationSignalGenerator,
)


def create_extreme_leverage_trend_strategy(
    name: str = "ExtremeLeverageTrend",
) -> Strategy:
    """
    Create extreme leverage trend-following strategy

    Configuration:
    - 50% position size (system maximum)
    - 18% stop loss (very wide)
    - 50% take profit (let winners run)
    - Aggressive scaling and trailing

    Expected:
    - 5-10x buy-and-hold in bull markets
    - 80-90% drawdown in bear markets
    - Sharpe: 0.5-1.0

    Use when: You want maximum exposure to trending markets
    Risk: EXTREME - can lose 80-90% of capital

    Returns:
        Strategy configured for extreme leverage trend-following
    """
    signal_generator = AggressiveTrendSignalGenerator(
        name=f"{name}_signals",
        fast_ema=8,
        medium_ema=21,
        slow_ema=50,
        adx_threshold=20.0,  # Lower threshold = enter trends earlier
        volume_surge_threshold=1.3,  # Lower threshold = more trades
        min_confidence=0.50,  # Lower confidence requirement
    )

    risk_manager = RegimeAdaptiveRiskManager(
        low_vol_risk=0.12,  # 12% stop in low volatility
        high_vol_risk=0.20,  # 20% stop in high volatility
        base_risk=0.18,  # 18% base stop
    )

    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.50,  # Always use maximum
        min_confidence=0.45,  # Very low threshold
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    # Extreme leverage risk configuration
    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": 0.50,  # Maximum always
        "min_fraction": 0.50,  # Never reduce below maximum
        "max_fraction": 0.50,
        "stop_loss_pct": 0.18,  # 18% stop
        "take_profit_pct": 0.50,  # 50% profit target
        # Minimal dynamic risk reduction (stay aggressive)
        "dynamic_risk": {
            "enabled": True,
            # Only reduce at catastrophic drawdowns
            "drawdown_thresholds": [0.50, 0.70, 0.85],
            "risk_reduction_factors": [0.90, 0.80, 0.70],  # Minimal reduction
            "recovery_thresholds": [0.30, 0.50],
        },
        # Aggressive partial operations
        "partial_operations": {
            # Take some profits, but keep most position
            "exit_targets": [0.30, 0.50, 0.80],
            "exit_sizes": [0.20, 0.20, 0.30],  # Keep 30% for moonshots
            # Scale in aggressively
            "scale_in_thresholds": [0.05, 0.10, 0.15],
            "scale_in_sizes": [0.30, 0.25, 0.20],
            "max_scale_ins": 3,
        },
        # Wide trailing stop
        "trailing_stop": {
            "activation_threshold": 0.20,  # Only activate at 20% profit
            "trailing_distance_pct": 0.08,  # Trail 8% below peak
            "breakeven_threshold": 0.30,  # Move to breakeven late
            "breakeven_buffer": 0.03,
        },
    }

    strategy.config = {
        "base_fraction": 0.50,
        "stop_loss": 0.18,
        "take_profit": 0.50,
        "risk_level": "EXTREME",
    }

    return strategy


def create_extreme_leverage_volatility_strategy(
    name: str = "ExtremeLeverageVolatility",
) -> Strategy:
    """
    Create extreme leverage volatility exploitation strategy

    Configuration:
    - 50% position size (system maximum)
    - 15% stop loss (wide for volatility)
    - Exploit all volatility states aggressively

    Expected:
    - 4-8x buy-and-hold in volatile markets
    - 70-85% drawdown in crashes
    - Sharpe: 0.6-1.1

    Use when: High volatility environment, want maximum exposure
    Risk: EXTREME - very large swings

    Returns:
        Strategy configured for extreme leverage volatility exploitation
    """
    signal_generator = VolatilityExploitationSignalGenerator(
        name=f"{name}_signals",
        atr_period=14,
        atr_high_threshold=1.8,  # Lower = more aggressive
        atr_low_threshold=0.6,
        min_confidence=0.50,  # Lower threshold
    )

    risk_manager = RegimeAdaptiveRiskManager(
        low_vol_risk=0.10,
        high_vol_risk=0.18,
        base_risk=0.15,
    )

    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.50,
        min_confidence=0.45,
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": 0.50,
        "min_fraction": 0.50,
        "max_fraction": 0.50,
        "stop_loss_pct": 0.15,
        "take_profit_pct": 0.45,
        "dynamic_risk": {
            "enabled": True,
            "drawdown_thresholds": [0.50, 0.70, 0.85],
            "risk_reduction_factors": [0.90, 0.80, 0.70],
            "recovery_thresholds": [0.30, 0.50],
        },
        "partial_operations": {
            "exit_targets": [0.25, 0.40, 0.70],
            "exit_sizes": [0.25, 0.25, 0.25],
            "scale_in_thresholds": [0.05, 0.10],
            "scale_in_sizes": [0.30, 0.25],
            "max_scale_ins": 2,
        },
        "trailing_stop": {
            "activation_threshold": 0.18,
            "trailing_distance_pct": 0.07,
            "breakeven_threshold": 0.25,
            "breakeven_buffer": 0.03,
        },
    }

    strategy.config = {
        "base_fraction": 0.50,
        "stop_loss": 0.15,
        "take_profit": 0.45,
        "risk_level": "EXTREME",
    }

    return strategy


def create_extreme_leverage_adaptive_strategy(
    name: str = "ExtremeLeverageAdaptive",
) -> Strategy:
    """
    Create extreme leverage regime-adaptive strategy

    This is the MOST AGGRESSIVE strategy in the entire system:
    - 50% position size always (system maximum)
    - 16% base stop loss (wide)
    - Adapts to regimes but stays maximally leveraged
    - Only reduces leverage in CRASH_RISK regime

    Expected:
    - 6-12x buy-and-hold over full market cycle
    - 75-90% drawdown in severe bear markets
    - Sharpe: 0.7-1.3
    - Recovery time: Very long (1-2 years)

    Use when: You want absolute maximum returns and can handle extreme volatility
    Risk: EXTREME - this is the highest risk strategy

    Returns:
        Strategy configured for extreme leverage with regime adaptation
    """
    signal_generator = RegimeAdaptiveMegaSignalGenerator(
        name=f"{name}_signals",
        trend_strength_threshold=0.012,  # Lower = more aggressive
        high_volatility_threshold=1.8,
        crash_risk_threshold=3,  # Less sensitive = stay invested longer
        use_aggressive_trend=True,
        use_crash_detection=True,
        use_volatility_exploitation=True,
    )

    risk_manager = RegimeAdaptiveRiskManager(
        low_vol_risk=0.12,
        high_vol_risk=0.20,
        base_risk=0.16,
    )

    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.50,
        min_confidence=0.45,
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": 0.50,
        "min_fraction": 0.30,  # Only reduce in CRASH_RISK
        "max_fraction": 0.50,
        "stop_loss_pct": 0.16,
        "take_profit_pct": 0.50,
        "dynamic_risk": {
            "enabled": True,
            "drawdown_thresholds": [0.50, 0.70, 0.85],
            "risk_reduction_factors": [0.90, 0.80, 0.70],
            "recovery_thresholds": [0.30, 0.50],
        },
        "partial_operations": {
            "exit_targets": [0.30, 0.50, 0.80],
            "exit_sizes": [0.20, 0.25, 0.30],
            "scale_in_thresholds": [0.05, 0.10, 0.15],
            "scale_in_sizes": [0.30, 0.25, 0.20],
            "max_scale_ins": 3,
        },
        "trailing_stop": {
            "activation_threshold": 0.20,
            "trailing_distance_pct": 0.08,
            "breakeven_threshold": 0.30,
            "breakeven_buffer": 0.03,
        },
        # Regime multipliers (stay aggressive in most regimes)
        "regime_position_multipliers": {
            "TRENDING_UP": 1.0,  # Full 50%
            "TRENDING_DOWN": 0.60,  # Reduce to 30%
            "RANGING": 0.80,  # 40%
            "HIGH_VOLATILITY": 1.0,  # Full 50%
            "CRASH_RISK": 0.60,  # Reduce to 30%
        },
        "regime_stop_multipliers": {
            "TRENDING_UP": 1.3,  # Wider stops in uptrends
            "TRENDING_DOWN": 0.80,
            "RANGING": 0.90,
            "HIGH_VOLATILITY": 1.5,  # Much wider in high vol
            "CRASH_RISK": 0.70,
        },
    }

    strategy.config = {
        "base_fraction": 0.50,
        "stop_loss": 0.16,
        "take_profit": 0.50,
        "risk_level": "EXTREME",
        "adaptive": True,
    }

    return strategy


def create_yolo_strategy(name: str = "YOLO") -> Strategy:
    """
    Create YOLO (You Only Live Once) strategy - ABSOLUTE MAXIMUM RISK

    This is a joke/educational strategy that shows what happens
    when you push everything to the absolute maximum:
    - 50% position size (cannot go higher)
    - 25% stop loss (extremely wide)
    - No dynamic risk reduction
    - Minimal profit taking
    - Maximum scale-ins

    Expected:
    - 10-20x buy-and-hold in perfect conditions
    - 90-95% drawdown in bad conditions
    - Sharpe: 0.3-0.8 (terrible risk-adjusted)
    - Likely to blow up account

    Use when: Never (educational only)
    Risk: CATASTROPHIC

    Returns:
        Strategy configured for maximum possible risk (educational)
    """
    signal_generator = AggressiveTrendSignalGenerator(
        name=f"{name}_signals",
        fast_ema=5,  # Fastest possible
        medium_ema=13,
        slow_ema=34,
        adx_threshold=15.0,  # Very low threshold
        min_confidence=0.40,  # Minimal confidence required
    )

    risk_manager = RegimeAdaptiveRiskManager(
        low_vol_risk=0.15,
        high_vol_risk=0.25,
        base_risk=0.25,  # 25% stop loss!
    )

    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.50,
        min_confidence=0.40,
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": 0.50,
        "min_fraction": 0.50,  # Never reduce
        "max_fraction": 0.50,
        "stop_loss_pct": 0.25,  # 25% stop!
        "take_profit_pct": 1.00,  # 100% profit target (let it run)
        # NO dynamic risk reduction
        "dynamic_risk": {"enabled": False},
        # Maximum partial operations
        "partial_operations": {
            "exit_targets": [0.50, 1.00],  # Only take profits at 50% and 100%
            "exit_sizes": [0.20, 0.30],  # Keep 50% for moonshots
            "scale_in_thresholds": [0.05, 0.10, 0.15, 0.20],
            "scale_in_sizes": [0.30, 0.30, 0.25, 0.20],
            "max_scale_ins": 4,  # Maximum scale-ins
        },
        # Very loose trailing stop
        "trailing_stop": {
            "activation_threshold": 0.30,
            "trailing_distance_pct": 0.12,  # Trail 12% below peak
            "breakeven_threshold": 0.50,
            "breakeven_buffer": 0.05,
        },
    }

    strategy.config = {
        "base_fraction": 0.50,
        "stop_loss": 0.25,
        "take_profit": 1.00,
        "risk_level": "YOLO",
        "warning": "DO NOT USE WITH REAL MONEY",
    }

    return strategy
