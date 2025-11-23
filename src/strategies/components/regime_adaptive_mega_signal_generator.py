"""
Regime-Adaptive Signal Generator

This is the "mega" signal generator that combines multiple signal generation
strategies and adapts its behavior based on detected market regime.

Regime Detection:
1. TRENDING_UP: Strong uptrend, use aggressive trend-following
2. TRENDING_DOWN: Strong downtrend, use crash detection + exits
3. RANGING: Sideways market, reduce activity or mean reversion
4. HIGH_VOLATILITY: Volatile conditions, use volatility exploitation
5. CRASH_RISK: Elevated crash indicators, defensive positioning

Meta-Strategy Approach:
- Detect current regime using multiple indicators
- Select appropriate signal generation logic for regime
- Adapt position sizing, stops, and targets based on regime
- Smoothly transition between regimes (avoid whipsaws)

This is the "Swiss Army knife" strategy that should perform well
across all market conditions by adapting its approach.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .aggressive_trend_signal_generator import AggressiveTrendSignalGenerator
from .crash_detection_signal_generator import CrashDetectionSignalGenerator
from .regime_context import RegimeContext, TrendLabel, VolLabel
from .signal_generator import Signal, SignalDirection, SignalGenerator
from .volatility_exploitation_signal_generator import VolatilityExploitationSignalGenerator


class RegimeAdaptiveMegaSignalGenerator(SignalGenerator):
    """
    Meta signal generator that adapts to market regimes

    This generator combines multiple signal generation strategies
    and selects the most appropriate one based on current market regime.
    """

    def __init__(
        self,
        name: str = "regime_adaptive_mega_signals",
        # Regime detection thresholds
        trend_strength_threshold: float = 0.015,  # 1.5% momentum for trending
        high_volatility_threshold: float = 2.0,  # 2x ATR for high volatility
        crash_risk_threshold: int = 2,  # 2+ crash indicators
        # Signal generator instances
        use_aggressive_trend: bool = True,
        use_crash_detection: bool = True,
        use_volatility_exploitation: bool = True,
    ):
        """
        Initialize regime-adaptive mega signal generator

        Args:
            name: Signal generator name
            trend_strength_threshold: Momentum threshold for trending regime
            high_volatility_threshold: ATR ratio for high volatility
            crash_risk_threshold: Number of crash indicators for crash regime
            use_aggressive_trend: Enable aggressive trend component
            use_crash_detection: Enable crash detection component
            use_volatility_exploitation: Enable volatility exploitation component
        """
        super().__init__(name)
        self.trend_strength_threshold = trend_strength_threshold
        self.high_volatility_threshold = high_volatility_threshold
        self.crash_risk_threshold = crash_risk_threshold

        # Create component signal generators
        self.trend_generator = (
            AggressiveTrendSignalGenerator(name="trend_component")
            if use_aggressive_trend
            else None
        )
        self.crash_generator = (
            CrashDetectionSignalGenerator(name="crash_component")
            if use_crash_detection
            else None
        )
        self.volatility_generator = (
            VolatilityExploitationSignalGenerator(name="volatility_component")
            if use_volatility_exploitation
            else None
        )

    @property
    def warmup_period(self) -> int:
        """Return maximum warmup period from all components"""
        periods = [0]
        if self.trend_generator:
            periods.append(self.trend_generator.warmup_period)
        if self.crash_generator:
            periods.append(self.crash_generator.warmup_period)
        if self.volatility_generator:
            periods.append(self.volatility_generator.warmup_period)
        return max(periods)

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None
    ) -> Signal:
        """
        Generate regime-adaptive signal

        Signal Selection Logic:
        1. Detect current market regime
        2. Select primary signal generator for regime
        3. Get signal from selected generator
        4. Potentially blend with secondary generators
        5. Return adapted signal with regime metadata

        Args:
            df: DataFrame with OHLCV data
            index: Current candle index
            regime: Optional regime context (will detect if not provided)

        Returns:
            Signal adapted to current market regime
        """
        self.validate_inputs(df, index)

        # Detect market regime if not provided
        detected_regime = self._detect_market_regime(df, index, regime)

        # Select appropriate signal generator based on regime
        primary_generator, regime_name = self._select_primary_generator(detected_regime, df, index)

        # Generate signal from primary generator
        if primary_generator:
            signal = primary_generator.generate_signal(df, index, regime)
        else:
            # Fallback to HOLD if no generator selected
            signal = Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.5,
                metadata={"generator": self.name, "regime": regime_name, "no_generator": True},
            )

        # Enhance signal metadata with regime information
        signal.metadata.update(
            {
                "meta_generator": self.name,
                "detected_regime": regime_name,
                "primary_generator": primary_generator.name if primary_generator else None,
                "regime_trend": detected_regime["trend"],
                "regime_volatility": detected_regime["volatility"],
                "crash_risk_level": detected_regime["crash_risk"],
            }
        )

        return signal

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence based on regime clarity"""
        self.validate_inputs(df, index)
        detected_regime = self._detect_market_regime(df, index, None)

        # High confidence when regime is clear
        if detected_regime["regime_clarity"] > 0.7:
            return 0.75
        elif detected_regime["regime_clarity"] > 0.5:
            return 0.65
        else:
            return 0.50

    def _detect_market_regime(
        self, df: pd.DataFrame, index: int, regime_context: Optional[RegimeContext]
    ) -> dict:
        """
        Detect current market regime using multiple indicators

        Returns dict with:
        - trend: "TRENDING_UP", "TRENDING_DOWN", "RANGING"
        - volatility: "HIGH", "MEDIUM", "LOW"
        - crash_risk: 0-5 (number of crash indicators active)
        - regime_clarity: 0-1 (how clear the regime is)
        """
        # Calculate momentum for trend detection
        momentum = self._calculate_momentum(df, index, period=20)

        # Determine trend
        if momentum > self.trend_strength_threshold:
            trend = "TRENDING_UP"
        elif momentum < -self.trend_strength_threshold:
            trend = "TRENDING_DOWN"
        else:
            trend = "RANGING"

        # Calculate volatility
        atr_ratio = self._calculate_atr_ratio(df, index)
        if atr_ratio > self.high_volatility_threshold:
            volatility = "HIGH"
        elif atr_ratio < 0.7:
            volatility = "LOW"
        else:
            volatility = "MEDIUM"

        # Check crash risk if generator available
        crash_risk = 0
        if self.crash_generator:
            crash_indicators = self.crash_generator._detect_crash_indicators(df, index)
            crash_risk = sum(
                [
                    crash_indicators["rsi_divergence"],
                    crash_indicators["volume_panic"],
                    crash_indicators["volatility_expansion"],
                    crash_indicators["momentum_breakdown"],
                    crash_indicators["waterfall_decline"],
                ]
            )

        # Calculate regime clarity (how confident we are)
        clarity_factors = []

        # Trend clarity
        trend_clarity = min(abs(momentum) / 0.05, 1.0)  # Normalize to 0-1
        clarity_factors.append(trend_clarity)

        # Volatility clarity
        vol_clarity = abs(atr_ratio - 1.0)  # Distance from normal
        clarity_factors.append(min(vol_clarity, 1.0))

        regime_clarity = np.mean(clarity_factors)

        return {
            "trend": trend,
            "volatility": volatility,
            "crash_risk": crash_risk,
            "regime_clarity": regime_clarity,
            "momentum": momentum,
            "atr_ratio": atr_ratio,
        }

    def _select_primary_generator(
        self, detected_regime: dict, df: pd.DataFrame, index: int
    ) -> tuple[Optional[SignalGenerator], str]:
        """
        Select primary signal generator based on detected regime

        Selection Priority:
        1. CRASH_RISK (highest priority) - Preserve capital
        2. HIGH_VOLATILITY - Exploit volatility
        3. TRENDING - Follow the trend
        4. RANGING - Reduce activity or mean reversion
        """
        trend = detected_regime["trend"]
        volatility = detected_regime["volatility"]
        crash_risk = detected_regime["crash_risk"]

        # Priority 1: Crash risk detected
        if crash_risk >= self.crash_risk_threshold and self.crash_generator:
            return self.crash_generator, f"CRASH_RISK_{crash_risk}"

        # Priority 2: High volatility
        if volatility == "HIGH" and self.volatility_generator:
            return self.volatility_generator, f"HIGH_VOLATILITY_{trend}"

        # Priority 3: Trending market
        if trend in ["TRENDING_UP", "TRENDING_DOWN"] and self.trend_generator:
            return self.trend_generator, f"TREND_{trend}"

        # Priority 4: Ranging market with high volatility
        if trend == "RANGING" and volatility == "HIGH" and self.volatility_generator:
            return self.volatility_generator, "RANGING_VOLATILE"

        # Priority 5: Ranging market with medium/low volatility
        if trend == "RANGING":
            # In ranging market with low volatility, reduce activity
            return None, "RANGING_CALM"

        # Default: Use trend generator if available
        if self.trend_generator:
            return self.trend_generator, f"DEFAULT_{trend}"

        return None, "NO_REGIME"

    def _calculate_momentum(self, df: pd.DataFrame, index: int, period: int = 20) -> float:
        """Calculate price momentum"""
        if index < period:
            return 0.0

        current_price = df["close"].iloc[index]
        past_price = df["close"].iloc[index - period]

        if past_price == 0:
            return 0.0

        return (current_price - past_price) / past_price

    def _calculate_atr_ratio(self, df: pd.DataFrame, index: int) -> float:
        """Calculate ATR ratio (current vs historical average)"""
        if index < 30:
            return 1.0

        # Current ATR (14 period)
        current_atr = self._calculate_atr(df, index, 14)

        # Historical ATR (30 period average)
        hist_atr = self._calculate_atr(df, index - 14, 30)

        if hist_atr == 0:
            return 1.0

        return current_atr / hist_atr

    def _calculate_atr(self, df: pd.DataFrame, index: int, period: int) -> float:
        """Calculate Average True Range"""
        if index < period:
            return 0.0

        high = df["high"].iloc[max(0, index - period) : index + 1].values
        low = df["low"].iloc[max(0, index - period) : index + 1].values
        close = df["close"].iloc[max(0, index - period) : index + 1].values

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        return np.mean(tr)

    def get_feature_generators(self) -> Sequence:
        """No additional features required"""
        return []
