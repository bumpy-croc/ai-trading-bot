"""
Aggressive Trend-Following Signal Generator

This signal generator implements aggressive trend detection and entry logic
designed to capture major trend moves with maximum leverage. It uses multiple
confirmation signals to ensure high-probability trend entries.

Key Features:
- Multiple EMA crossovers for trend detection (8, 21, 50 period)
- ADX for trend strength measurement (>25 = trending)
- Volume surge detection for momentum confirmation
- Lower confidence threshold for early entries
- Higher signal strength in strong trending conditions

Design Philosophy:
Beat buy-and-hold by catching trends early and riding them aggressively with
high position sizes. Accept higher whipsaws in exchange for capturing full
trend moves with leverage.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .signal_generator import Signal, SignalDirection, SignalGenerator
from .regime_context import RegimeContext, TrendLabel


class AggressiveTrendSignalGenerator(SignalGenerator):
    """
    Generate signals for aggressive trend-following strategy

    Uses multiple technical indicators to detect strong trends early
    and generate high-confidence signals for leveraged position entry.
    """

    def __init__(
        self,
        name: str = "aggressive_trend_signals",
        fast_ema: int = 8,  # Fast EMA period
        medium_ema: int = 21,  # Medium EMA period
        slow_ema: int = 50,  # Slow EMA period
        adx_period: int = 14,  # ADX trend strength period
        adx_threshold: float = 25.0,  # Minimum ADX for trending market
        volume_surge_multiplier: float = 1.5,  # Volume surge detection
        min_confidence: float = 0.55,  # Lower threshold for aggressive entry
    ):
        """
        Initialize aggressive trend signal generator

        Args:
            name: Signal generator name
            fast_ema: Fast EMA period (default: 8)
            medium_ema: Medium EMA period (default: 21)
            slow_ema: Slow EMA period (default: 50)
            adx_period: ADX calculation period (default: 14)
            adx_threshold: Minimum ADX for trend confirmation (default: 25)
            volume_surge_multiplier: Volume surge threshold (default: 1.5x)
            min_confidence: Minimum confidence for signal generation (default: 0.55)
        """
        super().__init__(name)
        self.fast_ema = fast_ema
        self.medium_ema = medium_ema
        self.slow_ema = slow_ema
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_surge_multiplier = volume_surge_multiplier
        self.min_confidence = min_confidence

    @property
    def warmup_period(self) -> int:
        """Return minimum history required"""
        return max(self.slow_ema, self.adx_period) + 5

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None
    ) -> Signal:
        """
        Generate aggressive trend-following signal

        Signal Logic:
        1. BUY when:
           - Fast EMA > Medium EMA > Slow EMA (bullish alignment)
           - ADX > threshold (strong trend)
           - Volume surge detected (momentum confirmation)
           - Price > all EMAs (trend strength)

        2. SELL when:
           - Fast EMA < Medium EMA (trend reversal)
           - ADX declining (weakening trend)

        3. HOLD otherwise

        Args:
            df: DataFrame with OHLCV and indicators
            index: Current candle index
            regime: Optional regime context

        Returns:
            Signal with direction, strength, and confidence
        """
        self.validate_inputs(df, index)

        # Calculate indicators
        indicators = self._calculate_indicators(df, index)

        # Detect trend alignment
        trend_aligned = self._detect_trend_alignment(indicators)

        # Check trend strength (ADX)
        strong_trend = indicators["adx"] > self.adx_threshold

        # Check volume surge
        volume_surge = indicators["volume_ratio"] > self.volume_surge_multiplier

        # Price above all EMAs (bullish)
        price_above_emas = (
            indicators["price"] > indicators["fast_ema"]
            and indicators["price"] > indicators["medium_ema"]
            and indicators["price"] > indicators["slow_ema"]
        )

        # Generate signal
        if trend_aligned and strong_trend:
            # Strong bullish trend - BUY signal
            direction = SignalDirection.BUY

            # Calculate signal strength (0-1)
            strength = self._calculate_strength(indicators, volume_surge, price_above_emas)

            # Calculate confidence (0-1)
            confidence = self._calculate_confidence(indicators, regime)

            # Only generate BUY if confidence exceeds threshold
            if confidence < self.min_confidence:
                direction = SignalDirection.HOLD
                strength = 0.0
                confidence = 0.5

        elif not trend_aligned or indicators["adx"] < self.adx_threshold * 0.8:
            # Trend weakening or reversing - SELL signal
            direction = SignalDirection.SELL
            strength = 0.7
            confidence = 0.8

        else:
            # No clear signal - HOLD
            direction = SignalDirection.HOLD
            strength = 0.0
            confidence = 0.5

        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                "generator": self.name,
                "index": index,
                "fast_ema": indicators["fast_ema"],
                "medium_ema": indicators["medium_ema"],
                "slow_ema": indicators["slow_ema"],
                "adx": indicators["adx"],
                "volume_ratio": indicators["volume_ratio"],
                "trend_aligned": trend_aligned,
                "strong_trend": strong_trend,
                "volume_surge": volume_surge,
                "price_above_emas": price_above_emas,
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence score for current market conditions"""
        self.validate_inputs(df, index)
        indicators = self._calculate_indicators(df, index)
        return self._calculate_confidence(indicators, regime=None)

    def _calculate_indicators(self, df: pd.DataFrame, index: int) -> dict:
        """Calculate all required indicators"""
        # EMAs
        fast_ema = df["close"].ewm(span=self.fast_ema, adjust=False).mean().iloc[index]
        medium_ema = df["close"].ewm(span=self.medium_ema, adjust=False).mean().iloc[index]
        slow_ema = df["close"].ewm(span=self.slow_ema, adjust=False).mean().iloc[index]

        # ADX calculation
        adx = self._calculate_adx(df, index)

        # Volume ratio (current volume vs 20-period average)
        volume_avg = df["volume"].rolling(20).mean().iloc[index]
        volume_ratio = df["volume"].iloc[index] / volume_avg if volume_avg > 0 else 1.0

        # Current price
        price = df["close"].iloc[index]

        return {
            "price": price,
            "fast_ema": fast_ema,
            "medium_ema": medium_ema,
            "slow_ema": slow_ema,
            "adx": adx,
            "volume_ratio": volume_ratio,
        }

    def _calculate_adx(self, df: pd.DataFrame, index: int) -> float:
        """Calculate ADX (Average Directional Index) for trend strength"""
        period = self.adx_period

        # Need enough history
        if index < period + 1:
            return 0.0

        # Get recent candles
        high = df["high"].iloc[max(0, index - period) : index + 1].values
        low = df["low"].iloc[max(0, index - period) : index + 1].values
        close = df["close"].iloc[max(0, index - period) : index + 1].values

        # Calculate +DM and -DM
        high_diff = np.diff(high)
        low_diff = -np.diff(low)

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Smooth with EMA
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1]
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().iloc[-1] / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().iloc[-1] / atr

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        return dx

    def _detect_trend_alignment(self, indicators: dict) -> bool:
        """
        Detect if EMAs are aligned for strong trend

        Bullish alignment: Fast > Medium > Slow
        """
        return (
            indicators["fast_ema"] > indicators["medium_ema"]
            and indicators["medium_ema"] > indicators["slow_ema"]
        )

    def _calculate_strength(
        self, indicators: dict, volume_surge: bool, price_above_emas: bool
    ) -> float:
        """
        Calculate signal strength (0-1)

        Factors:
        - EMA separation (wider = stronger)
        - ADX value (higher = stronger)
        - Volume surge (adds strength)
        - Price position relative to EMAs
        """
        # EMA separation (normalized)
        ema_separation = (
            (indicators["fast_ema"] - indicators["slow_ema"]) / indicators["slow_ema"]
            if indicators["slow_ema"] > 0
            else 0
        )
        ema_strength = min(abs(ema_separation) * 20, 1.0)  # Normalize to 0-1

        # ADX strength (normalized)
        adx_strength = min(indicators["adx"] / 50.0, 1.0)  # 50+ ADX = max strength

        # Volume bonus
        volume_bonus = 0.2 if volume_surge else 0.0

        # Price position bonus
        price_bonus = 0.15 if price_above_emas else 0.0

        # Combined strength
        base_strength = (ema_strength * 0.4) + (adx_strength * 0.4)
        total_strength = min(base_strength + volume_bonus + price_bonus, 1.0)

        return total_strength

    def _calculate_confidence(self, indicators: dict, regime: Optional[RegimeContext]) -> float:
        """
        Calculate signal confidence (0-1)

        Factors:
        - ADX above threshold (trend confirmation)
        - Volume confirmation
        - Regime alignment (if available)
        """
        confidence = 0.5  # Base confidence

        # ADX confidence boost
        if indicators["adx"] > self.adx_threshold:
            adx_boost = min((indicators["adx"] - self.adx_threshold) / 25.0, 0.25)
            confidence += adx_boost

        # Volume confidence boost
        if indicators["volume_ratio"] > self.volume_surge_multiplier:
            confidence += 0.15

        # Regime alignment boost
        if regime and regime.trend == TrendLabel.TRENDING_UP:
            confidence += 0.10

        return min(confidence, 1.0)

    def get_feature_generators(self) -> Sequence:
        """No additional features required - uses standard OHLCV"""
        return []
