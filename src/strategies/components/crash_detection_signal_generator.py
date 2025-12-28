"""
Crash Detection Signal Generator

This signal generator detects early warning signs of market crashes and
generates SELL signals to exit positions before major drawdowns occur.

Crash Detection Methodology:
1. RSI Divergence: Price making new highs while RSI makes lower highs
2. Volume Spike: Extreme volume suggests panic selling or distribution
3. Volatility Expansion: ATR breakout indicates instability
4. Momentum Breakdown: Multiple timeframe weakness
5. Sentiment Extreme: Extreme fear/greed precedes reversals

Historical Crashes Analyzed:
- 2018 Crash: BTC $20k → $3k (85% drawdown)
- COVID Mar 2020: BTC $10k → $4k (60% drawdown)
- May 2021: BTC $65k → $30k (54% drawdown)
- 2022 LUNA/FTX: BTC $48k → $16k (66% drawdown)

Design Goal:
Exit before losing more than 20% from peak, allowing buy-and-hold to
suffer 60-80% drawdowns while we preserve capital.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .signal_generator import Signal, SignalDirection, SignalGenerator
from .regime_context import RegimeContext


class CrashDetectionSignalGenerator(SignalGenerator):
    """
    Detect market crash conditions and generate early exit signals

    This generator monitors multiple crash indicators and generates
    SELL signals when crash risk is elevated, allowing capital
    preservation before major drawdowns.
    """

    def __init__(
        self,
        name: str = "crash_detection_signals",
        # RSI divergence parameters
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        divergence_lookback: int = 20,
        # Volume parameters
        volume_spike_threshold: float = 3.0,  # 3x average = panic
        volume_lookback: int = 20,
        # Volatility parameters
        atr_period: int = 14,
        atr_expansion_threshold: float = 2.0,  # 2x normal = instability
        # Momentum parameters
        momentum_period: int = 10,
        momentum_threshold: float = -0.05,  # -5% momentum = weakness
        # Price action parameters
        waterfall_threshold: float = -0.15,  # -15% rapid drop = crash
        waterfall_lookback: int = 5,  # 5 candles
        # Crash confidence
        min_indicators: int = 2,  # Require 2+ indicators for crash signal
    ):
        """
        Initialize crash detection signal generator

        Args:
            name: Signal generator name
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold (potential top)
            divergence_lookback: Candles to check for divergence
            volume_spike_threshold: Volume spike multiplier (panic threshold)
            volume_lookback: Volume average calculation period
            atr_period: ATR calculation period
            atr_expansion_threshold: ATR expansion multiplier
            momentum_period: Momentum calculation period
            momentum_threshold: Momentum weakness threshold
            waterfall_threshold: Rapid price drop threshold
            waterfall_lookback: Lookback for waterfall detection
            min_indicators: Minimum crash indicators required
        """
        super().__init__(name)
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.divergence_lookback = divergence_lookback
        self.volume_spike_threshold = volume_spike_threshold
        self.volume_lookback = volume_lookback
        self.atr_period = atr_period
        self.atr_expansion_threshold = atr_expansion_threshold
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.waterfall_threshold = waterfall_threshold
        self.waterfall_lookback = waterfall_lookback
        self.min_indicators = min_indicators

    @property
    def warmup_period(self) -> int:
        """Return minimum history required"""
        return max(self.divergence_lookback, self.atr_period, self.momentum_period) + 10

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None
    ) -> Signal:
        """
        Generate crash detection signal

        Signal Logic:
        - SELL when 2+ crash indicators detected
        - BUY when recovery conditions met (stabilization)
        - HOLD otherwise

        Args:
            df: DataFrame with OHLCV data
            index: Current candle index
            regime: Optional regime context

        Returns:
            Signal with crash risk assessment
        """
        self.validate_inputs(df, index)

        # Detect individual crash indicators
        indicators = self._detect_crash_indicators(df, index)

        # Count active crash indicators
        crash_score = sum(
            [
                indicators["rsi_divergence"],
                indicators["volume_panic"],
                indicators["volatility_expansion"],
                indicators["momentum_breakdown"],
                indicators["waterfall_decline"],
            ]
        )

        # Generate signal based on crash score
        if crash_score >= self.min_indicators:
            # CRASH DETECTED - Exit immediately
            direction = SignalDirection.SELL
            strength = min(crash_score / 5.0, 1.0)  # Normalize to 0-1
            confidence = 0.7 + (crash_score * 0.06)  # Higher confidence with more indicators

        elif indicators["recovery_signal"]:
            # RECOVERY DETECTED - Consider re-entry
            direction = SignalDirection.BUY
            strength = 0.6
            confidence = 0.65

        else:
            # NO CLEAR SIGNAL
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
                "crash_score": crash_score,
                "rsi_divergence": indicators["rsi_divergence"],
                "volume_panic": indicators["volume_panic"],
                "volatility_expansion": indicators["volatility_expansion"],
                "momentum_breakdown": indicators["momentum_breakdown"],
                "waterfall_decline": indicators["waterfall_decline"],
                "recovery_signal": indicators["recovery_signal"],
                "rsi": indicators["rsi"],
                "volume_ratio": indicators["volume_ratio"],
                "atr_ratio": indicators["atr_ratio"],
                "momentum": indicators["momentum"],
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence in current market conditions"""
        self.validate_inputs(df, index)
        indicators = self._detect_crash_indicators(df, index)
        crash_score = sum(
            [
                indicators["rsi_divergence"],
                indicators["volume_panic"],
                indicators["volatility_expansion"],
                indicators["momentum_breakdown"],
                indicators["waterfall_decline"],
            ]
        )
        return 0.5 + (crash_score * 0.1)

    def _detect_crash_indicators(self, df: pd.DataFrame, index: int) -> dict:
        """Detect all crash indicators"""
        # Calculate RSI
        rsi = self._calculate_rsi(df, index)
        rsi_divergence = self._detect_rsi_divergence(df, index, rsi)

        # Calculate volume metrics
        volume_ratio = self._calculate_volume_ratio(df, index)
        volume_panic = volume_ratio > self.volume_spike_threshold

        # Calculate volatility metrics
        atr_ratio = self._calculate_atr_expansion(df, index)
        volatility_expansion = atr_ratio > self.atr_expansion_threshold

        # Calculate momentum
        momentum = self._calculate_momentum(df, index)
        momentum_breakdown = momentum < self.momentum_threshold

        # Detect waterfall decline
        waterfall_decline = self._detect_waterfall(df, index)

        # Detect recovery
        recovery_signal = self._detect_recovery(df, index, rsi, volume_ratio)

        return {
            "rsi": rsi,
            "rsi_divergence": rsi_divergence,
            "volume_ratio": volume_ratio,
            "volume_panic": volume_panic,
            "atr_ratio": atr_ratio,
            "volatility_expansion": volatility_expansion,
            "momentum": momentum,
            "momentum_breakdown": momentum_breakdown,
            "waterfall_decline": waterfall_decline,
            "recovery_signal": recovery_signal,
        }

    def _calculate_rsi(self, df: pd.DataFrame, index: int) -> float:
        """Calculate RSI indicator"""
        if index < self.rsi_period:
            return 50.0

        close = df["close"].iloc[max(0, index - self.rsi_period) : index + 1].values
        deltas = np.diff(close)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _detect_rsi_divergence(self, df: pd.DataFrame, index: int, current_rsi: float) -> bool:
        """
        Detect bearish RSI divergence

        Divergence: Price making new highs while RSI makes lower highs
        This suggests weakening momentum and potential reversal
        """
        if index < self.divergence_lookback:
            return False

        # Get recent prices and RSI values
        lookback_start = max(0, index - self.divergence_lookback)
        recent_prices = df["close"].iloc[lookback_start : index + 1].values

        # Calculate RSI for lookback period
        recent_rsi = []
        for i in range(lookback_start, index + 1):
            recent_rsi.append(self._calculate_rsi(df, i))

        # Check if current price is near recent high
        max_price = np.max(recent_prices)
        current_price = recent_prices[-1]
        price_near_high = current_price >= max_price * 0.98  # Within 2% of high

        # Check if RSI is below recent high
        max_rsi = np.max(recent_rsi[:-1])  # Exclude current
        rsi_below_high = current_rsi < max_rsi * 0.95  # 5% below RSI high

        # Divergence detected
        return price_near_high and rsi_below_high and current_rsi > self.rsi_overbought

    def _calculate_volume_ratio(self, df: pd.DataFrame, index: int) -> float:
        """Calculate current volume vs average"""
        if index < self.volume_lookback:
            return 1.0

        recent_volume = df["volume"].iloc[max(0, index - self.volume_lookback) : index].values
        avg_volume = np.mean(recent_volume)

        if avg_volume == 0:
            return 1.0

        current_volume = df["volume"].iloc[index]
        return current_volume / avg_volume

    def _calculate_atr_expansion(self, df: pd.DataFrame, index: int) -> float:
        """Calculate ATR expansion ratio"""
        if index < self.atr_period * 2:
            return 1.0

        # Current ATR
        current_atr = self._calculate_atr(df, index, self.atr_period)

        # Historical ATR (longer lookback)
        hist_atr = self._calculate_atr(df, index - self.atr_period, self.atr_period)

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

    def _calculate_momentum(self, df: pd.DataFrame, index: int) -> float:
        """Calculate price momentum"""
        if index < self.momentum_period:
            return 0.0

        current_price = df["close"].iloc[index]
        past_price = df["close"].iloc[index - self.momentum_period]

        if past_price == 0:
            return 0.0

        return (current_price - past_price) / past_price

    def _detect_waterfall(self, df: pd.DataFrame, index: int) -> bool:
        """Detect rapid waterfall decline (flash crash)"""
        if index < self.waterfall_lookback:
            return False

        current_price = df["close"].iloc[index]
        past_price = df["close"].iloc[index - self.waterfall_lookback]

        if past_price == 0:
            return False

        decline = (current_price - past_price) / past_price
        return decline < self.waterfall_threshold

    def _detect_recovery(
        self, df: pd.DataFrame, index: int, rsi: float, volume_ratio: float
    ) -> bool:
        """
        Detect recovery/stabilization conditions

        Recovery signals:
        - RSI oversold (<30) - capitulation
        - Volume decreasing - panic subsiding
        - Price stabilizing - finding support
        """
        if index < 10:
            return False

        # RSI oversold (capitulation)
        rsi_oversold = rsi < 30

        # Volume normalizing (panic over)
        volume_normalizing = volume_ratio < 1.5

        # Price stabilizing (not making new lows)
        recent_low = df["low"].iloc[max(0, index - 5) : index].min()
        current_low = df["low"].iloc[index]
        price_stable = current_low >= recent_low * 0.98

        # Require 2+ recovery signals
        recovery_signals = sum([rsi_oversold, volume_normalizing, price_stable])
        return recovery_signals >= 2

    def get_feature_generators(self) -> Sequence:
        """No additional features required"""
        return []
