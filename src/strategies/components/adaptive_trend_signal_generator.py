"""Adaptive Trend Signal Generator

A trend-following signal generator designed for crypto assets on daily timeframes.
Uses a single long-period EMA as a trend filter with asymmetric confirmation:
fast entry (when trend is clearly established), slow exit (to avoid whipsaw).

The strategy is intentionally very slow-moving to minimize whipsaw and capture
multi-month trends. It targets 3-8 round-trip trades over a 5-year period.

Core approach:
- Single EMA trend filter (price above/below EMA)
- Asymmetric confirmation: quick entry, slow exit
- Consecutive-day requirements to confirm crossovers
- Long-only (no shorting)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .regime_context import RegimeContext, TrendLabel
from .signal_generator import Signal, SignalDirection, SignalGenerator

logger = logging.getLogger(__name__)


class AdaptiveTrendSignalGenerator(SignalGenerator):
    """Trend-following signal generator using single-EMA trend detection.

    Uses the relationship between price and a long-period EMA to determine
    market trend direction. Consecutive-day confirmation prevents whipsaw.
    Asymmetric entry/exit thresholds: enter faster, exit slower.

    The generator targets very few trades (3-8 over 5 years) to minimize
    fee drag and capture major multi-month moves.
    """

    def __init__(
        self,
        name: str = "adaptive_trend",
        trend_ema_period: int = 100,
        entry_confirmation_days: int = 5,
        exit_confirmation_days: int = 10,
        entry_buffer_pct: float = 0.02,
        exit_buffer_pct: float = 0.02,
        exit_ratio_threshold: float = 0.75,
        ema_slope_lookback: int = 20,
        momentum_lookback: int = 30,
        atr_period: int = 14,
        # Legacy params for backward compat
        fast_ema_period: int | None = None,
        slow_ema_period: int | None = None,
        trend_confirmation_period: int | None = None,
    ) -> None:
        """Initialize the adaptive trend signal generator.

        Args:
            name: Signal generator name.
            trend_ema_period: Period for the main trend EMA.
            entry_confirmation_days: Consecutive days above EMA to confirm entry.
            exit_confirmation_days: Window size for ratio-based exit confirmation.
            entry_buffer_pct: Price must be this % above EMA for entry.
            exit_buffer_pct: Price must be this % below EMA for exit.
            exit_ratio_threshold: Fraction of exit_confirmation_days that must be
                below the exit threshold to trigger exit. For example, 0.75 means
                75% of recent days must be below EMA*(1-buffer). Ratio-based
                counting is robust to single-day bounces during bear markets.
            ema_slope_lookback: Number of days to measure EMA slope direction.
                Entry is only allowed when EMA is rising (positive slope),
                which filters out bear market rally entries.
            momentum_lookback: Lookback period for momentum calculation.
            atr_period: Period for ATR calculation.
            fast_ema_period: Deprecated, use trend_ema_period instead.
            slow_ema_period: Deprecated, use trend_ema_period instead.
            trend_confirmation_period: Deprecated, use entry/exit_confirmation_days.
        """
        super().__init__(name)

        # Handle legacy parameter mapping
        if fast_ema_period is not None and slow_ema_period is not None:
            trend_ema_period = slow_ema_period
        if trend_confirmation_period is not None:
            entry_confirmation_days = trend_confirmation_period
            exit_confirmation_days = trend_confirmation_period * 2

        self.trend_ema_period = trend_ema_period
        self.entry_confirmation_days = entry_confirmation_days
        self.exit_confirmation_days = exit_confirmation_days
        self.entry_buffer_pct = entry_buffer_pct
        self.exit_buffer_pct = exit_buffer_pct
        self.exit_ratio_threshold = exit_ratio_threshold
        self.ema_slope_lookback = ema_slope_lookback
        self.momentum_lookback = momentum_lookback
        self.atr_period = atr_period

        # Cache for precomputed EMA values (avoids recalculating every bar)
        self._cached_ema: np.ndarray | None = None
        self._cached_ema_length: int = 0
        self._cached_data_hash: int | None = None

    @property
    def warmup_period(self) -> int:
        """Minimum history needed before generating signals.

        Only the EMA period is required for meaningful signals. Entry and exit
        confirmation logic gracefully handles limited history via ratio-based
        counting and backward iteration.
        """
        return self.trend_ema_period + 5

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        """Generate a trend-following signal based on price vs EMA.

        Args:
            df: DataFrame with OHLCV data.
            index: Current bar index.
            regime: Optional regime context.

        Returns:
            Signal with direction, strength, and confidence.
        """
        self.validate_inputs(df, index)

        if index < self.warmup_period:
            return self._hold_signal(index, "warmup_period")

        close = df["close"].values

        # Compute EMA for the full series (cached for performance)
        ema_values = self._compute_ema_series(close, index)

        current_price = float(close[index])
        current_ema = float(ema_values[index])

        if current_ema <= 0:
            return self._hold_signal(index, "invalid_ema")

        # Price position relative to EMA
        price_vs_ema_pct = (current_price - current_ema) / current_ema

        # Entry: consecutive days above EMA (with buffer for robustness)
        days_above = self._count_consecutive_days_above(
            close, ema_values, index, self.entry_buffer_pct
        )
        # Exit: ratio of recent days below EMA threshold (robust to bounces)
        days_below_ratio = self._count_ratio_days_below(
            close, ema_values, index, self.exit_buffer_pct, self.exit_confirmation_days
        )

        # Calculate EMA slope: rising EMA confirms uptrend, declining filters bear rallies
        ema_slope = self._calculate_ema_slope(ema_values, index)

        # Calculate momentum
        momentum = self._calculate_momentum(close, index)

        # Calculate ATR for metadata
        high = df["high"].values
        low = df["low"].values
        atr = self._calculate_atr(high, low, close, index)
        atr_pct = atr / current_price if current_price > 0 else 0.0

        # Build metadata
        metadata: dict[str, Any] = {
            "generator": self.name,
            "index": index,
            "trend_ema": current_ema,
            "price_vs_ema_pct": price_vs_ema_pct,
            "days_above_ema": days_above,
            "days_below_ratio": days_below_ratio,
            "ema_slope": ema_slope,
            "momentum": momentum,
            "atr_pct": atr_pct,
        }

        if regime is not None:
            metadata["regime_trend"] = regime.trend.value
            metadata["regime_vol"] = regime.volatility.value
            metadata["regime_confidence"] = regime.confidence

        # --- EXIT (SELL) logic ---
        # Ratio-based: sufficient fraction of recent days below EMA threshold.
        # Robust to single-day bounces during bear markets while still catching
        # sustained downtrends. For example, with window=40 and threshold=0.75,
        # price must be below EMA*(1-buffer) for 30 of the last 40 days.
        price_below_with_buffer = current_price < current_ema * (1 - self.exit_buffer_pct)
        if price_below_with_buffer and days_below_ratio >= self.exit_ratio_threshold:
            strength = min(1.0, abs(price_vs_ema_pct) * 10)
            confidence = self._calculate_exit_confidence(
                price_vs_ema_pct, momentum, days_below_ratio, regime
            )
            metadata["signal_reason"] = "price_below_ema_confirmed"
            return Signal(
                direction=SignalDirection.SELL,
                strength=max(0.1, strength),
                confidence=max(0.1, confidence),
                metadata=metadata,
            )

        # --- ENTRY (BUY) logic ---
        # Price has been above EMA for entry_confirmation_days + buffer
        price_above_with_buffer = current_price > current_ema * (1 + self.entry_buffer_pct)
        if price_above_with_buffer and days_above >= self.entry_confirmation_days:
            # EMA slope filter: only enter when EMA is rising (confirms uptrend).
            # In bear markets the EMA declines even during rallies, preventing
            # false entries that would result in losses.
            if ema_slope < 0:
                return self._hold_signal(index, "declining_ema", metadata)

            # Require non-negative momentum to avoid late-cycle entries
            if momentum <= -0.05:
                return self._hold_signal(index, "negative_momentum", metadata)

            strength = min(1.0, abs(price_vs_ema_pct) * 10)
            confidence = self._calculate_entry_confidence(
                price_vs_ema_pct, momentum, days_above, atr_pct, regime
            )
            metadata["signal_reason"] = "price_above_ema_confirmed"
            return Signal(
                direction=SignalDirection.BUY,
                strength=max(0.1, strength),
                confidence=max(0.5, confidence),
                metadata=metadata,
            )

        # --- HOLD ---
        return self._hold_signal(index, "no_confirmed_signal", metadata)

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence score at the given index."""
        self.validate_inputs(df, index)
        if index < self.warmup_period:
            return 0.0

        close = df["close"].values
        ema_values = self._compute_ema_series(close, index)
        current_ema = float(ema_values[index])
        if current_ema <= 0:
            return 0.0

        price_vs_ema = abs(float(close[index]) - current_ema) / current_ema
        return min(1.0, price_vs_ema * 10)

    def get_parameters(self) -> dict[str, Any]:
        """Get signal generator parameters."""
        params = super().get_parameters()
        params.update(
            {
                "trend_ema_period": self.trend_ema_period,
                "entry_confirmation_days": self.entry_confirmation_days,
                "exit_confirmation_days": self.exit_confirmation_days,
                "entry_buffer_pct": self.entry_buffer_pct,
                "exit_buffer_pct": self.exit_buffer_pct,
                "exit_ratio_threshold": self.exit_ratio_threshold,
                "ema_slope_lookback": self.ema_slope_lookback,
                "momentum_lookback": self.momentum_lookback,
                "atr_period": self.atr_period,
            }
        )
        return params

    def _compute_ema_series(self, close: np.ndarray, max_index: int) -> np.ndarray:
        """Compute EMA for the series up to max_index, using cache.

        Args:
            close: Close price array.
            max_index: Maximum index to compute EMA up to.

        Returns:
            Array of EMA values.
        """
        needed_length = max_index + 1
        data_hash = hash(close[:needed_length].tobytes())

        if (
            self._cached_ema is not None
            and self._cached_ema_length >= needed_length
            and self._cached_data_hash == data_hash
        ):
            return self._cached_ema

        series = pd.Series(close[:needed_length])
        ema_values = series.ewm(span=self.trend_ema_period, adjust=False).mean().values
        self._cached_ema = ema_values
        self._cached_ema_length = needed_length
        self._cached_data_hash = data_hash
        return ema_values

    def _count_consecutive_days_above(
        self, close: np.ndarray, ema: np.ndarray, index: int, buffer_pct: float = 0.0
    ) -> int:
        """Count consecutive days where close > EMA * (1 + buffer_pct).

        Using a buffer prevents counting marginal crossings and reduces
        whipsaw in sideways markets.

        Args:
            close: Close prices.
            ema: EMA values.
            index: Current bar index.
            buffer_pct: Minimum percentage above EMA to count as "above".

        Returns:
            Number of consecutive days above threshold (from current bar backward).
        """
        count = 0
        for i in range(index, -1, -1):
            threshold = float(ema[i]) * (1 + buffer_pct)
            if float(close[i]) > threshold:
                count += 1
            else:
                break
        return count

    def _count_ratio_days_below(
        self,
        close: np.ndarray,
        ema: np.ndarray,
        index: int,
        buffer_pct: float,
        window: int,
    ) -> float:
        """Calculate the ratio of recent days where close < EMA * (1 - buffer).

        Ratio-based counting is more robust than consecutive-day counting
        because it tolerates single-day bounces that are common in crypto
        bear markets. A bear rally that lasts 1-2 days won't reset the count.

        Args:
            close: Close prices.
            ema: EMA values.
            index: Current bar index.
            buffer_pct: Minimum percentage below EMA to count as "below".
            window: Number of recent days to examine.

        Returns:
            Ratio (0.0 to 1.0) of days below threshold within the window.
        """
        start = max(0, index - window + 1)
        actual_window = index - start + 1
        if actual_window <= 0:
            return 0.0

        below_count = 0
        for i in range(start, index + 1):
            threshold = float(ema[i]) * (1 - buffer_pct)
            if float(close[i]) < threshold:
                below_count += 1

        return below_count / actual_window

    def _calculate_ema_slope(self, ema: np.ndarray, index: int) -> float:
        """Calculate the slope of the EMA over the slope lookback period.

        Positive slope means the EMA is rising (bullish), negative means
        declining (bearish). Used to filter out bear market rally entries.

        Args:
            ema: EMA values array.
            index: Current bar index.

        Returns:
            EMA slope as a decimal (e.g., 0.05 = 5% rise over lookback).
        """
        lookback_index = index - self.ema_slope_lookback
        if lookback_index < 0:
            return 0.0

        past_ema = float(ema[lookback_index])
        if past_ema <= 0:
            return 0.0

        return (float(ema[index]) - past_ema) / past_ema

    def _calculate_momentum(self, close: np.ndarray, index: int) -> float:
        """Calculate price momentum over lookback period.

        Args:
            close: Close price array.
            index: Current bar index.

        Returns:
            Momentum as decimal.
        """
        lookback_index = index - self.momentum_lookback
        if lookback_index < 0:
            return 0.0
        past_price = float(close[lookback_index])
        if past_price <= 0:
            return 0.0
        return (float(close[index]) - past_price) / past_price

    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        index: int,
    ) -> float:
        """Calculate Average True Range at the given index."""
        if index < self.atr_period:
            return float(np.mean(high[: index + 1] - low[: index + 1]))

        start = index - self.atr_period + 1
        tr_values = []
        for i in range(start, index + 1):
            hl = float(high[i]) - float(low[i])
            hc = abs(float(high[i]) - float(close[i - 1])) if i > 0 else hl
            lc = abs(float(low[i]) - float(close[i - 1])) if i > 0 else hl
            tr_values.append(max(hl, hc, lc))

        return float(np.mean(tr_values))

    def _calculate_entry_confidence(
        self,
        price_vs_ema_pct: float,
        momentum: float,
        days_above: int,
        atr_pct: float,
        regime: RegimeContext | None,
    ) -> float:
        """Calculate confidence for BUY signal."""
        confidence = 0.50  # Base confidence for confirmed trend

        # Price distance above EMA (further = stronger trend)
        if price_vs_ema_pct > 0.10:
            confidence += 0.15
        elif price_vs_ema_pct > 0.05:
            confidence += 0.10

        # Momentum contribution
        if momentum > 0.10:
            confidence += 0.15
        elif momentum > 0.05:
            confidence += 0.10
        elif momentum > 0:
            confidence += 0.05

        # Persistence bonus (longer above EMA = more reliable)
        if days_above > 30:
            confidence += 0.10
        elif days_above > 15:
            confidence += 0.05

        # Regime alignment
        if regime is not None and regime.trend == TrendLabel.TREND_UP:
            confidence += 0.05

        return max(0.5, min(1.0, confidence))

    def _calculate_exit_confidence(
        self,
        price_vs_ema_pct: float,
        momentum: float,
        below_ratio: float,
        regime: RegimeContext | None,
    ) -> float:
        """Calculate confidence for SELL signal.

        Args:
            price_vs_ema_pct: Current price position vs EMA as decimal.
            momentum: Recent momentum as decimal.
            below_ratio: Ratio of recent days below exit threshold (0-1).
            regime: Optional regime context.
        """
        confidence = 0.50

        # Price distance below EMA
        if price_vs_ema_pct < -0.10:
            confidence += 0.15
        elif price_vs_ema_pct < -0.05:
            confidence += 0.10

        # Negative momentum
        if momentum < -0.10:
            confidence += 0.15
        elif momentum < -0.05:
            confidence += 0.10

        # Higher ratio = more persistent bearishness
        if below_ratio > 0.90:
            confidence += 0.10
        elif below_ratio > 0.80:
            confidence += 0.05

        # Regime confirmation
        if regime is not None and regime.trend == TrendLabel.TREND_DOWN:
            confidence += 0.10

        return max(0.5, min(1.0, confidence))

    def _hold_signal(
        self, index: int, reason: str, metadata: dict[str, Any] | None = None
    ) -> Signal:
        """Create a HOLD signal."""
        meta = metadata.copy() if metadata else {}
        meta.update({"generator": self.name, "index": index, "signal_reason": reason})
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.5,
            metadata=meta,
        )
