"""
Momentum Signal Generator Component

Generates momentum/trend-based signals derived from the prior MomentumLeverage
strategy, designed for component-based composition.
"""

from typing import Any

import pandas as pd

from src.config.constants import (
    DEFAULT_CONFIDENCE_SCALE_FACTOR,
    DEFAULT_CONFIDENCE_SCALE_FACTOR_MOMENTUM,
)

from .regime_context import RegimeContext
from .signal_generator import Signal, SignalDirection, SignalGenerator


class MomentumSignalGenerator(SignalGenerator):
    """
    Momentum-based signal generator using multi-timeframe momentum,
    EMA trend alignment, breakout detection, and volatility context.
    """

    def __init__(
        self,
        name: str = "momentum_signal_generator",
        momentum_entry_threshold: float = 0.01,  # 1%
        strong_momentum_threshold: float = 0.025,  # 2.5%
        ema_fast: int = 12,
        ema_mid: int = 26,
        ema_slow: int = 50,
        momentum_fast_window: int = 3,
        momentum_mid_window: int = 7,
        momentum_slow_window: int = 20,
        breakout_lookback: int = 20,
    ):
        super().__init__(name)
        self.momentum_entry_threshold = momentum_entry_threshold
        self.strong_momentum_threshold = strong_momentum_threshold
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.momentum_fast_window = momentum_fast_window
        self.momentum_mid_window = momentum_mid_window
        self.momentum_slow_window = momentum_slow_window
        self.breakout_lookback = breakout_lookback

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        self.validate_inputs(df, index)

        # Ensure derived features exist; compute on the fly minimally
        momentum_3 = self._pct_change(df, "close", self.momentum_fast_window, index)
        momentum_7 = self._pct_change(df, "close", self.momentum_mid_window, index)

        ema_12 = self._ema(df, "close", self.ema_fast, index)
        ema_26 = self._ema(df, "close", self.ema_mid, index)
        ema_50 = self._ema(df, "close", self.ema_slow, index)

        trend_strength = self._safe_div(ema_12 - ema_26, ema_26)
        long_trend = self._safe_div(ema_26 - ema_50, ema_50)

        # Breakout vs N-high
        breakout = self._is_breakout(df, index, self.breakout_lookback)

        # Composite momentum strength
        strong_momentum = (
            (momentum_3 is not None and momentum_3 > self.momentum_entry_threshold)
            and (momentum_7 is not None and momentum_7 > 0.01)
            and (trend_strength is not None and trend_strength > 0.005)
        )

        bullish_trend = (
            ema_12 is not None
            and ema_26 is not None
            and ema_50 is not None
            and ema_12 > ema_26 > ema_50
        )

        # Entry conditions
        decision_buy = (
            (
                momentum_3 is not None
                and momentum_3 > self.momentum_entry_threshold
                and trend_strength
                and trend_strength > 0.005
            )
            or (breakout and momentum_3 and momentum_3 > 0.003)
            or (strong_momentum)
            or (bullish_trend and momentum_7 is not None and momentum_7 > 0.003)
        )

        if decision_buy:
            # Strength scaled by fast momentum, clipped
            fast = abs(momentum_3 or 0.0)
            strength = float(max(0.0, min(1.0, fast * DEFAULT_CONFIDENCE_SCALE_FACTOR)))
            confidence = float(max(0.0, min(1.0, fast * DEFAULT_CONFIDENCE_SCALE_FACTOR_MOMENTUM)))
            return Signal(
                direction=SignalDirection.BUY,
                strength=strength,
                confidence=confidence,
                metadata={
                    "generator": self.name,
                    "momentum_3": momentum_3 or 0.0,
                    "momentum_7": momentum_7 or 0.0,
                    "trend_strength": trend_strength or 0.0,
                    "long_trend": long_trend or 0.0,
                    "breakout": breakout,
                },
            )

        # No short logic by default (consistent with momentum long bias); HOLD otherwise
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata={
                "generator": self.name,
                "reason": "no_momentum_entry",
                "momentum_3": momentum_3 or 0.0,
                "momentum_7": momentum_7 or 0.0,
                "trend_strength": trend_strength or 0.0,
                "breakout": breakout,
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        momentum_3 = self._pct_change(df, "close", self.momentum_fast_window, index) or 0.0
        return float(max(0.0, min(1.0, abs(momentum_3) * DEFAULT_CONFIDENCE_SCALE_FACTOR_MOMENTUM)))

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required before producing valid signals.

        Requires max of EMA slow period and breakout lookback for reliable calculations.
        """
        return max(self.ema_slow, self.breakout_lookback)

    @staticmethod
    def _ema(df: pd.DataFrame, col: str, span: int, index: int) -> float | None:
        """Calculate EMA value at index, returning None on failure."""
        try:
            series = df[col].ewm(span=span).mean()
            return float(series.iloc[index])
        except (KeyError, IndexError, ValueError, TypeError):
            return None

    @staticmethod
    def _pct_change(df: pd.DataFrame, col: str, periods: int, index: int) -> float | None:
        """Calculate percent change, returning None on failure."""
        if index < periods:
            return None
        try:
            return float(
                (df[col].iloc[index] - df[col].iloc[index - periods])
                / max(df[col].iloc[index - periods], 1e-12)
            )
        except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def _is_breakout(df: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if current close is a breakout above prior high."""
        if index < lookback:
            return False
        try:
            prior_high = float(df["high"].iloc[index - lookback : index].max())
            return float(df["close"].iloc[index]) > prior_high
        except (KeyError, IndexError, ValueError, TypeError):
            return False

    @staticmethod
    def _safe_div(a: float | None, b: float | None) -> float | None:
        """Perform safe division, returning None on failure."""
        try:
            if a is None or b is None or b == 0:
                return None
            return float(a / b)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params.update(
            {
                "momentum_entry_threshold": self.momentum_entry_threshold,
                "strong_momentum_threshold": self.strong_momentum_threshold,
                "ema_fast": self.ema_fast,
                "ema_mid": self.ema_mid,
                "ema_slow": self.ema_slow,
                "momentum_fast_window": self.momentum_fast_window,
                "momentum_mid_window": self.momentum_mid_window,
                "momentum_slow_window": self.momentum_slow_window,
                "breakout_lookback": self.breakout_lookback,
            }
        )
        return params
