"""Sentiment-extreme mean-reversion overlay (#804).

At Fear & Greed extremes, fading beats following: shorting into capitulation
(F&G very low) gets squeezed by relief rallies, and blindly following euphoria
(F&G very high) buys the top. This overlay wraps a signal generator and, at
extremes, vetoes the dangerous direction and narrowly permits the contrarian one:

- **Extreme fear** (F&G < ``extreme_fear``): block new SHORT entries; permit new
  LONG entries only when price is within a configurable band of a structural
  support level (a config *parameter*, since market levels go stale). With no
  support level configured, longs pass through and only shorts are blocked.
- **Extreme greed** (F&G > ``extreme_greed``) while regime = trend_down: permit
  small fade SHORTs (no veto) — this branch is permissive.

Implemented as a ``SignalGenerator`` decorator so it composes with the ETF flow
gate (#803) and applies in both engines via the strategy — most-restrictive-wins
(any veto turns the signal into HOLD). Fear & Greed comes from the
``FearGreedProvider`` (which degrades to neutral 0.5 offline, making the overlay
inert), read once per candle from an in-memory series.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.config.constants import (
    DEFAULT_SENTIMENT_EXTREME_FEAR,
    DEFAULT_SENTIMENT_EXTREME_GREED,
    DEFAULT_SENTIMENT_SUPPORT_BAND_PCT,
    DEFAULT_SENTIMENT_SUPPORT_LEVEL,
)
from src.config.feature_flags import is_enabled
from src.strategies.components.regime_utils import RegimeHelper
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator

if TYPE_CHECKING:
    from src.data_providers.feargreed_provider import FearGreedProvider
    from src.strategies.components.regime_context import RegimeContext

logger = logging.getLogger(__name__)

FEATURE_FLAG = "enable_sentiment_extreme_overlay"


class SentimentExtremeOverlay(SignalGenerator):
    """Fades Fear & Greed extremes: blocks capitulation shorts + euphoria-chasing.

    Args:
        base: Wrapped signal generator.
        provider: Fear & Greed provider (defaults to a real ``FearGreedProvider``).
        extreme_fear/extreme_greed: F&G→[0,1] thresholds.
        support_level: Structural support price for the mean-reversion long band
            (``None`` disables the band restriction). Config, not a constant.
        support_band_pct: Fractional half-width of the support band.
    """

    def __init__(
        self,
        base: SignalGenerator,
        provider: FearGreedProvider | None = None,
        *,
        extreme_fear: float = DEFAULT_SENTIMENT_EXTREME_FEAR,
        extreme_greed: float = DEFAULT_SENTIMENT_EXTREME_GREED,
        support_level: float | None = DEFAULT_SENTIMENT_SUPPORT_LEVEL,
        support_band_pct: float = DEFAULT_SENTIMENT_SUPPORT_BAND_PCT,
    ) -> None:
        super().__init__(name=f"sentiment_overlay({getattr(base, 'name', 'signal')})")
        self._base = base
        self._provider = provider
        self._extreme_fear = float(extreme_fear)
        self._extreme_greed = float(extreme_greed)
        self._support_level = float(support_level) if support_level is not None else None
        self._support_band_pct = float(support_band_pct)

    def _get_provider(self) -> FearGreedProvider:
        if self._provider is None:
            from src.data_providers.feargreed_provider import FearGreedProvider

            self._provider = FearGreedProvider()
        return self._provider

    def _fear_greed(self, as_of: datetime) -> float | None:
        try:
            data = self._get_provider().get_sentiment_for_date(as_of)
        except Exception:  # noqa: BLE001 - sentiment lookup must never break entries
            logger.warning("SentimentExtremeOverlay: F&G lookup failed", exc_info=True)
            return None
        value = data.get("sentiment_primary")
        if not isinstance(value, int | float) or isinstance(value, bool):
            return None
        return float(value)

    def _within_support_band(self, price: float) -> bool:
        if self._support_level is None or price <= 0:
            # No configured support -> band restriction inactive (allow longs).
            return True
        low = self._support_level * (1.0 - self._support_band_pct)
        high = self._support_level * (1.0 + self._support_band_pct)
        return low <= price <= high

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        signal = self._base.generate_signal(df, index, regime)
        if signal.direction == SignalDirection.HOLD:
            return signal
        as_of = _timestamp_at(df, index)
        if as_of is None:
            return signal
        fng = self._fear_greed(as_of)
        if fng is None:
            return signal

        if fng < self._extreme_fear:
            if signal.direction == SignalDirection.SELL:
                return _to_hold(signal, f"sentiment_extreme_fear_short_block_{fng:.2f}")
            # BUY: mean-reversion long only near the structural support level.
            if self._support_level is not None:
                price = _close_at(df, index)
                if price is None or not self._within_support_band(price):
                    return _to_hold(signal, f"sentiment_fear_long_outside_support_{fng:.2f}")
            return signal

        if fng > self._extreme_greed and RegimeHelper.is_bear_market(regime):
            # Permit small fade shorts in euphoric downtrends — no veto.
            return signal

        return signal

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return self._base.get_confidence(df, index)

    @property
    def warmup_period(self) -> int:
        return getattr(self._base, "warmup_period", 0)

    def get_feature_generators(self):
        getter = getattr(self._base, "get_feature_generators", None)
        return getter() if callable(getter) else []


def _to_hold(signal: Signal, reason: str) -> Signal:
    metadata = dict(signal.metadata)
    metadata["sentiment_overlay_blocked"] = reason
    metadata["sentiment_overlay_original_direction"] = signal.direction.value
    logger.debug("Sentiment overlay vetoed %s: %s", signal.direction.value, reason)
    return Signal(
        direction=SignalDirection.HOLD,
        strength=0.0,
        confidence=signal.confidence,
        metadata=metadata,
    )


def _close_at(df: pd.DataFrame, index: int) -> float | None:
    try:
        return float(df["close"].iloc[index])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _timestamp_at(df: pd.DataFrame, index: int) -> datetime | None:
    """Best-effort UTC timestamp for candle ``index`` (None if unavailable)."""
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[index]
        elif "timestamp" in df.columns:
            ts = pd.Timestamp(df["timestamp"].iloc[index])
        else:
            return None
    except (IndexError, KeyError, ValueError):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def maybe_wrap_with_sentiment_overlay(
    signal_generator: SignalGenerator,
    provider: FearGreedProvider | None = None,
    **kwargs: object,
) -> SignalGenerator:
    """Wrap ``signal_generator`` with the overlay iff the feature flag is on.

    Resolves the flag ONCE (at strategy-build time). Returns the generator
    unchanged when the flag is off (zero cost).
    """
    if is_enabled(FEATURE_FLAG, default=False):
        return SentimentExtremeOverlay(signal_generator, provider, **kwargs)  # type: ignore[arg-type]
    return signal_generator


__all__ = ["SentimentExtremeOverlay", "maybe_wrap_with_sentiment_overlay"]
