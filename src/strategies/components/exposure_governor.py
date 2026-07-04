"""Regime-gated gross exposure caps (#802).

Regime detection already adapts entry *thresholds*; in a bear market the total
*exposure* is the primary risk lever. The :class:`ExposureGovernor` caps total
gross open exposure per regime, applied after position sizing / dynamic risk and
before order placement, identically in the backtest and live engines (via the
shared entry-handler mixin).

The governor returns an **absolute cap** on the resulting gross fraction, not a
multiplier — composed with other pre-order gates by taking the most-restrictive
(minimum) result, so it never double-counts the graduated drawdown throttle that
dynamic risk already applies.
"""

from __future__ import annotations

import logging
import math

from src.config.constants import DEFAULT_EXPOSURE_CAP_UNKNOWN, DEFAULT_EXPOSURE_CAPS
from src.config.feature_flags import is_enabled
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.regime_utils import RegimeHelper

logger = logging.getLogger(__name__)

FEATURE_FLAG = "enable_exposure_governor"


def _validate_cap(name: str, value: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"exposure cap {name!r} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(f"exposure cap {name!r} must be finite in (0, 1], got {value!r}")
    return value


class ExposureGovernor:
    """Caps total gross open exposure by market regime.

    Args:
        caps: Mapping ``"{trend}_{vol}" -> cap`` (fraction of equity). Defaults to
            :data:`DEFAULT_EXPOSURE_CAPS`. Keys use the same labels as
            ``RegimeHelper`` (``trend_up``/``trend_down``/``range`` ×
            ``low_vol``/``high_vol``).
        unknown_cap: Cap used when the regime (or its trend) is unavailable —
            the most-conservative ceiling so an unknown regime never over-exposes.
        enabled: Force enabled/disabled, bypassing the feature flag (tests/wiring).
            ``None`` (default) reads the ``enable_exposure_governor`` flag.
    """

    def __init__(
        self,
        caps: dict[str, float] | None = None,
        *,
        unknown_cap: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        source = caps if caps is not None else DEFAULT_EXPOSURE_CAPS
        self.caps = {k: _validate_cap(k, v) for k, v in source.items()}
        self.unknown_cap = _validate_cap(
            "unknown",
            unknown_cap if unknown_cap is not None else DEFAULT_EXPOSURE_CAP_UNKNOWN,
        )
        self._enabled_override = enabled

    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return is_enabled(FEATURE_FLAG, default=False)

    def regime_cap(self, regime: RegimeContext | None) -> float:
        """Return the gross-exposure cap for ``regime`` (unknown -> conservative)."""
        trend = RegimeHelper.get_trend(regime)
        if regime is None or trend is None:
            return self.unknown_cap
        vol_key = "high_vol" if RegimeHelper.is_high_volatility(regime) else "low_vol"
        return self.caps.get(f"{trend}_{vol_key}", self.unknown_cap)

    def cap_fraction(
        self,
        size_fraction: float,
        *,
        regime: RegimeContext | None,
        gross_exposure_fraction: float,
        extra_factor: float = 1.0,
    ) -> tuple[float, str | None]:
        """Cap ``size_fraction`` so total gross exposure stays within the regime cap.

        Args:
            size_fraction: Proposed new-position fraction (post sizing/dynamic risk).
            regime: Current regime context (``None`` -> conservative cap).
            gross_exposure_fraction: Current open exposure as a fraction of equity,
                excluding the proposed new leg.
            extra_factor: Multiplier applied to the cap before headroom is computed
                (``<= 1`` to tighten, e.g. #806 event windows). Clamped to [0, 1].

        Returns:
            ``(allowed_fraction, reason_or_None)``. ``reason`` is set only when the
            proposed size was reduced (for entry-decision logging).
        """
        if size_fraction <= 0:
            return size_fraction, None
        factor = min(max(float(extra_factor), 0.0), 1.0)
        cap = self.regime_cap(regime) * factor
        current = gross_exposure_fraction if math.isfinite(gross_exposure_fraction) else 0.0
        headroom = cap - max(current, 0.0)
        if headroom <= 0:
            return 0.0, f"exposure_cap_reached_{cap:.4f}"
        if headroom < size_fraction:
            return headroom, f"exposure_capped_{cap:.4f}"
        return size_fraction, None
