"""MFE-conditioned early-cut exit policy.

Cuts a position whose maximum favorable excursion (MFE) never reached a
threshold within the first N hours after entry — the live-trade-review
hypothesis that losers identify themselves early with tiny MFE and then
bleed for days (docs/research/notes/2026-07-12_live-trade-review.md, #971).

Design notes (money-path constraints):

- **Stateless & shared**: the window MFE is recomputed from candle history on
  every evaluation instead of being accumulated on the position object, so
  both engines get bit-identical decisions from the same candles (the
  ``barrier_touch`` pattern from #948) and a live restart cannot corrupt the
  running maximum.
- **Raw price excursion**: MFE here is the raw price move from entry
  (``(high - entry) / entry`` for longs), recomputed from candles rather
  than read from ``MFEMAETracker`` / the DB ``trades.mfe`` column — the
  stateless recompute is what gives both engines bit-identical decisions
  and restart safety. ``MFEMAETracker`` now uses the same raw-excursion
  units, but its running maximum is in-memory (sampled, reset on restart),
  so it is still not a valid input for exit decisions.
- **Entry-bar excluded**: the entry bar's extremes may predate the fill, so
  only bars strictly after ``entry_time`` count — the same convention as
  ``is_same_bar_entry`` exit protection.
- **Unknown history never cuts**: if the available candles do not reach back
  to the entry, the true window MFE is unknowable and the policy holds
  (conservative: never flatten a position on incomplete information).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_VALID_SIDES = frozenset({"long", "short"})


@dataclass(frozen=True)
class EarlyCutDecision:
    """Result of an early-cut evaluation.

    Attributes:
        should_exit: True when the position should be flattened.
        reason: Human/DB-facing exit reason (stable ``"Early cut"`` prefix)
            when ``should_exit`` is True.
        window_mfe_pct: Raw price MFE observed inside the evaluation window
            (decimal fraction, >= 0), or None when it could not be computed
            (no data, or history not covering the entry).
    """

    should_exit: bool
    reason: str | None = None
    window_mfe_pct: float | None = None


def _timeframe_to_hours(timeframe: str) -> float | None:
    """Convert a timeframe string (``15m``, ``1h``, ``4h``, ``1d``, ``1w``)
    to hours. Returns None when unrecognized. Mirrors the unit conventions
    of ``CachedDataProvider._get_timeframe_timedelta``."""
    if not timeframe or not isinstance(timeframe, str):
        return None
    try:
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1]) if len(timeframe) > 1 else 1
    except (ValueError, TypeError):
        return None
    if value <= 0:
        return None
    if unit == "m":
        return value / 60.0
    if unit == "h":
        return float(value)
    if unit == "d":
        return value * 24.0
    if unit == "w":
        return value * 24.0 * 7.0
    return None


def _as_index_timestamp(dt: datetime, index: pd.Index) -> pd.Timestamp:
    """Normalize a datetime to the tz-awareness of ``index`` (UTC semantics).

    All timestamps in this system are UTC; naive values are promoted to UTC
    rather than guessed (same convention as ``is_same_bar_entry``).
    """
    ts = pd.Timestamp(dt)
    index_tz = getattr(index, "tz", None)
    if index_tz is None:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    elif ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def compute_window_raw_mfe(
    df: pd.DataFrame | None,
    *,
    side: str,
    entry_price: float,
    entry_time: datetime,
    window_end: datetime,
) -> float | None:
    """Compute the raw price MFE over bars strictly inside (entry, window_end).

    Args:
        df: OHLCV frame with a DatetimeIndex of bar open times.
        side: ``"long"`` or ``"short"``.
        entry_price: Position entry price.
        entry_time: Position entry time (bar open in backtest, fill time live).
        window_end: End of the evaluation window (exclusive).

    Returns:
        Max favorable excursion as a decimal fraction (floored at 0.0), or
        None when it cannot be computed — invalid entry price, empty data,
        history that does not reach back to the entry, or a window that
        contains no bars at all (data gap, or a window no longer than the
        bar interval). The caller must treat None as "do not cut": a
        fabricated 0% from unknowable data would flatten every position.

    Raises:
        ValueError: If ``side`` is not "long" or "short".
    """
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None
    if df is None or len(df) == 0:
        return None

    index = df.index
    entry_ts = _as_index_timestamp(entry_time, index)
    end_ts = _as_index_timestamp(window_end, index)

    # Coverage guard: without a bar at or before entry we cannot know the
    # window's true excursion (e.g. live restart with a short kline buffer).
    if index[0] > entry_ts:
        return None

    window = df[(index > entry_ts) & (index < end_ts)]
    if window.empty:
        # No bar inside the window (data gap, or window <= bar interval):
        # the true excursion is unknowable — never fabricate 0% (#976 F1).
        return None

    if side == "long":
        extreme = float(window["high"].max())
        if not math.isfinite(extreme):
            return None
        return max(0.0, (extreme - entry_price) / entry_price)

    extreme = float(window["low"].min())
    if not math.isfinite(extreme):
        return None
    return max(0.0, (entry_price - extreme) / entry_price)


@dataclass(frozen=True)
class EarlyCutPolicy:
    """Flatten a position whose MFE never reached a threshold early on.

    Attributes:
        mfe_threshold_pct: Raw price MFE (decimal fraction, e.g. 0.015 =
            +1.5% from entry) the position must have reached inside the
            evaluation window to be allowed to keep running.
        evaluation_window_hours: Length of the window, in hours from entry.
    """

    mfe_threshold_pct: float
    evaluation_window_hours: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mfe_threshold_pct, int | float)
            or not math.isfinite(self.mfe_threshold_pct)
            or self.mfe_threshold_pct <= 0
        ):
            raise ValueError(
                f"mfe_threshold_pct must be a positive finite fraction, "
                f"got {self.mfe_threshold_pct!r}"
            )
        if (
            not isinstance(self.evaluation_window_hours, int | float)
            or not math.isfinite(self.evaluation_window_hours)
            or self.evaluation_window_hours <= 0
        ):
            raise ValueError(
                f"evaluation_window_hours must be a positive finite number, "
                f"got {self.evaluation_window_hours!r}"
            )

    def validate_for_timeframe(self, timeframe: str) -> None:
        """Reject window sizes that cannot work on the traded timeframe.

        Engines call this before trading starts (#976 review F1/F2):

        - A window no longer than one bar leaves ZERO bars strictly inside
          ``(entry, entry + window)`` — every evaluation would be
          unknowable, silently disabling the policy the config asked for.
        - A window that is not a whole multiple of the bar interval makes
          live (forming boundary bar) and backtest (completed bar) see
          different window contents — a latent parity divergence.

        Args:
            timeframe: Bar interval string, e.g. ``"15m"``, ``"1h"``, ``"4h"``.

        Raises:
            ValueError: If the timeframe is unrecognized, the window is not
                strictly longer than one bar, or the window is not a whole
                multiple of the bar interval.
        """
        bar_hours = _timeframe_to_hours(timeframe)
        if bar_hours is None:
            raise ValueError(
                f"Cannot validate early-cut evaluation window: unrecognized "
                f"timeframe {timeframe!r} (expected e.g. '15m', '1h', '4h', '1d')"
            )
        if self.evaluation_window_hours <= bar_hours:
            raise ValueError(
                f"early_cut evaluation_window_hours="
                f"{self.evaluation_window_hours:g} must be strictly longer "
                f"than one {timeframe} bar ({bar_hours:g}h): a window this "
                f"short contains no completed bars, so the window MFE is "
                f"never computable and the policy cannot work"
            )
        ratio = self.evaluation_window_hours / bar_hours
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"early_cut evaluation_window_hours="
                f"{self.evaluation_window_hours:g} must be a whole multiple "
                f"of the {timeframe} bar interval ({bar_hours:g}h): "
                f"non-bar-aligned windows evaluate a forming bar in live but "
                f"a completed bar in backtest, breaking parity"
            )

    def check_early_cut_conditions(
        self,
        *,
        side: str,
        entry_price: float,
        entry_time: datetime,
        now_time: datetime,
        df: pd.DataFrame | None,
    ) -> EarlyCutDecision:
        """Evaluate the early-cut rule for a position.

        The window MFE is frozen at the first ``evaluation_window_hours``
        after entry: favorable moves AFTER the window never rescue the trade,
        and the cut only requires MFE to be strictly below the threshold.

        Args:
            side: ``"long"`` or ``"short"``.
            entry_price: Position entry price.
            entry_time: Position entry time.
            now_time: Evaluation time (bar open in backtest, wall clock live).
            df: OHLCV history covering the entry (bar open times as index).

        Returns:
            EarlyCutDecision; ``should_exit`` is False whenever the window
            has not elapsed or the window MFE cannot be computed.
        """
        window_end = entry_time + timedelta(hours=self.evaluation_window_hours)
        try:
            elapsed_window = now_time >= window_end
        except TypeError:
            # Mixed aware/naive comparison — promote the naive side to UTC
            # via pandas, matching _as_index_timestamp semantics.
            now_ts = pd.Timestamp(now_time)
            end_ts = pd.Timestamp(window_end)
            if now_ts.tzinfo is None:
                now_ts = now_ts.tz_localize("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            elapsed_window = now_ts >= end_ts
        if not elapsed_window:
            return EarlyCutDecision(should_exit=False)

        mfe = compute_window_raw_mfe(
            df,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            window_end=window_end,
        )
        if mfe is None:
            logger.debug(
                "Early-cut window MFE not computable (insufficient history) — holding position"
            )
            return EarlyCutDecision(should_exit=False, window_mfe_pct=None)

        if mfe < self.mfe_threshold_pct:
            reason = (
                f"Early cut (MFE {mfe:.2%} < {self.mfe_threshold_pct:.2%} "
                f"in {self.evaluation_window_hours:g}h)"
            )
            return EarlyCutDecision(should_exit=True, reason=reason, window_mfe_pct=mfe)

        return EarlyCutDecision(should_exit=False, window_mfe_pct=mfe)


__all__ = ["EarlyCutPolicy", "EarlyCutDecision", "compute_window_raw_mfe"]
