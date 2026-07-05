"""Gross-exposure accounting shared by the backtest and live engines (#802).

Both engines must compute "current gross open exposure" identically for the
regime-gated exposure caps to hold with backtest-live parity. The arithmetic
lives here once; each engine only supplies its open-position objects (both use
``BasePosition``, which carries ``size`` — a fraction of the balance at entry —
and ``entry_balance``), so the notional definition can never drift between them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.strategies.components.exposure_governor import ExposureGovernor

logger = logging.getLogger(__name__)


def position_notional(position: Any) -> float:
    """Return a position's current quote-currency notional.

    A position's size is a fraction of the balance held at entry (see
    ``LivePositionTracker`` quantity derivation: ``size * entry_balance /
    entry_price``), so its notional is ``size * entry_balance``. We use
    ``current_size`` — the live fraction after partial exits (smaller) and
    scale-ins (larger) — when present, falling back to the original ``size``.
    Using ``size`` would overstate gross exposure after a partial exit (#802
    follow-up P2). Returns 0.0 for positions missing the fields or carrying
    non-finite values (logged), so a single bad record cannot silently corrupt
    the gross-exposure total.
    """
    # ``current_size`` reflects partial-exit / scale-in state; prefer it. Only a
    # positive, finite value counts — otherwise fall back to the original size.
    current_size = getattr(position, "current_size", None)
    size = current_size if _is_positive_number(current_size) else getattr(position, "size", None)
    entry_balance = getattr(position, "entry_balance", None)
    if size is None or entry_balance is None:
        return 0.0
    try:
        notional = float(size) * float(entry_balance)
    except (TypeError, ValueError):
        logger.warning("position_notional: non-numeric size/entry_balance on %r", position)
        return 0.0
    if not math.isfinite(notional):
        logger.warning("position_notional: non-finite notional on %r", position)
        return 0.0
    return abs(notional)


def gross_exposure_fraction(positions: Iterable[Any], equity: float) -> float:
    """Return total gross open exposure as a fraction of current equity.

    Defined as ``sum(|entry notional|) / equity`` over the open positions,
    **excluding** any proposed new leg (the exposure governor adds that). Uses
    entry notional (not mark-to-market) for determinism. Returns 0.0 when equity
    is non-positive/non-finite (no meaningful fraction).
    """
    if not math.isfinite(equity) or equity <= 0:
        return 0.0
    total = 0.0
    for position in positions:
        total += position_notional(position)
    fraction = total / equity
    if not math.isfinite(fraction):
        return 0.0
    return fraction


def _is_positive_number(value: Any) -> bool:
    """True when ``value`` is a finite, positive real number (not a bool)."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    return math.isfinite(value) and value > 0


def scale_in_gross_cap_headroom(
    governor: ExposureGovernor | None,
    positions: Iterable[Any],
    equity: float | None,
) -> float | None:
    """Max additional exposure a scale-in may add under the regime gross cap.

    Scale-ins increase exposure, so they must respect the #802 gross cap just
    like new entries (#802 follow-up P3). Returns the headroom
    ``max(0, cap - current_gross)`` as a fraction of equity, or ``None`` when the
    governor is absent/disabled (no clamp — inert). Uses the **conservative
    (unknown-regime) cap**: the exit path has no regime context, and bounding
    scale-ins to the tightest cap is the safe, bear-appropriate choice.
    """
    if governor is None or not governor.enabled:
        return None
    if equity is None or not math.isfinite(equity) or equity <= 0:
        return None  # cannot compute gross without a valid equity denominator
    gross = gross_exposure_fraction(positions, equity)
    cap = governor.regime_cap(None)  # conservative cap; exit path lacks regime
    return max(0.0, cap - gross)
