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
from typing import Any

logger = logging.getLogger(__name__)


def position_notional(position: Any) -> float:
    """Return a position's quote-currency notional at entry.

    A position's ``size`` is a fraction of the balance held at entry (see
    ``LivePositionTracker`` quantity derivation: ``size * entry_balance /
    entry_price``), so its notional is ``size * entry_balance``. Returns 0.0 for
    positions missing the fields or carrying non-finite values (logged), so a
    single bad record cannot silently corrupt the gross-exposure total.
    """
    size = getattr(position, "size", None)
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
