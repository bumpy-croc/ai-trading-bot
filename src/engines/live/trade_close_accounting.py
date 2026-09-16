"""Close-accounting helpers shared by the live exit and recovery paths.

Compute the base quantity, entry-fee (USD), and original-position portion that
a closing ``Trade`` row should record, honoring partial-exit/scale-in state.
Moved verbatim from ``trading_engine`` (#486); the engine re-exports them so
existing imports keep working.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.engines.live.execution.position_tracker import LivePosition as Position

logger = logging.getLogger(__name__)


def held_base_quantity(
    qty: float | None,
    current_size: float | None,
    original_size: float | None,
    *,
    allow_scale_in: bool = False,
) -> float | None:
    """Base quantity represented by ``qty`` scaled by ``current_size / original_size``.

    The single implementation of the "held quantity" ratio duplicated (with divergent,
    inconsistent guards) across ``reconciliation.py`` and ``stop_loss_manager.py`` before
    #1208 consolidated it. Guards mirror ``_closed_base_quantity``: returns ``None`` when
    ``qty`` is missing/non-finite/non-positive, ``original_size`` is not finite-positive,
    or ``current_size`` is not finite-non-negative (``0.0`` is a valid, fully-exited
    holding, not corrupt state — CODE.md Position Fields).

    ``current_size > original_size`` (a scale-in) is guarded by default (``None``): before
    #1206 this was the NORMAL shape of a scaled-in position (``apply_scale_in`` grew
    ``current_size`` without growing ``original_size``/``quantity`` in lockstep), so it is
    only a corrupted/legacy-state signal now that #1206 keeps them in lockstep going
    forward. ``allow_scale_in=True`` (the default at every consolidated call site except
    ``_log_reconciliation_trade``) computes the scaled value anyway rather than returning
    ``None`` for it: these are operational quantities (stop-loss sizing, notional
    estimates, P&L on close) where under-sizing a real held amount to 0 risks leaving live
    inventory unprotected, which is worse than a ratio past 1.0 for the rare legacy/
    corrupted case. ``_log_reconciliation_trade`` keeps the strict default because its
    consumer is a persisted ``trades.quantity`` audit column, matching
    ``_closed_base_quantity``'s own no-fabrication policy exactly (same purpose, same
    guard).
    """
    if qty is None:
        return None
    try:
        qty_f = float(qty)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(qty_f) or qty_f <= 0:
        return None
    if current_size is None or original_size is None:
        return None
    try:
        original_f = float(original_size)
        current_f = float(current_size)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(original_f) and original_f > 0):
        return None
    if not (math.isfinite(current_f) and current_f >= 0):
        return None
    if current_f > original_f and not allow_scale_in:
        return None
    return qty_f * (current_f / original_f)


def _closed_base_quantity(position: Position) -> float | None:
    """Actual filled base-asset quantity represented by a closing ``Trade`` row.

    ``position.quantity`` is the authoritative filled base quantity of the *original*
    position (set from the entry fill, see LiveExecutionEngine.execute_entry). Partial
    exits reduce ``current_size`` (a fraction of balance) but never mutate ``quantity``,
    so the quantity actually being closed is ``position.quantity`` scaled by the fraction
    of the original position remaining.

    Returns ``None`` (so the Trade row stores NULL rather than a fabricated or negative
    value) when the filled quantity is unknown or the sizing inputs are corrupt:
    missing/non-positive/non-finite ``quantity``, ``original_size`` not finite-positive,
    or ``current_size`` not finite-non-negative.
    """
    qty = getattr(position, "quantity", None)
    if qty is None:
        return None
    try:
        qty = float(qty)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(qty) or qty <= 0:
        return None
    # Fall back to ``size`` only when ``original_size`` was never set (None). An
    # explicit 0 / negative / non-finite original_size is corrupt sizing state — return
    # None rather than silently scaling against ``size`` and fabricating a quantity.
    original = position.original_size if position.original_size is not None else position.size
    current = position.current_size if position.current_size is not None else position.size
    try:
        original_f = float(original)
        current_f = float(current)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(original_f) and original_f > 0):
        return None
    if not (math.isfinite(current_f) and current_f > 0):
        # current_size <= 0 is a degenerate/flat slice — store NULL rather than a fabricated
        # 0.0 quantity (which _close_position_portion would still pair with the full fee).
        return None
    if current_f > original_f:
        # Scale-ins grow current_size beyond original_size but do NOT update
        # ``quantity`` (the original entry fill), so the held base quantity cannot
        # be derived by scaling. Store NULL rather than over-report the fill.
        return None
    return qty * (current_f / original_f)


def _close_entry_fee_usd(position: Position, execution_engine: Any) -> float:
    """Entry commission (USD) for a closing ``Trade`` row's ``commission``.

    Prefer the exact value booked at open (``position.metadata['entry_fee']``, the same
    amount deducted from ``account_balances`` as the ``entry_fee_<symbol>`` event).

    Positions recovered after a restart lose that metadata (the ``positions`` table does
    not persist entry fee), so fall back to an **approximate** reconstruction: the
    execution engine's fee model applied to the recovered entry notional. This equals the
    booked fee exactly in the common case (entries booked at the modelled rate) and is a
    close estimate otherwise (e.g. a live fill booked at the actual exchange commission) —
    preferred over dropping the entry leg, which would understate ``trades.commission``
    versus the ledger. Never raises; returns ``0.0`` when neither the metadata nor a
    usable notional is available.
    """
    meta = getattr(position, "metadata", None) or {}
    if "entry_fee" in meta:
        try:
            fee = float(meta["entry_fee"])
            if math.isfinite(fee):
                return fee
        except (TypeError, ValueError):
            pass
    # Recovered position: reconstruct from persisted entry economics (quantity * entry
    # price, falling back to size * entry_balance), via the same USD fee model.
    try:
        qty = getattr(position, "quantity", None)
        entry_price = float(position.entry_price) if position.entry_price is not None else 0.0
        if qty is not None and entry_price > 0:
            notional = float(qty) * entry_price
        elif position.entry_balance is not None and position.size is not None:
            notional = float(position.entry_balance) * float(position.size)
        else:
            return 0.0
        if notional > 0 and math.isfinite(notional):
            fee = float(execution_engine.calculate_entry_fee(notional))
            fee = fee if math.isfinite(fee) and fee >= 0 else 0.0
            logger.info(
                "Reconstructed approximate entry fee $%.4f for %s (no entry-fee "
                "metadata; recovered position) from notional $%.2f",
                fee,
                getattr(position, "symbol", "?"),
                notional,
            )
            return fee
    except (AttributeError, TypeError, ValueError):
        pass
    return 0.0


def _close_position_portion(position: Position) -> float:
    """Fraction of the ORIGINAL position represented by this close
    (``current_size`` / ``original_size``).

    Used to scale the **entry** fee onto a closing ``Trade`` row so its
    ``commission`` matches the row's portion-level ``size``/``quantity``/``pnl``
    (a partially-exited position's final close is only the remaining slice). Returns
    ``1.0`` — i.e. do not scale — for a full close (the common case), for missing or
    corrupt sizing, and for scale-in state (``current_size > original_size``), so the
    entry fee is never inflated.
    """
    original = position.original_size if position.original_size is not None else position.size
    current = position.current_size if position.current_size is not None else position.size
    try:
        original_f = float(original)
        current_f = float(current)
    except (TypeError, ValueError):
        return 1.0
    if not (
        math.isfinite(original_f) and original_f > 0 and math.isfinite(current_f) and current_f >= 0
    ):
        return 1.0
    ratio = current_f / original_f
    if ratio <= 0 or ratio > 1:
        return 1.0
    return ratio
