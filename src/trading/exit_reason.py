"""Closed taxonomy of position-exit categories, shared by the backtest and live engines.

``trades.exit_reason`` stays free text — operators read it, and the live engine embeds it
in the ``account_balances`` ledger key (``realized_pnl_<SYM>_<reason>``), so its values are
historical record and must not be renamed. The category in this module is the *typed*
companion: every exit decision carries an :class:`ExitReason`, control flow branches on
that enum, and it is persisted alongside the prose in ``trades.exit_category``.

The distinction that motivated the taxonomy (GH #1115): a stop that **protected** capital
and a trailing stop that **took profit** are opposite outcomes, and prod recorded both as
"Stop loss". Which one occurred is not derivable from the prose — it is derivable from the
position's ``trailing_stop_activated`` / ``breakeven_triggered`` flags, which both engines
maintain through the shared ``TrailingStopManager``. :func:`classify_stop_exit` is the one
place that reads them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

__all__ = [
    "LEGACY_EXIT_REASON_CATEGORIES",
    "STOP_EXIT_CATEGORIES",
    "ExitReason",
    "classify_stop_exit",
    "coerce_exit_category",
    "infer_legacy_category",
]


class ExitReason(StrEnum):
    """Why a position was closed. Values are stable wire/DB tokens — never rename one.

    A category answers "what kind of event ended this position", not "under what
    circumstances was it recorded". An exchange stop that filled while the bot was offline
    is still the same logical exit as one the engine observed live; the circumstance lives
    in the free-text detail, so that grouping by category does not re-fragment the very
    thing this enum exists to unify.
    """

    STOP_LOSS = "stop_loss"
    """Protective stop hit before the stop was ever moved off its initial level."""

    BREAKEVEN_STOP = "breakeven_stop"
    """Stop hit after it was pulled to breakeven but before trailing activated."""

    TRAILING_STOP = "trailing_stop"
    """Trailing stop hit after activation — a profit-taking exit, not a protective one."""

    TAKE_PROFIT = "take_profit"
    SIGNAL_EXIT = "signal_exit"
    """Strategy signal or signal reversal asked to close."""

    TIME_EXIT = "time_exit"
    """Holding-period, weekend-flat, or end-of-day-flat policy closed the position."""

    EARLY_CUT = "early_cut"
    """MFE early-cut policy closed a position that failed to make progress."""

    PARTIAL_EXIT_COMPLETE = "partial_exit_complete"
    """Scaled-out targets consumed the whole position; the remainder was closed."""

    EMERGENCY_CLOSE = "emergency_close"
    """Operational close forced by a failure (stop-loss placement, risk-manager sync)."""

    ENGINE_SHUTDOWN = "engine_shutdown"
    STRATEGY_CHANGE = "strategy_change"
    """Position closed because the running strategy was swapped out."""

    EXTERNAL_CLOSE = "external_close"
    """Closed outside the bot (manual or exchange action), detected by reconciliation."""

    RECOVERED = "recovered"
    """Reconstructed after the fact from exchange state; the true trigger is unknown."""

    UNKNOWN = "unknown"
    """No category was supplied. Never write this deliberately."""


STOP_EXIT_CATEGORIES: frozenset[ExitReason] = frozenset(
    {ExitReason.STOP_LOSS, ExitReason.BREAKEVEN_STOP, ExitReason.TRAILING_STOP}
)
"""Categories that execute as a stop order and price through the stop level on a gap."""


def _is_set(position: Any, flag: str) -> bool:
    """Read a boolean position flag strictly.

    ``is True`` rather than truthiness: test doubles expose every attribute as a truthy
    ``MagicMock``, which would silently classify every stop exit as trailing
    (``.claude/LESSONS.md`` — MagicMock reads ``getattr(m, "flag", False)`` as truthy).
    """
    return getattr(position, flag, False) is True


def classify_stop_exit(position: Any) -> ExitReason:
    """Categorize a stop-level exit by how far the stop had been moved.

    Args:
        position: The position being closed. Reads ``trailing_stop_activated`` and
            ``breakeven_triggered``, which both engines maintain via the shared
            ``TrailingStopManager`` and which survive a restart on the ``positions`` row.

    Returns:
        ``TRAILING_STOP`` once trailing activated, ``BREAKEVEN_STOP`` if the stop only
        reached breakeven, otherwise ``STOP_LOSS``.
    """
    if _is_set(position, "trailing_stop_activated"):
        return ExitReason.TRAILING_STOP
    if _is_set(position, "breakeven_triggered"):
        return ExitReason.BREAKEVEN_STOP
    return ExitReason.STOP_LOSS


LEGACY_EXIT_REASON_CATEGORIES: dict[str, ExitReason] = {
    # ---- prose emitted by the engines before #1115 ----
    "Stop loss": ExitReason.STOP_LOSS,
    "stop_loss": ExitReason.STOP_LOSS,
    "stop_loss_offline": ExitReason.STOP_LOSS,
    "stop_loss_filled_offline": ExitReason.STOP_LOSS,
    "Take profit": ExitReason.TAKE_PROFIT,
    "take_profit": ExitReason.TAKE_PROFIT,
    "Signal reversal": ExitReason.SIGNAL_EXIT,
    "Strategy signal": ExitReason.SIGNAL_EXIT,
    "Time exit": ExitReason.TIME_EXIT,
    "time_exit": ExitReason.TIME_EXIT,
    "Max holding period": ExitReason.TIME_EXIT,
    "Weekend flat": ExitReason.TIME_EXIT,
    "End of day flat": ExitReason.TIME_EXIT,
    "Engine shutdown": ExitReason.ENGINE_SHUTDOWN,
    "Strategy change - close requested": ExitReason.STRATEGY_CHANGE,
    "Stop-loss placement failed - emergency close": ExitReason.EMERGENCY_CLOSE,
    "Risk manager sync failure": ExitReason.EMERGENCY_CLOSE,
    "manual_close": ExitReason.EXTERNAL_CLOSE,
    "external_close_recovery": ExitReason.EXTERNAL_CLOSE,
    "recovered_from_exchange": ExitReason.RECOVERED,
    "exit_order_recovery": ExitReason.RECOVERED,
}
"""Exact historical ``exit_reason`` strings mapped to their category.

Backfill/reporting only — **never** consult this from engine control flow. It cannot
recover the protected-vs-trailing distinction, because the prose never carried it: every
legacy stop spelling maps to ``STOP_LOSS`` and any row it classifies is *inferred*, not
known. `agents/research/1115-exit-taxonomy.md` records the per-row prod mapping, which
recovers the real category from the ``positions`` row instead.
"""


def infer_legacy_category(exit_reason: str | None) -> ExitReason:
    """Best-effort category for a pre-#1115 row, for reporting over historical data.

    Args:
        exit_reason: The stored free-text reason.

    Returns:
        The mapped category, or ``UNKNOWN`` when the text is absent or unrecognised.
        Prefixed variants that carry a runtime suffix (early cut, partial exits) are
        matched on their stable prefix.
    """
    if not exit_reason:
        return ExitReason.UNKNOWN
    mapped = LEGACY_EXIT_REASON_CATEGORIES.get(exit_reason)
    if mapped is not None:
        return mapped
    if exit_reason.startswith("Early cut"):
        return ExitReason.EARLY_CUT
    if exit_reason.startswith("Partial exits complete"):
        return ExitReason.PARTIAL_EXIT_COMPLETE
    return ExitReason.UNKNOWN


def coerce_exit_category(value: object) -> ExitReason:
    """Coerce an arbitrary value to a category, defaulting to ``UNKNOWN``.

    Used at boundaries that receive the category from a caller we do not control —
    a persisted string, or a test double whose attributes are ``MagicMock``. Never
    raises, so a malformed category can degrade a report but not abort a close.
    """
    if isinstance(value, ExitReason):
        return value
    if isinstance(value, str):
        try:
            return ExitReason(value)
        except ValueError:
            return ExitReason.UNKNOWN
    return ExitReason.UNKNOWN
