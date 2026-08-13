"""Durable-history lineage for the restart-safe risk seeders (#1036).

Both loop-time seeders — the ``MaxDrawdownGuard`` peak (#1001) and the
``AccountCircuitBreaker`` daily baseline + drawdown peak (#1032) — need to know
which PRIOR trading session holds the ``account_history`` rows they must
baseline from. They used to read ``_recovered_inactive_session_id``, whose
lifetime belongs to a different concern entirely: the #668 carry-forward
re-entry guard clears it in ``LiveStartupSequencer.carry_forward_open_positions``
BEFORE the first loop iteration runs, so on exactly the boot path that needs it
(clean restart → NEW session → positions carried forward) the seeders read an
empty value and silently self-anchored to the post-restart balance — the
#845/#847 peak-reset class the seeding exists to prevent.

The fix is a dedicated field, ``_history_seed_session_id``, written once by
``LiveSessionRecoverer`` when a prior session is found and never cleared: its
lifetime is owned by the seeders' need rather than by the reassign guard. It
also doubles as the honest answer to "does durable history exist at all?",
which is what separates a legitimate self-anchor (genuinely fresh session, no
prior history) from a seeding DEFECT (lineage lost, or the read came back empty
when history was expected).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Seeding provenance values (surfaced on breaker trips as ``peak_seed`` and in
# the guard's arming log). These must tell the truth about where the baseline
# came from — a self-anchor that happened because a lookup FAILED is not the
# same event as a self-anchor because there is genuinely nothing to seed from.
PEAK_SEED_DB_SESSION_MAX = "db_session_max"
"""Seeded from the durable ``account_history`` session max."""

PEAK_SEED_SELF_ANCHORED = "self_anchored"
"""Legitimately anchored to current equity: no prior session, no history."""

PEAK_SEED_UNAVAILABLE = "seed_unavailable"
"""DEFECT: durable history was expected but could not be obtained."""


@runtime_checkable
class HistoryLineageState(Protocol):
    """The engine-state surface the lineage resolver reads."""

    _history_seed_session_id: int | None
    _recovered_inactive_session_id: int | None


@dataclass(frozen=True)
class HistoryLineage:
    """Which session holds the durable history, and whether any is expected."""

    fallback_session_id: int | None
    history_expected: bool

    @property
    def describe(self) -> str:
        """Short provenance string for log lines."""
        if self.fallback_session_id is None:
            return "no prior session"
        return f"prior session {self.fallback_session_id}"


def _as_session_id(value: object) -> int | None:
    """Coerce a session-id attribute to an int, rejecting anything else.

    Deliberately strict: engine state is duck-typed and frequently stubbed, and
    a ``MagicMock`` attribute is truthy — reading one as "a prior session
    exists" would invent history that is not there (LESSONS.md: MagicMock
    truthiness). ``bool`` is excluded because ``True`` is an ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def resolve_history_lineage(state: object) -> HistoryLineage:
    """Resolve the prior session whose ``account_history`` rows seed the risk baselines.

    Prefers the durable ``_history_seed_session_id`` (set once at recovery,
    never cleared) and falls back to ``_recovered_inactive_session_id`` so a
    state object that predates the durable field still seeds as before.
    """
    durable = _as_session_id(getattr(state, "_history_seed_session_id", None))
    legacy = _as_session_id(getattr(state, "_recovered_inactive_session_id", None))
    fallback = durable if durable is not None else legacy
    return HistoryLineage(fallback_session_id=fallback, history_expected=fallback is not None)
