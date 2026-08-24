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
when history was expected). ``_history_seed_lookup_failed`` covers the third
case — the lookup itself raised — so "could not determine" never presents as
"nothing to determine".

Drill note (#1036 P4 restart drill, ``[D-2026-08-13-01]``): the drill must
assert the seeded peak VALUE equals the prior session's ``account_history``
max, NOT merely that the provenance reads ``db_session_max``.
``DatabaseManager._get_session_peak`` is a UNION max over
``session_id IN (new, prior)``, so it returns a non-``None`` value — and the
seeders latch ``db_session_max`` — even when the only contributing row is the
NEW session's own opening snapshot. The string alone would therefore pass on a
build where the lineage was lost again. (Being a union max, row ORDER is
irrelevant: an early new-session row cannot shadow the prior session's higher
peak.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


class HistoryLineageState(Protocol):
    """The engine-state surface the lineage resolver reads."""

    trading_session_id: int | None
    _history_seed_session_id: int | None
    _history_seed_lookup_failed: bool


@dataclass(frozen=True)
class HistoryLineage:
    """Which session holds the durable history, and whether any is expected."""

    fallback_session_id: int | None
    history_expected: bool
    lookup_failed: bool = False
    current_session_id: int | None = None

    @property
    def is_distinct_prior_session(self) -> bool:
        """True when the history lives under a session OTHER than the live one.

        Separates the clean-restart boot (NEW session, history under the prior
        one) from the session-REUSE boot (crash recovery reuses the session, so
        the lineage id IS the live one). Both expect history; only the former
        can lose it to a lineage bug, which is why the "seeding failed"
        diagnostics must not describe them identically.
        """
        return (
            self.fallback_session_id is not None
            and self.fallback_session_id != self.current_session_id
        )

    @property
    def describe(self) -> str:
        """Short provenance string for log lines."""
        if self.fallback_session_id is None:
            return "prior-session lookup FAILED" if self.lookup_failed else "no prior session"
        if not self.is_distinct_prior_session:
            return f"reused session {self.fallback_session_id}"
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


def resolve_history_lineage(state: HistoryLineageState) -> HistoryLineage:
    """Resolve the prior session whose ``account_history`` rows seed the risk baselines.

    Reads ONLY the durable ``_history_seed_session_id`` (set once at recovery,
    never cleared). ``_recovered_inactive_session_id`` is deliberately not
    consulted: its lifetime belongs to the #668 carry-forward guard, and a
    fallback to it would leave the seeding contract hostage to that field's
    lifetime again.

    ``_history_seed_lookup_failed`` makes an *undetermined* lineage expect
    history: a recovery lookup that raised could not prove the account is
    fresh, so treating it as "nothing to seed from" would launder a defect
    into a legitimate self-anchor. Read with ``is True`` because engine state
    is routinely stubbed and a ``MagicMock`` attribute is truthy
    (LESSONS.md: MagicMock truthiness).
    """
    fallback = _as_session_id(getattr(state, "_history_seed_session_id", None))
    lookup_failed = getattr(state, "_history_seed_lookup_failed", False) is True
    return HistoryLineage(
        fallback_session_id=fallback,
        history_expected=fallback is not None or lookup_failed,
        lookup_failed=lookup_failed,
        current_session_id=_as_session_id(getattr(state, "trading_session_id", None)),
    )
