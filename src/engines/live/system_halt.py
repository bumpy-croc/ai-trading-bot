"""Loop-owned mirror of the DB ``system_halt`` manual kill-switch flag (#922)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SystemHaltState:
    """In-memory halt state shared by the loop enforcer and the entry gates.

    ``SystemHaltEnforcer`` writes it after polling the database (a priming
    read at engine construction, then once per trading-loop iteration); the
    ``EntryPauseGate`` instances in the entry coordinator and exit handler
    read it to suppress exposure increases (entries + scale-ins) while exits,
    stops and reconciliation continue. Written and read on the trading-loop
    thread only, matching the loop-owned close-only flag.

    ``established`` is False until the FIRST successful poll. The gates treat
    an unestablished state as halted (fail closed): a boot that cannot verify
    the flag — e.g. an active halt row behind an unreachable database — must
    not trade on the optimistic default.

    ``since`` mirrors the flag row's ``updated_at`` so a monitor can report how
    long the halt has really been in force — a durable timestamp that survives
    the restarts an in-process counter would reset (#1096 review).
    """

    active: bool = False
    reason: str | None = None
    established: bool = False
    since: datetime | None = None
