"""Entry/scale-in gate shared by the live entry and scale-in paths.

Two operator levers converge here, both with identical no-new-risk semantics
(exposure must not INCREASE: new entries and scale-ins are skipped, while
exits, partial exits, stop-loss management, reconciliation and monitoring
continue):

- ``FEATURE_ENTRY_PAUSE`` — env-var flag; requires a restart/redeploy to flip.
- The manual kill-switch (#922) — the DB ``system_halt`` flag mirrored into a
  ``SystemHaltState`` by the loop enforcer; takes effect without a restart.
  A halt state that was never successfully read (``established=False``) gates
  as if halted — fail closed until the database confirms otherwise.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.config.constants import ENTRY_PAUSE_WARNING_INTERVAL_SECONDS
from src.config.feature_flags import is_enabled
from src.engines.live.system_halt import SystemHaltState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntryBlock:
    """The single authoritative answer to "why are entries blocked right now?".

    ``key`` identifies the lever (``system_halt`` / ``entry_pause``) so monitors
    can label the condition; ``reason`` is the operator-facing text, reused
    verbatim in the skip log and in the latched-condition announcements. One
    source of truth: a monitor that re-derives this can drift out of agreement
    with the gate the entry path actually consults (#1096 review).
    """

    key: str
    reason: str


class EntryPauseGate:
    """Rate-limit-logging gate for the ``entry_pause`` flag and manual halt.

    Each consumer holds its own gate instance so warnings rate-limit per
    path. State is written only from the trading-loop thread; a benign
    race would at worst duplicate a log line.
    """

    def __init__(self, halt_state: SystemHaltState | None = None) -> None:
        """Bind the optional manual-halt state; first skip always warns."""
        self._halt_state = halt_state
        self._last_warning: float | None = None

    def paused(self, context: str) -> bool:
        """True when a pause source suppresses the given action; logs rate-limited.

        Warns at most once per ENTRY_PAUSE_WARNING_INTERVAL_SECONDS to avoid
        log spam from the trading loop.
        """
        block = self.entry_block()
        if block is None:
            return False
        cause = block.reason
        now = time.monotonic()
        if (
            self._last_warning is None
            or now - self._last_warning >= ENTRY_PAUSE_WARNING_INTERVAL_SECONDS
        ):
            self._last_warning = now
            logger.warning(
                "%s — skipping %s "
                "(exits, partial exits, stop-loss management and reconciliation continue)",
                cause,
                context,
            )
        else:
            logger.debug("%s — skipping %s", cause, context)
        return True

    def entry_blocks(self) -> tuple[EntryBlock, ...]:
        """EVERY lever currently blocking entries, in precedence order.

        The authority for "are entries blocked": the entry/scale-in paths gate
        on it and the latched-condition monitor announces it, so the two cannot
        disagree about a blocked state (#1096 review — a monitor blind to the
        fail-closed unverified case reproduced #1094's exact shape). All active
        levers are reported, not just the first: a monitor that saw only the
        winning cause would record the shadowed one as *cleared* while it is
        still in force.
        """
        blocks: list[EntryBlock] = []
        if self._halt_state is not None:
            if not self._halt_state.established:
                # Fail closed: the halt flag has never been successfully read
                # (e.g. DB unreachable at boot) — do not add risk on the
                # optimistic default.
                blocks.append(
                    EntryBlock(
                        "system_halt",
                        "MANUAL SYSTEM HALT state UNVERIFIED "
                        "(system_halt flag not successfully read yet — failing closed)",
                    )
                )
            elif self._halt_state.active:
                blocks.append(
                    EntryBlock(
                        "system_halt",
                        "MANUAL SYSTEM HALT active "
                        f"(reason: {self._halt_state.reason or 'no reason recorded'})",
                    )
                )
        if is_enabled("entry_pause", default=False):
            blocks.append(EntryBlock("entry_pause", "FEATURE_ENTRY_PAUSE active"))
        return tuple(blocks)

    def entry_block(self) -> EntryBlock | None:
        """The highest-precedence active pause source, or None when allowed."""
        blocks = self.entry_blocks()
        return blocks[0] if blocks else None

    def entry_block_reason(self) -> str | None:
        """The active pause source's log prefix, or None when not paused."""
        block = self.entry_block()
        return None if block is None else block.reason
