"""Entry/scale-in gate shared by the live entry and scale-in paths.

Two operator levers converge here, both with identical no-new-risk semantics
(exposure must not INCREASE: new entries and scale-ins are skipped, while
exits, partial exits, stop-loss management, reconciliation and monitoring
continue):

- ``FEATURE_ENTRY_PAUSE`` — env-var flag; requires a restart/redeploy to flip.
- The manual kill-switch (#922) — the DB ``system_halt`` flag mirrored into a
  ``SystemHaltState`` by the loop enforcer; takes effect without a restart.
"""

from __future__ import annotations

import logging
import time

from src.config.constants import ENTRY_PAUSE_WARNING_INTERVAL_SECONDS
from src.config.feature_flags import is_enabled
from src.engines.live.system_halt import SystemHaltState

logger = logging.getLogger(__name__)


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
        cause = self._active_cause()
        if cause is None:
            return False
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

    def _active_cause(self) -> str | None:
        """The active pause source's log prefix, or None when not paused."""
        if self._halt_state is not None and self._halt_state.active:
            return (
                "MANUAL SYSTEM HALT active "
                f"(reason: {self._halt_state.reason or 'no reason recorded'})"
            )
        if is_enabled("entry_pause", default=False):
            return "FEATURE_ENTRY_PAUSE active"
        return None
