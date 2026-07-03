"""FEATURE_ENTRY_PAUSE gate shared by the live entry and scale-in paths."""

from __future__ import annotations

import logging
import time

from src.config.constants import ENTRY_PAUSE_WARNING_INTERVAL_SECONDS
from src.config.feature_flags import is_enabled

logger = logging.getLogger(__name__)


class EntryPauseGate:
    """Rate-limit-logging gate for the ``entry_pause`` feature flag.

    When FEATURE_ENTRY_PAUSE is truthy the live engine must not INCREASE
    exposure: new entries and scale-ins are skipped, while exits, partial
    exits, stop-loss management, reconciliation and monitoring continue.
    The flag lets a human flatten risk ahead of macro events (FOMC/CPI)
    with a single env var.

    Each consumer holds its own gate instance so warnings rate-limit per
    path. State is written only from the trading-loop thread; a benign
    race would at worst duplicate a log line.
    """

    def __init__(self) -> None:
        """Initialize with no warning emitted yet (first skip always warns)."""
        self._last_warning: float | None = None

    def paused(self, context: str) -> bool:
        """True when the flag suppresses the given action; logs rate-limited.

        Warns at most once per ENTRY_PAUSE_WARNING_INTERVAL_SECONDS to avoid
        log spam from the trading loop.
        """
        if not is_enabled("entry_pause", default=False):
            return False
        now = time.monotonic()
        if (
            self._last_warning is None
            or now - self._last_warning >= ENTRY_PAUSE_WARNING_INTERVAL_SECONDS
        ):
            self._last_warning = now
            logger.warning(
                "FEATURE_ENTRY_PAUSE active — skipping %s "
                "(exits, partial exits, stop-loss management and reconciliation continue)",
                context,
            )
        else:
            logger.debug("FEATURE_ENTRY_PAUSE active — skipping %s", context)
        return True
