"""Escalating re-page schedule shared by every latched-condition announcer.

A condition that holds for a month must produce ~31 pages, not ~720: a few in
the first day, then one a day. The in-process ``LatchedConditionMonitor`` and
the external write-gap watchdog both follow this schedule, so an operator reads
one cadence regardless of which watcher raised the page.

Kept free of engine imports so a CLI watchdog can use it without loading the
live-trading package.
"""

from __future__ import annotations

# Escalating re-pages measured from the latch: 1h, 4h, 12h...
PAGE_SCHEDULE_SECONDS = (3600.0, 14400.0, 43200.0)
# ...then once a day for as long as it holds. Bound: <=3 re-pages in the first
# 24h and <=1 per 24h afterwards.
PAGE_STEADY_INTERVAL_SECONDS = 86400.0


def repage_offset(index: int) -> float:
    """Seconds after the latch at which re-page number ``index`` (0-based) is due."""
    if index < 0:
        raise ValueError("re-page index must be >= 0")
    if index < len(PAGE_SCHEDULE_SECONDS):
        return PAGE_SCHEDULE_SECONDS[index]
    steady_pages = index - len(PAGE_SCHEDULE_SECONDS) + 1
    return PAGE_SCHEDULE_SECONDS[-1] + steady_pages * PAGE_STEADY_INTERVAL_SECONDS


def format_elapsed(seconds: float) -> str:
    """Human-readable elapsed time, e.g. ``3d 4h``, ``5h 12m``, ``45m``."""
    total = max(int(seconds), 0)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
