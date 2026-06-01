"""Process-wide trading-loop liveness heartbeat.

The live engine's trading loop and the embedded ``/health`` HTTP server run in the
same process but different threads, and the health handler holds no reference to
the engine. This module is their shared signal: the loop records a heartbeat each
iteration, and ``/health`` reports unhealthy once it goes stale — so a zombie
process (HTTP server alive, trading loop dead) is detectable instead of always
reporting healthy, which masked the 2026-05-19 silent outage (#627).
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_last_beat: float | None = None  # time.monotonic() of the last loop iteration


def beat() -> None:
    """Record that the trading loop completed an iteration."""
    global _last_beat
    with _lock:
        _last_beat = time.monotonic()


def seconds_since_beat() -> float | None:
    """Seconds since the last loop heartbeat, or None if the loop has not beaten yet."""
    with _lock:
        last = _last_beat
    return None if last is None else time.monotonic() - last


def reset() -> None:
    """Clear the heartbeat (used by tests and when the loop stops cleanly)."""
    global _last_beat
    with _lock:
        _last_beat = None
