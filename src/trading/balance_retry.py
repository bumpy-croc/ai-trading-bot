"""Retry a free-balance read that may still reflect a just-cancelled order.

Cancelling a resting order and immediately reading an exchange balance can
observe the pre-cancel snapshot for several seconds: Binance's cross-margin
wallet endpoint (``/sapi/v1/margin/account``) is eventually consistent, so a
freshly-freed ``locked`` amount can keep reporting as unavailable for a short
window after the cancel confirms (#1165).

Two independent call sites hit this same race immediately after cancelling a
stop-loss: the close path (``LiveExecutionEngine._free_base_for_close``) and
the re-protect path (``BinanceProvider.place_stop_loss_order``, #1173). Both
need the identical retry-while-stale loop, so it lives here once rather than
being copied a third time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Empirically, Binance's margin wallet read settles well within this budget
# (#1165); the loop returns as soon as a read clears ``min_required`` rather
# than always paying the full budget.
POST_CANCEL_BALANCE_RETRY_ATTEMPTS = 5
POST_CANCEL_BALANCE_RETRY_DELAY_SECONDS = 0.3


def read_free_balance_with_retry(
    read_balance: Callable[[], float | None],
    *,
    min_required: float | None,
    context: str,
    attempts: int = POST_CANCEL_BALANCE_RETRY_ATTEMPTS,
    delay_seconds: float = POST_CANCEL_BALANCE_RETRY_DELAY_SECONDS,
) -> float | None:
    """Read a free balance, retrying while it looks like a stale post-cancel snapshot.

    ``read_balance`` must fail open on its own (return ``None`` on any lookup
    or conversion error) — this loop does not catch exceptions from it, so a
    read that raises propagates past a single call rather than being retried
    or swallowed here. Every current caller's ``read_balance`` already wraps
    its own try/except for exactly this reason.

    When ``min_required`` is ``None`` (no cancel just happened, so a low
    reading isn't "stale" — it's just correctly low), this makes exactly one
    read and returns it, paying no retry latency. Only a caller that knows a
    cancel just preceded this read should pass ``min_required``.

    Returns the last read once either: the read clears ``min_required`` (is
    no longer stale), the read comes back ``None`` (a lookup failure — never
    retried, so a transient error doesn't burn the whole budget), or the
    attempt budget is exhausted (the final, still-stale read is returned
    unchanged so the caller's own gate can still refuse honestly-locked
    inventory — this loop only removes the false-positive delay).
    """
    effective_attempts = attempts if min_required is not None else 1
    free: float | None = None
    for attempt in range(effective_attempts):
        free = read_balance()
        is_stale = free is not None and min_required is not None and free < min_required
        if not is_stale or attempt == effective_attempts - 1:
            return free
        logger.info(
            "Free balance %.8f for %s is still below the %.8f required after a "
            "cancel (attempt %d/%d) — retrying after a short settlement wait.",
            free,
            context,
            min_required,
            attempt + 1,
            effective_attempts,
        )
        time.sleep(delay_seconds)
    return free
