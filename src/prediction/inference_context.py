"""Process-wide inference execution context.

Backtest results must be bit-identical run-to-run regardless of CPU load, so
in the DETERMINISTIC context (the default) inference is never aborted on
wall-clock time — a load-dependent timeout silently substituting a failed
prediction changes trade sequences between identical runs (#912 side-finding).

Live trading opts into the LIVE context, where inference runs under a bounded
latency budget (``PredictionConfig.live_inference_timeout``) so the trading
loop cannot block indefinitely on a hung model. Timeouts there are accounted
loudly: WARNING log, engine counter, and a ``timed_out`` result stamp.

The context is pinned by engine constructors (``Backtester`` -> DETERMINISTIC,
``LiveTradingEngine`` -> LIVE); nothing else should need to set it.
"""

from __future__ import annotations

import threading
from enum import Enum


class InferenceContext(Enum):
    """Latency policy for model inference."""

    DETERMINISTIC = "deterministic"
    LIVE = "live"


_lock = threading.Lock()
_context: InferenceContext = InferenceContext.DETERMINISTIC


def get_inference_context() -> InferenceContext:
    """Return the current inference context."""
    return _context


def set_inference_context(context: InferenceContext) -> None:
    """Set the process-wide inference context.

    Raises:
        ValueError: If ``context`` is not an :class:`InferenceContext`.
    """
    if not isinstance(context, InferenceContext):
        raise ValueError(f"context must be an InferenceContext, got {context!r}")
    global _context
    with _lock:
        _context = context


def reset_inference_context() -> None:
    """Restore the default (deterministic) context. Intended for tests."""
    set_inference_context(InferenceContext.DETERMINISTIC)
