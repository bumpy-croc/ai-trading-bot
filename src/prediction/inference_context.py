"""Scoped inference execution context.

Backtest results must be bit-identical run-to-run regardless of CPU load, so
in the DETERMINISTIC context (the default) inference is never aborted on
wall-clock time — a load-dependent timeout silently substituting a failed
prediction changes trade sequences between identical runs (#912 side-finding).

Live trading opts into the LIVE context, where inference runs under a bounded
latency budget (``PredictionConfig.live_inference_timeout``) so the trading
loop cannot block indefinitely on a hung model. Timeouts there are accounted
loudly: WARNING log, engine counter, and a ``timed_out`` result stamp.

The context is a :class:`contextvars.ContextVar`, not a process-wide global
(#926): constructing a ``Backtester`` in the same process as a live engine
(e.g. an in-process validation gate) must never strip the live inference
deadline. Policy is selected where inference actually runs — the live
trading loop pins LIVE for its thread, and ``Backtester.run`` executes under
an :func:`inference_scope` that restores the caller's policy on exit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum


class InferenceContext(Enum):
    """Latency policy for model inference."""

    DETERMINISTIC = "deterministic"
    LIVE = "live"


_context_var: ContextVar[InferenceContext] = ContextVar(
    "inference_context", default=InferenceContext.DETERMINISTIC
)


def _validated(context: InferenceContext) -> InferenceContext:
    if not isinstance(context, InferenceContext):
        raise ValueError(f"context must be an InferenceContext, got {context!r}")
    return context


def get_inference_context() -> InferenceContext:
    """Return the inference context for the current thread/context."""
    return _context_var.get()


@contextmanager
def inference_scope(context: InferenceContext) -> Iterator[None]:
    """Run a block under ``context``, restoring the previous policy on exit.

    This is the composition-safe way to select a policy: a deterministic
    backtest nested inside a live process gets its reproducibility guarantee
    for exactly the duration of the run, then the live deadline is restored.

    Raises:
        ValueError: If ``context`` is not an :class:`InferenceContext`.
    """
    token = _context_var.set(_validated(context))
    try:
        yield
    finally:
        _context_var.reset(token)


def set_inference_context(context: InferenceContext) -> None:
    """Pin the ambient context for the current thread/context (no restore).

    ContextVar values do not propagate to threads started afterwards, so this
    only affects code running in the caller's context — use
    :func:`inference_scope` for anything that should compose or restore.
    ``LiveTradingEngine`` pins LIVE at construction for its constructing
    thread; the trading loop thread scopes itself.

    Raises:
        ValueError: If ``context`` is not an :class:`InferenceContext`.
    """
    _context_var.set(_validated(context))


def reset_inference_context() -> None:
    """Restore the default (deterministic) context. Intended for tests."""
    _context_var.set(InferenceContext.DETERMINISTIC)
