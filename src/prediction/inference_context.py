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

import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InferenceContext(Enum):
    """Latency policy for model inference."""

    DETERMINISTIC = "deterministic"
    LIVE = "live"


_context_var: ContextVar[InferenceContext] = ContextVar(
    "inference_context", default=InferenceContext.DETERMINISTIC
)


# True once a policy was chosen explicitly for this context (inference_scope /
# set_inference_context). Lets the live-process guard tell a deliberate
# DETERMINISTIC scope (a nested backtest) from a thread that never chose one.
_explicit_var: ContextVar[bool] = ContextVar("inference_context_explicit", default=False)

_live_process_registered = False
_unscoped_warned_threads: set[int] = set()


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
    explicit_token = _explicit_var.set(True)
    try:
        yield
    finally:
        _explicit_var.reset(explicit_token)
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
    _explicit_var.set(True)


def reset_inference_context() -> None:
    """Restore the default (deterministic) context. Intended for tests."""
    global _live_process_registered
    _context_var.set(InferenceContext.DETERMINISTIC)
    _explicit_var.set(False)
    _live_process_registered = False
    _unscoped_warned_threads.clear()


def register_live_process() -> None:
    """Declare that this process runs live trading.

    Enables :func:`warn_if_unscoped_in_live_process`. Called by
    ``LiveTradingEngine`` at construction.
    """
    global _live_process_registered
    _live_process_registered = True


def spawn_live_thread(
    target: Callable[..., Any],
    *,
    name: str | None = None,
    daemon: bool = True,
    args: Iterable[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> threading.Thread:
    """Build (not start) a thread whose body runs under the LIVE inference scope.

    New threads do not inherit the parent's context variables, so a bare
    ``threading.Thread`` in a live process infers with no deadline — it fails
    open. Every engine-started thread must be created through this helper.
    """
    call_args = tuple(args)
    call_kwargs = dict(kwargs or {})

    def _run() -> None:
        with inference_scope(InferenceContext.LIVE):
            target(*call_args, **call_kwargs)

    return threading.Thread(target=_run, name=name, daemon=daemon)


def warn_if_unscoped_in_live_process() -> None:
    """Log once per thread when inference runs unscoped in a live process.

    An unscoped thread silently gets the default DETERMINISTIC policy, i.e.
    no inference deadline. Threads that chose a policy explicitly (a nested
    backtest's DETERMINISTIC scope, or LIVE) are not flagged.
    """
    if not _live_process_registered or _explicit_var.get():
        return
    ident = threading.get_ident()
    if ident in _unscoped_warned_threads:
        return
    _unscoped_warned_threads.add(ident)
    logger.error(
        "Inference on thread %r has no inference scope in a live process, so it runs with "
        "NO deadline. Start engine threads via spawn_live_thread() or wrap the work in "
        "inference_scope(InferenceContext.LIVE).",
        threading.current_thread().name,
    )
