from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections.abc import Iterator
from typing import Any

# * Holds per-request/per-cycle logging context
_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "atb_log_context", default=None
)


def get_context() -> dict[str, Any]:
    """Return a shallow copy of the current logging context."""
    ctx = _LOG_CONTEXT.get()
    # Ensure we never leak internal mutable dict
    return dict(ctx) if ctx is not None else {}


def set_context(**kwargs: Any) -> None:
    """Replace or extend the current logging context with provided key/value pairs.

    Keys with value None are ignored.
    """
    ctx = _LOG_CONTEXT.get()
    current = dict(ctx) if ctx is not None else {}
    for key, value in kwargs.items():
        if value is None:
            continue
        current[key] = value
    _LOG_CONTEXT.set(current)


def update_context(**kwargs: Any) -> None:
    """Alias of set_context for readability where we conceptually update."""
    set_context(**kwargs)


def clear_context(*keys: str) -> None:
    """Clear context entirely or specific keys if provided."""
    if not keys:
        _LOG_CONTEXT.set({})
        return
    ctx = _LOG_CONTEXT.get()
    current = dict(ctx) if ctx is not None else {}
    for key in keys:
        current.pop(key, None)
    _LOG_CONTEXT.set(current)


@contextlib.contextmanager
def use_context(**kwargs: Any) -> Iterator[None]:
    """Context manager to temporarily extend the logging context."""
    prev = _LOG_CONTEXT.get()
    try:
        set_context(**kwargs)
        yield
    finally:
        _LOG_CONTEXT.set(prev)


def new_request_id() -> str:
    """Generate a new request/cycle identifier."""
    return uuid.uuid4().hex
