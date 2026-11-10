"""Thread factory for engine-started threads in a live process.

New threads do not inherit the parent's context variables, so a bare
``threading.Thread`` in a live process runs inference under the default
DETERMINISTIC policy, i.e. with no deadline (the fail-open shape of #1015).
Every engine-started thread is created through :func:`create_live_thread`.
"""

import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from src.prediction.inference_context import InferenceContext, inference_scope


def create_live_thread(
    target: Callable[..., Any],
    *,
    name: str | None = None,
    daemon: bool = True,
    args: Iterable[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> threading.Thread:
    """Build (not start) a thread whose body runs under the LIVE inference scope."""
    call_args = tuple(args)
    call_kwargs = dict(kwargs or {})

    def _run() -> None:
        with inference_scope(InferenceContext.LIVE):
            target(*call_args, **call_kwargs)

    return threading.Thread(target=_run, name=name, daemon=daemon)
