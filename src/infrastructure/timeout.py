"""Timeout utilities for protecting against blocking operations.

Provides timeout decorators and context managers for operations that may
block indefinitely (file I/O, model loading, network calls).
"""

import builtins
import logging
import math
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OperationTimeoutError(builtins.TimeoutError):
    """Raised when an operation exceeds its timeout.

    Inherits from the built-in TimeoutError to maintain compatibility
    with code that catches the built-in exception.
    """

    pass


# Backwards compatibility alias
TimeoutError = OperationTimeoutError  # noqa: A001


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for SIGALRM timeout."""
    raise OperationTimeoutError("Operation timed out")


@contextmanager
def timeout_context(seconds: float, operation_name: str = "operation"):
    """Context manager that raises OperationTimeoutError if block exceeds timeout.

    Args:
        seconds: Timeout in seconds (must be positive).
        operation_name: Name of operation for error messages.

    Raises:
        OperationTimeoutError: If the block exceeds the timeout.
        ValueError: If seconds is not positive.
        NotImplementedError: On Windows (signal.SIGALRM not supported).

    Example:
        with timeout_context(30.0, "model loading"):
            model = load_heavy_model()
    """
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError(f"timeout seconds must be a positive finite number, got {seconds}")

    # Check if signal.SIGALRM is available (Unix-like systems)
    if not hasattr(signal, "SIGALRM"):
        logger.warning("timeout_context not supported on this platform (no SIGALRM)")
        yield  # No-op on Windows
        return

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    except OperationTimeoutError:
        logger.error("%s exceeded timeout of %.1fs", operation_name, seconds)
        raise
    finally:
        # Cancel the alarm and restore old handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def with_timeout(
    seconds: float,
    operation_name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds timeout protection to a function.

    Args:
        seconds: Timeout in seconds (must be positive).
        operation_name: Optional name for logging (defaults to function name).

    Returns:
        Decorated function with timeout protection.

    Raises:
        OperationTimeoutError: If the function exceeds the timeout.
        ValueError: If seconds is not positive.

    Example:
        @with_timeout(30.0, "model loading")
        def load_model(path: str):
            return onnx.load(path)
    """
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError(f"timeout seconds must be a positive finite number, got {seconds}")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with timeout_context(seconds, op_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def run_with_timeout(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: float = 30.0,
    operation_name: str | None = None,
) -> T:
    """Run a function with timeout using threading (cross-platform).

    This is a cross-platform alternative to signal-based timeouts.
    Works on Windows and Unix-like systems.

    Limitations
    -----------
    The threading-based timeout cannot cancel a running thread. If the
    operation times out, the daemon thread continues executing in the
    background until completion. For truly blocking operations, consider
    using process-based isolation or async I/O with proper cancellation.

    Args:
        func: Function to run.
        args: Positional arguments for func.
        kwargs: Keyword arguments for func.
        timeout_seconds: Timeout in seconds (must be positive).
        operation_name: Optional name for logging.

    Returns:
        Result of func.

    Raises:
        OperationTimeoutError: If func exceeds timeout.
        ValueError: If timeout_seconds is not positive.
        Exception: Any exception raised by func.

    Example:
        result = run_with_timeout(
            load_model,
            args=("model.onnx",),
            timeout_seconds=30.0,
            operation_name="ONNX model loading"
        )
    """
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError(f"timeout_seconds must be a positive finite number, got {timeout_seconds}")
    if kwargs is None:
        kwargs = {}
    op_name = operation_name or func.__name__
    start_time = time.time()

    # Use mutable container for thread result/exception
    # These are local to each call and garbage collected after return
    result: list[T] = []
    exception: list[BaseException] = []

    def target():
        try:
            result.append(func(*args, **kwargs))
        except BaseException as e:
            # Catch BaseException to handle SystemExit, KeyboardInterrupt, etc.
            # This ensures exception list is always populated if func fails
            exception.append(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        elapsed = time.time() - start_time
        logger.error(
            "%s exceeded timeout of %.1fs (ran for %.2fs before timeout)",
            op_name,
            timeout_seconds,
            elapsed,
        )
        # Clear references to prevent holding onto large objects
        result.clear()
        exception.clear()
        raise OperationTimeoutError(f"{op_name} exceeded timeout of {timeout_seconds}s")

    if exception:
        exc = exception[0]
        exception.clear()
        raise exc

    if result:
        res = result[0]
        result.clear()
        return res

    raise RuntimeError(f"{op_name} completed but returned no result")
