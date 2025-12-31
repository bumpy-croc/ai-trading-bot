"""Timeout utilities for protecting against blocking operations.

Provides timeout decorators and context managers for operations that may
block indefinitely (file I/O, model loading, network calls).
"""

import logging
import signal
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""

    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for SIGALRM timeout."""
    raise TimeoutError("Operation timed out")


@contextmanager
def timeout_context(seconds: float, operation_name: str = "operation"):
    """Context manager that raises TimeoutError if block exceeds timeout.

    Args:
        seconds: Timeout in seconds.
        operation_name: Name of operation for error messages.

    Raises:
        TimeoutError: If the block exceeds the timeout.
        NotImplementedError: On Windows (signal.SIGALRM not supported).

    Example:
        with timeout_context(30.0, "model loading"):
            model = load_heavy_model()
    """
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
    except TimeoutError:
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
        seconds: Timeout in seconds.
        operation_name: Optional name for logging (defaults to function name).

    Returns:
        Decorated function with timeout protection.

    Raises:
        TimeoutError: If the function exceeds the timeout.

    Example:
        @with_timeout(30.0, "model loading")
        def load_model(path: str):
            return onnx.load(path)
    """

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

    Args:
        func: Function to run.
        args: Positional arguments for func.
        kwargs: Keyword arguments for func.
        timeout_seconds: Timeout in seconds.
        operation_name: Optional name for logging.

    Returns:
        Result of func.

    Raises:
        TimeoutError: If func exceeds timeout.
        Exception: Any exception raised by func.

    Example:
        result = run_with_timeout(
            load_model,
            args=("model.onnx",),
            timeout_seconds=30.0,
            operation_name="ONNX model loading"
        )
    """
    if kwargs is None:
        kwargs = {}
    op_name = operation_name or func.__name__

    result = []
    exception = []

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
        logger.error("%s exceeded timeout of %.1fs", op_name, timeout_seconds)
        raise TimeoutError(f"{op_name} exceeded timeout of {timeout_seconds}s")

    if exception:
        raise exception[0]

    if result:
        return result[0]

    raise RuntimeError(f"{op_name} completed but returned no result")
