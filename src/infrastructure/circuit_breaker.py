"""Circuit breaker pattern for fault tolerance.

Implements the circuit breaker pattern to prevent cascading failures and
provide fail-fast behavior when a component repeatedly fails.

Circuit States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing fast, requests immediately fail without calling wrapped function
- HALF_OPEN: Testing if service recovered, allows one test request
"""

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""

    pass


class CircuitBreaker:
    """Circuit breaker for protecting against repeated failures.

    Thread Safety
    -------------
    All state modifications are protected by a lock to ensure safe concurrent access.

    Example
    -------
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=RuntimeError
    )

    @breaker
    def risky_operation():
        # May fail repeatedly
        return call_external_service()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
        name: str | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit.
            recovery_timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN.
            expected_exception: Exception type(s) that count as failures.
            name: Optional name for logging.
        """
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if recovery_timeout <= 0:
            raise ValueError(f"recovery_timeout must be positive, got {recovery_timeout}")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "circuit_breaker"

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state (thread-safe)."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count (thread-safe)."""
        with self._lock:
            return self._failure_count

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state (thread-safe)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info("Circuit breaker '%s' reset to CLOSED", self.name)

    def _record_success(self) -> None:
        """Record successful call (resets failure count)."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "Circuit breaker '%s' test succeeded, transitioning to CLOSED", self.name
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _record_failure(self, exception: Exception) -> None:
        """Record failed call (may open circuit)."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "Circuit breaker '%s' test failed, reopening circuit: %s",
                    self.name,
                    exception,
                )
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                logger.error(
                    "Circuit breaker '%s' opened after %d consecutive failures. "
                    "Last error: %s",
                    self.name,
                    self._failure_count,
                    exception,
                )
                self._state = CircuitState.OPEN

    def _check_and_update_state(self) -> None:
        """Check if circuit should transition from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        "Circuit breaker '%s' transitioning to HALF_OPEN after %.1fs",
                        self.name,
                        elapsed,
                    )
                    self._state = CircuitState.HALF_OPEN

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call wrapped function with circuit breaker protection.

        Args:
            func: Function to call.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func.

        Raises:
            CircuitBreakerError: If circuit is open.
            Exception: Any exception raised by func.
        """
        # Check if we should transition to HALF_OPEN
        self._check_and_update_state()

        # Check current state
        with self._lock:
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN after {self._failure_count} "
                    f"consecutive failures. Will retry after {self.recovery_timeout}s."
                )

        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator interface for circuit breaker.

        Example:
            breaker = CircuitBreaker(failure_threshold=3)

            @breaker
            def my_function():
                return risky_operation()
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(func, *args, **kwargs)

        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics (thread-safe).

        Returns:
            Dictionary with state, failure count, and other metrics.
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._last_failure_time,
                "time_since_last_failure": (
                    time.time() - self._last_failure_time
                    if self._last_failure_time is not None
                    else None
                ),
            }
