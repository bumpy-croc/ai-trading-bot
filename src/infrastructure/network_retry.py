"""Network retry utilities with exponential backoff.

Provides reusable retry decorators for network operations with configurable
retry strategies, exponential backoff, and jitter.
"""

import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retryable HTTP status codes (transient errors)
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Retryable exception types
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
)


def with_network_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_status_codes: set[int] | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for network operations with exponential backoff and jitter.

    Retries on:
    - Network errors (timeout, connection errors)
    - Transient HTTP errors (429, 5xx)
    - Configurable exception types and status codes

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        retryable_status_codes: HTTP status codes to retry (default: RETRYABLE_STATUS_CODES)
        retryable_exceptions: Exception types to retry (default: RETRYABLE_EXCEPTIONS)

    Returns:
        Decorated function with retry logic

    Example:
        @with_network_retry(max_retries=5, base_delay=2.0)
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """
    if retryable_status_codes is None:
        retryable_status_codes = RETRYABLE_STATUS_CODES
    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_EXCEPTIONS

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    # Network-level errors - always retryable
                    last_exception = e
                    if attempt < max_retries:
                        delay = _calculate_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        logger.warning(
                            "Network error in %s: %s. Retrying in %.2fs (attempt %d/%d)",
                            func.__name__,
                            str(e),
                            delay,
                            attempt + 1,
                            max_retries,
                            exc_info=False,
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            "Network error in %s after %d retries: %s",
                            func.__name__,
                            max_retries,
                            str(e),
                            exc_info=True,
                        )
                        raise

                except requests.exceptions.HTTPError as e:
                    # HTTP-level errors - check if retryable by status code
                    # Safely access status_code with validation to prevent AttributeError
                    status_code = getattr(e.response, "status_code", None) if e.response else None
                    last_exception = e

                    if status_code is not None and status_code in retryable_status_codes:
                        if attempt < max_retries:
                            delay = _calculate_delay(
                                attempt, base_delay, max_delay, exponential_base, jitter
                            )
                            logger.warning(
                                "HTTP error %s in %s: %s. Retrying in %.2fs (attempt %d/%d)",
                                status_code,
                                func.__name__,
                                str(e),
                                delay,
                                attempt + 1,
                                max_retries,
                                exc_info=False,
                            )
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(
                                "HTTP error %s in %s after %d retries: %s",
                                status_code,
                                func.__name__,
                                max_retries,
                                str(e),
                                exc_info=True,
                            )
                            raise
                    else:
                        # Non-retryable HTTP error (4xx auth, etc.)
                        logger.error(
                            "Non-retryable HTTP error %s in %s: %s",
                            status_code,
                            func.__name__,
                            str(e),
                        )
                        raise

                except Exception:
                    # Non-network exceptions - don't retry
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed after {max_retries} retries")

        return wrapper

    return decorator


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """Calculate retry delay with exponential backoff and optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Calculated delay in seconds
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = base_delay * (exponential_base**attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter (±25% randomization to prevent thundering herd)
    if jitter:
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(0.1, delay)  # Ensure minimum delay

    return delay
