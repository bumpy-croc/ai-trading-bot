"""Unit tests for circuit breaker infrastructure.

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold behavior
- Recovery timeout behavior
- Thread safety
- Decorator and call interfaces
- get_stats() method
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    def test_default_initialization(self):
        """Test circuit breaker initializes with defaults."""
        breaker = CircuitBreaker()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker.name == "circuit_breaker"

    def test_custom_initialization(self):
        """Test circuit breaker with custom parameters."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ValueError,
            name="test_breaker",
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.expected_exception == ValueError
        assert breaker.name == "test_breaker"

    def test_invalid_failure_threshold_raises(self):
        """Test that failure_threshold < 1 raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreaker(failure_threshold=-1)

    def test_invalid_recovery_timeout_raises(self):
        """Test that recovery_timeout <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="recovery_timeout must be positive"):
            CircuitBreaker(recovery_timeout=0)

        with pytest.raises(ValueError, match="recovery_timeout must be positive"):
            CircuitBreaker(recovery_timeout=-1.0)


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_starts_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_stays_closed_on_success(self):
        """Test circuit stays CLOSED on successful calls."""
        breaker = CircuitBreaker(failure_threshold=3)

        for _ in range(10):
            result = breaker.call(lambda: "success")
            assert result == "success"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)

        def failing_func():
            raise ValueError("test error")

        # First 3 failures should trip the breaker
        for i in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    def test_open_circuit_raises_circuit_breaker_error(self):
        """Test that OPEN circuit raises CircuitBreakerError immediately."""
        breaker = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)

        # Trip the breaker
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerError without calling function
        mock_func = Mock()
        with pytest.raises(CircuitBreakerError, match="is OPEN"):
            breaker.call(mock_func)

        # Function should not have been called
        mock_func.assert_not_called()

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms for fast testing
            expected_exception=ValueError,
        )

        # Trip the breaker
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next call should transition to HALF_OPEN and execute
        result = breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_success_closes_circuit(self):
        """Test successful call in HALF_OPEN state closes circuit."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            expected_exception=ValueError,
        )

        # Trip the breaker
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.15)

        # Successful call should close circuit
        breaker.call(lambda: "success")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        """Test failure in HALF_OPEN state reopens circuit."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            expected_exception=ValueError,
        )

        # Trip the breaker
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for recovery
        time.sleep(0.15)

        # Fail again in HALF_OPEN
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset functionality."""

    def test_reset_clears_state(self):
        """Test reset() returns circuit to initial state."""
        breaker = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)

        # Trip the breaker
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_reset_allows_calls_again(self):
        """Test that reset allows calls after circuit was OPEN."""
        breaker = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)

        # Trip the breaker
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Reset and call should work
        breaker.reset()
        result = breaker.call(lambda: "success")
        assert result == "success"


class TestCircuitBreakerDecorator:
    """Tests for decorator interface."""

    def test_decorator_wraps_function(self):
        """Test @breaker decorator works correctly."""
        breaker = CircuitBreaker(failure_threshold=3)

        @breaker
        def my_function(x, y):
            return x + y

        result = my_function(2, 3)
        assert result == 5

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves __name__ and other metadata."""
        breaker = CircuitBreaker()

        @breaker
        def my_named_function():
            """My docstring."""
            pass

        assert my_named_function.__name__ == "my_named_function"
        assert my_named_function.__doc__ == """My docstring."""

    def test_decorator_handles_exceptions(self):
        """Test decorator properly tracks failures."""
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=RuntimeError)

        @breaker
        def failing_function():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            failing_function()

        assert breaker.failure_count == 1

        with pytest.raises(RuntimeError):
            failing_function()

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerStats:
    """Tests for get_stats() method."""

    def test_stats_initial_state(self):
        """Test stats returns correct initial values."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="test_breaker",
        )

        stats = breaker.get_stats()

        assert stats["name"] == "test_breaker"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["failure_threshold"] == 5
        assert stats["recovery_timeout"] == 60.0
        assert stats["last_failure_time"] is None
        assert stats["time_since_last_failure"] is None

    def test_stats_after_failures(self):
        """Test stats reflects failures correctly."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)

        # Record some failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = breaker.get_stats()

        assert stats["failure_count"] == 2
        assert stats["state"] == "closed"
        assert stats["last_failure_time"] is not None
        assert stats["time_since_last_failure"] is not None
        assert stats["time_since_last_failure"] >= 0

    def test_stats_when_open(self):
        """Test stats shows OPEN state correctly."""
        breaker = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = breaker.get_stats()
        assert stats["state"] == "open"


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_calls_are_safe(self):
        """Test circuit breaker handles concurrent calls safely."""
        breaker = CircuitBreaker(failure_threshold=100, expected_exception=ValueError)
        call_count = {"success": 0, "failure": 0}
        lock = threading.Lock()

        def worker(should_fail: bool):
            try:
                if should_fail:
                    breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
                else:
                    breaker.call(lambda: "success")
                with lock:
                    call_count["success"] += 1
            except (ValueError, CircuitBreakerError):
                with lock:
                    call_count["failure"] += 1

        # Run many concurrent calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(100):
                futures.append(executor.submit(worker, i % 3 == 0))

            for future in futures:
                future.result()

        # Should have processed all calls
        assert call_count["success"] + call_count["failure"] == 100

    def test_concurrent_state_access(self):
        """Test concurrent access to state is safe."""
        breaker = CircuitBreaker(failure_threshold=50, expected_exception=ValueError)
        states_seen = []
        lock = threading.Lock()

        def read_state():
            state = breaker.state
            count = breaker.failure_count
            with lock:
                states_seen.append((state, count))

        def cause_failure():
            try:
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except (ValueError, CircuitBreakerError):
                pass

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                if i % 2 == 0:
                    futures.append(executor.submit(read_state))
                else:
                    futures.append(executor.submit(cause_failure))

            for future in futures:
                future.result()

        # All reads should have returned valid states
        for state, count in states_seen:
            assert state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN)
            assert count >= 0


class TestCircuitBreakerExceptionFiltering:
    """Tests for exception type filtering."""

    def test_only_expected_exception_counts(self):
        """Test only expected exception type increments failure count."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)

        # ValueError should count
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.failure_count == 1

        # RuntimeError should NOT count (and should propagate)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("not counted")))

        assert breaker.failure_count == 1  # Still 1

    def test_multiple_expected_exception_types(self):
        """Test multiple exception types can be specified."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            expected_exception=(ValueError, TypeError),
        )

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert breaker.failure_count == 1

        with pytest.raises(TypeError):
            breaker.call(lambda: (_ for _ in ()).throw(TypeError("fail")))
        assert breaker.failure_count == 2

        # KeyError should NOT count
        with pytest.raises(KeyError):
            breaker.call(lambda: (_ for _ in ()).throw(KeyError("not counted")))
        assert breaker.failure_count == 2

    def test_success_resets_failure_count(self):
        """Test successful call resets failure count."""
        breaker = CircuitBreaker(failure_threshold=5, expected_exception=ValueError)

        # Accumulate some failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.failure_count == 3

        # Success should reset
        breaker.call(lambda: "success")

        assert breaker.failure_count == 0
