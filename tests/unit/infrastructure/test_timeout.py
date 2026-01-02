"""Unit tests for timeout infrastructure.

Tests cover:
- timeout_context context manager
- with_timeout decorator
- run_with_timeout function (cross-platform)
- Exception handling
- Platform compatibility
"""

import signal
import threading
import time

import pytest

from src.infrastructure.timeout import (
    TimeoutError,
    run_with_timeout,
    timeout_context,
    with_timeout,
)


class TestTimeoutContext:
    """Tests for timeout_context context manager."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_fast_operation_succeeds(self):
        """Test operations completing within timeout succeed."""
        result = None
        with timeout_context(5.0, "fast operation"):
            result = 1 + 1

        assert result == 2

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_slow_operation_times_out(self):
        """Test operations exceeding timeout raise TimeoutError."""
        with pytest.raises(TimeoutError):
            with timeout_context(0.1, "slow operation"):
                time.sleep(1.0)

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_exception_propagates(self):
        """Test exceptions inside context propagate correctly."""
        with pytest.raises(ValueError, match="test error"):
            with timeout_context(5.0, "error operation"):
                raise ValueError("test error")

    @pytest.mark.skipif(
        hasattr(signal, "SIGALRM"),
        reason="Test only for platforms without SIGALRM",
    )
    def test_no_op_on_unsupported_platform(self):
        """Test context manager is no-op on unsupported platforms."""
        # Should not raise, just pass through
        result = None
        with timeout_context(0.001, "unsupported"):
            result = "completed"
            time.sleep(0.1)  # Would timeout if enforced

        assert result == "completed"


class TestWithTimeoutDecorator:
    """Tests for with_timeout decorator."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_decorator_allows_fast_function(self):
        """Test decorated fast function works correctly."""

        @with_timeout(5.0, "fast function")
        def fast_function(x, y):
            return x + y

        result = fast_function(2, 3)
        assert result == 5

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_decorator_times_out_slow_function(self):
        """Test decorated slow function times out."""

        @with_timeout(0.1, "slow function")
        def slow_function():
            time.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError):
            slow_function()

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_decorator_preserves_function_name(self):
        """Test decorator preserves function metadata."""

        @with_timeout(5.0)
        def my_named_function():
            """My docstring."""
            pass

        assert my_named_function.__name__ == "my_named_function"
        assert my_named_function.__doc__ == """My docstring."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_decorator_uses_function_name_as_default(self):
        """Test decorator uses function name if no operation_name given."""

        # This test verifies the default behavior - hard to test directly
        # but we can verify the decorator works without operation_name
        @with_timeout(5.0)
        def custom_function_name():
            return "result"

        result = custom_function_name()
        assert result == "result"


class TestRunWithTimeout:
    """Tests for run_with_timeout function (cross-platform)."""

    def test_fast_function_succeeds(self):
        """Test fast functions complete successfully."""
        result = run_with_timeout(
            lambda: 1 + 1,
            timeout_seconds=5.0,
            operation_name="addition",
        )
        assert result == 2

    def test_function_with_args(self):
        """Test function receives args correctly."""

        def add(x, y):
            return x + y

        result = run_with_timeout(
            add,
            args=(2, 3),
            timeout_seconds=5.0,
        )
        assert result == 5

    def test_function_with_kwargs(self):
        """Test function receives kwargs correctly."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = run_with_timeout(
            greet,
            args=("World",),
            kwargs={"greeting": "Hi"},
            timeout_seconds=5.0,
        )
        assert result == "Hi, World!"

    def test_slow_function_times_out(self):
        """Test slow functions raise TimeoutError."""

        def slow_function():
            time.sleep(2.0)
            return "done"

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            run_with_timeout(
                slow_function,
                timeout_seconds=0.1,
                operation_name="slow operation",
            )

    def test_exception_propagates(self):
        """Test exceptions from function propagate correctly."""

        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_with_timeout(
                failing_function,
                timeout_seconds=5.0,
            )

    def test_operation_name_in_timeout_error(self):
        """Test operation name appears in timeout error message."""

        def slow_function():
            time.sleep(2.0)

        with pytest.raises(TimeoutError, match="my_operation"):
            run_with_timeout(
                slow_function,
                timeout_seconds=0.1,
                operation_name="my_operation",
            )

    def test_uses_function_name_as_default_operation_name(self):
        """Test function name used when no operation_name provided."""

        def my_slow_function():
            time.sleep(2.0)

        with pytest.raises(TimeoutError, match="my_slow_function"):
            run_with_timeout(
                my_slow_function,
                timeout_seconds=0.1,
            )

    def test_timeout_with_blocking_io_simulation(self):
        """Test timeout works with simulated blocking I/O."""

        def blocking_io():
            # Simulate blocking I/O with sleep
            time.sleep(1.0)
            return "data"

        with pytest.raises(TimeoutError):
            run_with_timeout(
                blocking_io,
                timeout_seconds=0.1,
            )

    def test_thread_daemon_flag(self):
        """Test that timeout thread is daemon (won't block exit)."""
        # This is more of a design verification - we can't easily test
        # the daemon behavior directly, but we can verify fast operations work
        result = run_with_timeout(
            lambda: "quick",
            timeout_seconds=5.0,
        )
        assert result == "quick"


class TestRunWithTimeoutEdgeCases:
    """Edge case tests for run_with_timeout."""

    def test_zero_timeout_raises_immediately(self):
        """Test zero timeout raises immediately."""
        # With zero timeout, the join returns immediately
        with pytest.raises(TimeoutError):
            run_with_timeout(
                lambda: time.sleep(0.1),
                timeout_seconds=0.0,
            )

    def test_very_small_timeout(self):
        """Test very small timeout still works."""
        with pytest.raises(TimeoutError):
            run_with_timeout(
                lambda: time.sleep(1.0),
                timeout_seconds=0.001,
            )

    def test_none_kwargs_handled(self):
        """Test None kwargs is handled correctly."""
        result = run_with_timeout(
            lambda: "result",
            args=(),
            kwargs=None,
            timeout_seconds=5.0,
        )
        assert result == "result"

    def test_empty_args_and_kwargs(self):
        """Test empty args and kwargs work."""
        result = run_with_timeout(
            lambda: 42,
            args=(),
            kwargs={},
            timeout_seconds=5.0,
        )
        assert result == 42


class TestTimeoutThreadCleanup:
    """Tests for thread cleanup behavior.

    Note: These tests verify documented behavior limitations.
    The threading-based timeout cannot actually cancel a running thread.
    """

    def test_result_lists_are_local(self):
        """Test result/exception lists don't leak between calls."""
        # Multiple calls should not interfere
        results = []
        for i in range(5):
            result = run_with_timeout(
                lambda x=i: x * 2,
                timeout_seconds=5.0,
            )
            results.append(result)

        assert results == [0, 2, 4, 6, 8]

    def test_concurrent_timeouts(self):
        """Test multiple concurrent timeout calls work correctly."""
        results = []
        errors = []
        lock = threading.Lock()

        def worker(value):
            try:
                result = run_with_timeout(
                    lambda v=value: v * 2,
                    timeout_seconds=5.0,
                )
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sorted(results) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
