"""HealthMonitor tracks engine health and provides adaptive behavior.

Monitors consecutive errors, calculates adaptive intervals, and determines
when the engine should shut down due to persistent failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.config.constants import DEFAULT_RECENT_TRADE_LOOKBACK_HOURS

if TYPE_CHECKING:
    from src.engines.live.execution.position_tracker import LivePosition

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_CONSECUTIVE_ERRORS = 10
DEFAULT_BASE_CHECK_INTERVAL = 60  # seconds
DEFAULT_MIN_CHECK_INTERVAL = 10  # seconds
DEFAULT_MAX_CHECK_INTERVAL = 300  # seconds
DEFAULT_ERROR_COOLDOWN = 30  # seconds


@dataclass
class HealthStatus:
    """Current health status of the trading engine."""

    is_healthy: bool
    consecutive_errors: int
    max_consecutive_errors: int
    should_shutdown: bool
    current_interval: int
    last_success_time: datetime | None
    error_messages: list[str]


class HealthMonitor:
    """Tracks engine health and provides adaptive behavior.

    This class encapsulates health monitoring including:
    - Consecutive error tracking
    - Adaptive check interval calculation
    - Shutdown determination
    - Health status reporting
    """

    def __init__(
        self,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        base_check_interval: int = DEFAULT_BASE_CHECK_INTERVAL,
        min_check_interval: int = DEFAULT_MIN_CHECK_INTERVAL,
        max_check_interval: int = DEFAULT_MAX_CHECK_INTERVAL,
        error_cooldown: int = DEFAULT_ERROR_COOLDOWN,
    ) -> None:
        """Initialize health monitor.

        Args:
            max_consecutive_errors: Maximum errors before shutdown.
            base_check_interval: Base interval between checks in seconds.
            min_check_interval: Minimum interval in seconds.
            max_check_interval: Maximum interval in seconds.
            error_cooldown: Cooldown time after errors in seconds.
        """
        self.max_consecutive_errors = max_consecutive_errors
        self.base_check_interval = base_check_interval
        self.min_check_interval = min_check_interval
        self.max_check_interval = max_check_interval
        self.error_cooldown = error_cooldown

        # State tracking
        self.consecutive_errors = 0
        self.current_interval = base_check_interval
        self.last_success_time: datetime | None = None
        self.last_error_time: datetime | None = None
        self.recent_errors: list[str] = []
        self._max_recent_errors = 10

    def record_success(self) -> None:
        """Record a successful operation, resetting error count."""
        self.consecutive_errors = 0
        self.last_success_time = datetime.now(UTC)

    def record_error(self, error: Exception | str) -> None:
        """Record an error occurrence.

        Args:
            error: The error that occurred.
        """
        self.consecutive_errors += 1
        self.last_error_time = datetime.now(UTC)

        error_msg = str(error) if isinstance(error, Exception) else error
        self.recent_errors.append(
            f"{datetime.now(UTC).isoformat()}: {error_msg}"
        )
        if len(self.recent_errors) > self._max_recent_errors:
            self.recent_errors = self.recent_errors[-self._max_recent_errors:]

        logger.error(
            "Error in trading loop (#%d): %s",
            self.consecutive_errors,
            error_msg,
        )

    def should_shutdown(self) -> bool:
        """Check if the engine should shut down due to errors.

        Returns:
            True if errors exceed threshold.
        """
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.error(
                "Too many consecutive errors (%d). Recommending shutdown.",
                self.consecutive_errors,
            )
            return True
        return False

    def calculate_adaptive_interval(
        self,
        positions: dict[str, Any] | None = None,
        current_price: float | None = None,
    ) -> int:
        """Calculate adaptive check interval based on activity and errors.

        Args:
            positions: Dictionary of active positions.
            current_price: Current market price (unused, for API compatibility).

        Returns:
            Recommended check interval in seconds.
        """
        interval = self.base_check_interval
        positions = positions or {}

        # Check for recent trading activity
        recent_trades = 0
        if positions:
            now = datetime.now(UTC)
            for pos in positions.values():
                entry_time = getattr(pos, "entry_time", None)
                if entry_time and entry_time > now - timedelta(
                    hours=DEFAULT_RECENT_TRADE_LOOKBACK_HOURS
                ):
                    recent_trades += 1

        if recent_trades > 0:
            # More frequent checks if we have recent activity
            interval = max(self.min_check_interval, interval // 2)
        elif len(positions) == 0:
            # Less frequent checks if no active positions
            interval = min(self.max_check_interval, interval * 2)

        # Consider time of day (basic market hours awareness, UTC)
        current_hour = datetime.now(UTC).hour
        if current_hour < 6 or current_hour > 22:
            interval = min(self.max_check_interval, int(interval * 1.5))

        # Increase interval on errors with cooldown
        if self.consecutive_errors > 0:
            error_multiplier = min(self.consecutive_errors, 5)
            interval = min(
                self.max_check_interval,
                interval * error_multiplier,
            )

        self.current_interval = int(interval)
        return self.current_interval

    def get_error_sleep_time(self) -> float:
        """Get sleep time after an error.

        Returns:
            Sleep time in seconds, accounting for consecutive errors.
        """
        return min(
            self.error_cooldown,
            self.current_interval * self.consecutive_errors,
        )

    def get_health_status(
        self,
        positions: dict[str, LivePosition] | None = None,
    ) -> HealthStatus:
        """Get current health status.

        Args:
            positions: Dictionary of active positions.

        Returns:
            HealthStatus with current health metrics.
        """
        return HealthStatus(
            is_healthy=self.consecutive_errors == 0,
            consecutive_errors=self.consecutive_errors,
            max_consecutive_errors=self.max_consecutive_errors,
            should_shutdown=self.should_shutdown(),
            current_interval=self.current_interval,
            last_success_time=self.last_success_time,
            error_messages=self.recent_errors.copy(),
        )

    def reset(self) -> None:
        """Reset all health monitoring state."""
        self.consecutive_errors = 0
        self.current_interval = self.base_check_interval
        self.last_success_time = None
        self.last_error_time = None
        self.recent_errors.clear()

    def get_stats(self) -> dict:
        """Get health monitoring statistics.

        Returns:
            Dictionary with health stats.
        """
        return {
            "consecutive_errors": self.consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "current_interval": self.current_interval,
            "base_check_interval": self.base_check_interval,
            "min_check_interval": self.min_check_interval,
            "max_check_interval": self.max_check_interval,
            "last_success_time": (
                self.last_success_time.isoformat()
                if self.last_success_time
                else None
            ),
            "last_error_time": (
                self.last_error_time.isoformat()
                if self.last_error_time
                else None
            ),
            "is_healthy": self.consecutive_errors == 0,
        }
