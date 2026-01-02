"""
Automatic Strategy Switching System

This module implements safe automatic strategy switching with validation,
cooling-off periods, audit trails, and performance impact analysis.
"""

import logging
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd

from .performance_monitor import DegradationSeverity, PerformanceMonitor, SwitchDecision
from .performance_tracker import PerformanceTracker
from .regime_context import RegimeContext
from .strategy_selector import StrategyScore, StrategySelector


class ExecutionTimeoutError(Exception):
    """Exception raised when a callback execution times out.

    Named to avoid shadowing the Python builtin TimeoutError.
    """

    pass


def execute_with_timeout(func: Callable, timeout_seconds: int, *args, **kwargs):
    """
    Execute a function with timeout using thread-safe mechanism

    This implementation uses concurrent.futures.ThreadPoolExecutor with timeout
    to ensure compatibility with worker threads and Windows systems.

    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time in seconds
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function execution

    Raises:
        ExecutionTimeoutError: If execution exceeds the specified timeout
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="strategy_switcher_timeout")
    future = executor.submit(func, *args, **kwargs)

    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        # Attempt to cancel the running future and shut down the executor without waiting
        future.cancel()
        raise ExecutionTimeoutError(f"Execution timed out after {timeout_seconds} seconds") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for executing code with a timeout using thread-safe mechanism

    This implementation uses threading.Timer with proper thread safety
    to ensure compatibility with worker threads and Windows systems.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        ExecutionTimeoutError: If execution exceeds the specified timeout
    """
    timeout_occurred = threading.Event()
    original_exception = None

    def timeout_handler():
        """Handler that sets the timeout event"""
        timeout_occurred.set()

    # Start the timeout timer
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
    except Exception as e:
        # Store the original exception
        original_exception = e
        raise
    finally:
        # Cancel the timer
        timer.cancel()

        # Check if timeout occurred
        if timeout_occurred.is_set():
            # If we had an original exception, preserve it
            if original_exception:
                raise original_exception
            else:
                raise ExecutionTimeoutError(f"Execution timed out after {seconds} seconds")


class SwitchTrigger(Enum):
    """Triggers for strategy switching"""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL_REQUEST = "manual_request"
    SCHEDULED_EVALUATION = "scheduled_evaluation"
    EMERGENCY_STOP = "emergency_stop"
    REGIME_CHANGE = "regime_change"


class SwitchStatus(Enum):
    """Status of strategy switch operations"""

    PENDING = "pending"
    VALIDATING = "validating"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ValidationResult(Enum):
    """Results of strategy switch validation"""

    APPROVED = "approved"
    REJECTED_COOLING_OFF = "rejected_cooling_off"
    REJECTED_INSUFFICIENT_DATA = "rejected_insufficient_data"
    REJECTED_NO_BETTER_ALTERNATIVE = "rejected_no_better_alternative"
    REJECTED_HIGH_RISK = "rejected_high_risk"
    REJECTED_MANUAL_OVERRIDE = "rejected_manual_override"
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"


@dataclass
class SwitchConfig:
    """Configuration for automatic strategy switching"""

    # Cooling-off periods
    min_switch_interval_hours: int = 24  # Minimum time between switches
    emergency_switch_interval_hours: int = 1  # For emergency switches

    # Validation thresholds
    min_improvement_threshold: float = 0.05  # 5% minimum improvement required
    max_risk_increase_threshold: float = 0.1  # 10% max risk increase allowed

    # Performance requirements for switching
    min_confidence_for_switch: float = 0.7  # Minimum confidence in switch decision
    min_alternative_performance_days: int = 30  # Min days of alternative performance data

    # Safety controls
    max_switches_per_day: int = 3
    max_switches_per_week: int = 10
    enable_emergency_stops: bool = True

    # Validation gates
    require_manual_approval_for_high_risk: bool = True
    high_risk_drawdown_threshold: float = 0.15  # 15% drawdown considered high risk

    # Performance impact tracking
    track_switch_performance: bool = True
    switch_performance_window_days: int = 7  # Days to track post-switch performance


@dataclass
class SwitchRequest:
    """Request for strategy switching"""

    request_id: str
    trigger: SwitchTrigger
    from_strategy: str
    to_strategy: str | None  # None for auto-selection
    reason: str
    requested_at: datetime
    requested_by: str
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=emergency

    # Switch decision context
    switch_decision: SwitchDecision | None = None
    alternative_scores: list[StrategyScore] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "request_id": self.request_id,
            "trigger": self.trigger.value,
            "from_strategy": self.from_strategy,
            "to_strategy": self.to_strategy,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "requested_by": self.requested_by,
            "priority": self.priority,
            "switch_decision": self.switch_decision.to_dict() if self.switch_decision else None,
            "alternative_scores": (
                [s.to_dict() for s in self.alternative_scores] if self.alternative_scores else None
            ),
        }


@dataclass
class SwitchRecord:
    """Record of completed strategy switch"""

    switch_id: str
    request: SwitchRequest
    validation_result: ValidationResult
    status: SwitchStatus
    executed_at: datetime | None = None
    completed_at: datetime | None = None

    # Performance tracking
    pre_switch_performance: dict[str, float] | None = None
    post_switch_performance: dict[str, float] | None = None
    performance_impact: dict[str, float] | None = None

    # Error information
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "switch_id": self.switch_id,
            "request": self.request.to_dict(),
            "validation_result": self.validation_result.value,
            "status": self.status.value,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "pre_switch_performance": self.pre_switch_performance,
            "post_switch_performance": self.post_switch_performance,
            "performance_impact": self.performance_impact,
            "error_message": self.error_message,
        }


class StrategySwitcher:
    """
    Automatic strategy switching system with safety controls

    This class implements safe automatic strategy switching with validation,
    cooling-off periods, audit trails, and performance impact analysis.
    """

    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        strategy_selector: StrategySelector,
        config: SwitchConfig | None = None,
    ):
        """
        Initialize strategy switcher

        Args:
            performance_monitor: Performance monitor for degradation detection
            strategy_selector: Strategy selector for choosing alternatives
            config: Configuration for switching behavior
        """
        self.performance_monitor = performance_monitor
        self.strategy_selector = strategy_selector
        self.config = config or SwitchConfig()
        self.logger = logging.getLogger("StrategySwitcher")

        # Switch history and state
        self.switch_history: deque[SwitchRecord] = deque(maxlen=1000)
        self.pending_requests: dict[str, SwitchRequest] = {}
        self.last_switch_time: datetime | None = None

        # Manual override controls
        self.manual_override_active = False
        self.manual_override_until: datetime | None = None
        self.manual_override_reason: str | None = None

        # Circuit breaker for critical failures
        self.circuit_breaker_active = False
        self.circuit_breaker_activated_at: datetime | None = None
        self.circuit_breaker_reason: str | None = None
        self.last_active_strategy: str | None = None

        # Switch callbacks
        self.pre_switch_callbacks: list[Callable[[str, str], bool]] = []
        self.post_switch_callbacks: list[Callable[[str, str, bool], None]] = []

        # Performance tracking
        self.switch_performance_tracking: dict[str, dict[str, Any]] = {}

        self.logger.info("StrategySwitcher initialized")

    def evaluate_switch_need(
        self,
        current_strategy_id: str,
        performance_tracker: PerformanceTracker,
        available_strategies: dict[str, PerformanceTracker],
        market_data: pd.DataFrame,
        current_regime: RegimeContext | None = None,
    ) -> SwitchRequest | None:
        """
        Evaluate if a strategy switch is needed

        Args:
            current_strategy_id: Currently active strategy
            performance_tracker: Performance tracker for current strategy
            available_strategies: All available strategies
            market_data: Recent market data
            current_regime: Current market regime

        Returns:
            SwitchRequest if switch is recommended, None otherwise
        """
        # Check if circuit breaker is active
        if self.circuit_breaker_active:
            self.logger.error(
                "Circuit breaker is active - automatic switching disabled. Manual intervention required."
            )
            return None

        # Check if manual override is active
        if self._is_manual_override_active():
            self.logger.debug("Manual override active, skipping automatic evaluation")
            return None

        # Check cooling-off period
        if not self._can_switch_now(SwitchTrigger.PERFORMANCE_DEGRADATION):
            self.logger.debug("Still in cooling-off period, skipping evaluation")
            return None

        # Evaluate performance degradation
        switch_decision = self.performance_monitor.should_switch_strategy(
            current_strategy_id, performance_tracker, market_data, current_regime
        )

        if not switch_decision.should_switch:
            self.logger.debug(f"No switch needed: {switch_decision.reason}")
            return None

        # Get alternative strategy rankings
        exclude_current = [current_strategy_id]
        alternative_scores = self.strategy_selector.rank_strategies(
            available_strategies, current_regime, exclude_current
        )

        if not alternative_scores:
            self.logger.warning("No alternative strategies available")
            return None

        # Create switch request
        request = SwitchRequest(
            request_id=f"auto_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            trigger=SwitchTrigger.PERFORMANCE_DEGRADATION,
            from_strategy=current_strategy_id,
            to_strategy=alternative_scores[0].strategy_id,
            reason=switch_decision.reason,
            requested_at=datetime.now(UTC),
            requested_by="automatic_evaluation",
            priority=self._determine_priority(switch_decision.degradation_severity),
            switch_decision=switch_decision,
            alternative_scores=alternative_scores,
        )

        self.logger.info(
            f"Switch evaluation recommends switching from {current_strategy_id} "
            f"to {request.to_strategy}: {request.reason}"
        )

        return request

    def request_manual_switch(
        self, from_strategy: str, to_strategy: str, reason: str, requested_by: str
    ) -> str:
        """
        Request manual strategy switch

        Args:
            from_strategy: Current strategy ID
            to_strategy: Target strategy ID
            reason: Reason for switch
            requested_by: Who requested the switch

        Returns:
            Request ID for tracking
        """
        request = SwitchRequest(
            request_id=f"manual_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=from_strategy,
            to_strategy=to_strategy,
            reason=reason,
            requested_at=datetime.now(UTC),
            requested_by=requested_by,
            priority=2,  # Medium priority for manual requests
        )

        self.pending_requests[request.request_id] = request

        self.logger.info(
            f"Manual switch requested: {from_strategy} -> {to_strategy} "
            f"by {requested_by}: {reason}"
        )

        return request.request_id

    def execute_switch(
        self,
        request: SwitchRequest,
        strategy_activation_callback: Callable[[str], bool],
        performance_trackers: dict[str, PerformanceTracker] | None = None,
    ) -> SwitchRecord:
        """
        Execute a strategy switch request

        Args:
            request: Switch request to execute
            strategy_activation_callback: Callback to activate the new strategy
            performance_trackers: Optional dict of performance trackers for detailed snapshot capture

        Returns:
            Switch record with execution results
        """
        switch_record = SwitchRecord(
            switch_id=f"switch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}",
            request=request,
            validation_result=ValidationResult.APPROVED,  # Will be updated
            status=SwitchStatus.PENDING,
        )

        try:
            # Validate switch request
            validation_result = self._validate_switch_request(request)
            switch_record.validation_result = validation_result

            if validation_result != ValidationResult.APPROVED:
                switch_record.status = SwitchStatus.REJECTED
                self.logger.warning(f"Switch request rejected: {validation_result.value}")
                return switch_record

            switch_record.status = SwitchStatus.EXECUTING
            switch_record.executed_at = datetime.now(UTC)

            # Capture pre-switch performance with detailed metrics if tracker available
            from_tracker = (
                performance_trackers.get(request.from_strategy) if performance_trackers else None
            )
            switch_record.pre_switch_performance = self._capture_performance_snapshot(
                request.from_strategy, from_tracker
            )

            # Execute pre-switch callbacks with timeout protection
            for i, callback in enumerate(self.pre_switch_callbacks):
                try:
                    result = execute_with_timeout(
                        callback,
                        30,  # 30 second timeout for callbacks
                        request.from_strategy,
                        request.to_strategy,
                    )
                    if not result:
                        switch_record.status = SwitchStatus.FAILED
                        switch_record.error_message = f"Pre-switch callback #{i} returned False"
                        return switch_record
                except ExecutionTimeoutError as e:
                    self.logger.warning("Pre-switch callback #%d timed out: %s", i, e)
                    switch_record.status = SwitchStatus.FAILED
                    switch_record.error_message = (
                        f"Pre-switch callback #{i} timed out after 30 seconds"
                    )
                    return switch_record
                except (ValueError, TypeError, RuntimeError) as e:
                    self.logger.exception("Pre-switch callback #%d error", i)
                    switch_record.status = SwitchStatus.FAILED
                    switch_record.error_message = f"Pre-switch callback #{i} error: {e}"
                    return switch_record

            # Activate new strategy
            try:
                success = strategy_activation_callback(request.to_strategy)
            except (ValueError, TypeError, RuntimeError) as activation_error:
                # Handle activation callback exceptions the same way as False returns
                self.logger.exception("Strategy activation callback raised exception")
                success = False
                # Store the exception for potential rollback error message
                activation_exception = activation_error

            if not success:
                # Rollback: try to reactivate the old strategy
                self.logger.warning(
                    f"Strategy activation failed for {request.to_strategy}, attempting rollback"
                )
                try:
                    rollback_success = strategy_activation_callback(request.from_strategy)

                    if rollback_success:
                        switch_record.status = SwitchStatus.FAILED
                        # Include activation exception info if available
                        if "activation_exception" in locals():
                            switch_record.error_message = (
                                f"Strategy activation failed with exception: {activation_exception}, "
                                "successfully rolled back to previous strategy"
                            )
                        else:
                            switch_record.error_message = "Strategy activation failed, successfully rolled back to previous strategy"
                        self.logger.info(f"Successfully rolled back to {request.from_strategy}")
                        self.last_active_strategy = request.from_strategy
                    else:
                        switch_record.status = SwitchStatus.FAILED
                        # Include activation exception info if available
                        if "activation_exception" in locals():
                            switch_record.error_message = (
                                f"Strategy activation failed with exception: {activation_exception}, "
                                "rollback callback returned False. Manual verification recommended."
                            )
                        else:
                            switch_record.error_message = "Strategy activation failed, rollback callback returned False. Manual verification recommended."
                        self.logger.warning(
                            "Rollback callback returned False; previous strategy may already be active or activation callback is non-idempotent"
                        )
                        self.last_active_strategy = request.from_strategy
                except (ValueError, TypeError, RuntimeError) as rollback_error:
                    # CRITICAL: Exception during rollback - activate circuit breaker
                    switch_record.status = SwitchStatus.FAILED
                    # Include both activation and rollback exception info if available
                    if "activation_exception" in locals():
                        switch_record.error_message = (
                            f"CRITICAL: Strategy activation failed with exception: {activation_exception}, "
                            f"rollback error: {rollback_error} - circuit breaker activated"
                        )
                        self.logger.critical(
                            "CIRCUIT BREAKER ACTIVATED: Activation exception: %s, Rollback error: %s",
                            activation_exception,
                            rollback_error,
                        )
                    else:
                        switch_record.error_message = f"CRITICAL: Strategy activation failed, rollback error: {rollback_error} - circuit breaker activated"
                        self.logger.critical(
                            "CIRCUIT BREAKER ACTIVATED: Rollback error: %s", rollback_error
                        )

                    self._activate_circuit_breaker(
                        f"Rollback exception: {rollback_error}", request.from_strategy
                    )

                return switch_record

            # Update switch state only after successful activation
            self.last_switch_time = datetime.now(UTC)
            switch_record.status = SwitchStatus.COMPLETED
            switch_record.completed_at = datetime.now(UTC)

            # Execute post-switch callbacks with timeout protection (non-blocking)
            for i, callback in enumerate(self.post_switch_callbacks):
                try:
                    execute_with_timeout(
                        callback,
                        30,  # 30 second timeout for callbacks
                        request.from_strategy,
                        request.to_strategy,
                        True,
                    )
                except ExecutionTimeoutError as e:
                    self.logger.warning("Post-switch callback #%d timed out: %s", i, e)
                except (ValueError, TypeError, RuntimeError):
                    self.logger.exception("Post-switch callback #%d error", i)

            # Start performance tracking
            if self.config.track_switch_performance:
                self._start_switch_performance_tracking(switch_record)

            self.logger.info(
                f"Strategy switch completed: {request.from_strategy} -> {request.to_strategy}"
            )

        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            switch_record.status = SwitchStatus.FAILED
            switch_record.error_message = str(e)
            self.logger.exception("Strategy switch failed")

            # Execute post-switch callbacks with failure flag and timeout protection
            for i, callback in enumerate(self.post_switch_callbacks):
                try:
                    execute_with_timeout(
                        callback,
                        30,  # 30 second timeout for callbacks
                        request.from_strategy,
                        request.to_strategy,
                        False,
                    )
                except ExecutionTimeoutError as te:
                    self.logger.warning("Post-switch callback #%d timed out: %s", i, te)
                except (ValueError, TypeError, RuntimeError):
                    self.logger.exception("Post-switch callback #%d error", i)

        finally:
            # Add to history
            self.switch_history.append(switch_record)

            # Remove from pending requests
            if request.request_id in self.pending_requests:
                del self.pending_requests[request.request_id]

        return switch_record

    def set_manual_override(
        self, active: bool, duration_hours: int | None = None, reason: str | None = None
    ) -> None:
        """
        Set manual override to prevent/allow automatic switching

        Args:
            active: Whether to activate manual override
            duration_hours: How long to keep override active (None = indefinite)
            reason: Reason for override
        """
        self.manual_override_active = active

        if active and duration_hours:
            self.manual_override_until = datetime.now(UTC) + timedelta(hours=duration_hours)
        else:
            self.manual_override_until = None

        self.manual_override_reason = reason

        status = "activated" if active else "deactivated"
        duration_str = f" for {duration_hours} hours" if duration_hours else ""
        reason_str = f": {reason}" if reason else ""

        self.logger.info(f"Manual override {status}{duration_str}{reason_str}")

    def get_switch_history(
        self, days: int = 30, strategy_id: str | None = None
    ) -> list[SwitchRecord]:
        """
        Get switch history for analysis

        Args:
            days: Number of days to look back
            strategy_id: Optional filter by strategy ID

        Returns:
            List of switch records
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        filtered_history = []
        for record in self.switch_history:
            if record.request.requested_at < cutoff_date:
                continue

            if strategy_id and strategy_id not in (
                record.request.from_strategy,
                record.request.to_strategy,
            ):
                continue

            filtered_history.append(record)

        return filtered_history

    def get_switch_statistics(self, days: int = 30) -> dict[str, Any]:
        """
        Get switch statistics for monitoring

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of switch statistics
        """
        history = self.get_switch_history(days)

        if not history:
            return {
                "total_switches": 0,
                "successful_switches": 0,
                "failed_switches": 0,
                "success_rate": 0.0,
                "avg_switches_per_day": 0.0,
                "triggers": {},
                "most_switched_from": None,
                "most_switched_to": None,
            }

        # Basic counts
        total_switches = len(history)
        successful_switches = sum(1 for r in history if r.status == SwitchStatus.COMPLETED)
        failed_switches = total_switches - successful_switches
        success_rate = successful_switches / total_switches if total_switches > 0 else 0.0

        # Trigger analysis
        trigger_counts = {}
        for record in history:
            trigger = record.request.trigger.value
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        # Strategy analysis
        from_strategies = [r.request.from_strategy for r in history]
        to_strategies = [r.request.to_strategy for r in history if r.request.to_strategy]

        most_switched_from = (
            max(set(from_strategies), key=from_strategies.count) if from_strategies else None
        )
        most_switched_to = (
            max(set(to_strategies), key=to_strategies.count) if to_strategies else None
        )

        return {
            "total_switches": total_switches,
            "successful_switches": successful_switches,
            "failed_switches": failed_switches,
            "success_rate": success_rate,
            "avg_switches_per_day": total_switches / days,
            "triggers": trigger_counts,
            "most_switched_from": most_switched_from,
            "most_switched_to": most_switched_to,
            "manual_override_active": self.manual_override_active,
            "last_switch_time": (
                self.last_switch_time.isoformat() if self.last_switch_time else None
            ),
        }

    def add_pre_switch_callback(self, callback: Callable[[str, str], bool]) -> None:
        """Add callback to execute before strategy switches"""
        self.pre_switch_callbacks.append(callback)

    def add_post_switch_callback(self, callback: Callable[[str, str, bool], None]) -> None:
        """Add callback to execute after strategy switches"""
        self.post_switch_callbacks.append(callback)

    def _validate_switch_request(self, request: SwitchRequest) -> ValidationResult:
        """Validate a switch request against safety controls"""
        # Check manual override
        if self._is_manual_override_active():
            return ValidationResult.REJECTED_MANUAL_OVERRIDE

        # Check cooling-off period
        if not self._can_switch_now(request.trigger):
            return ValidationResult.REJECTED_COOLING_OFF

        # Check daily/weekly limits
        if not self._within_switch_limits():
            return ValidationResult.REJECTED_COOLING_OFF

        # Check confidence threshold
        if (
            request.switch_decision
            and request.switch_decision.confidence < self.config.min_confidence_for_switch
        ):
            return ValidationResult.REJECTED_LOW_CONFIDENCE

        # Check if we have sufficient data for the target strategy
        if request.alternative_scores:
            target_score = next(
                (s for s in request.alternative_scores if s.strategy_id == request.to_strategy),
                None,
            )
            if not target_score:
                return ValidationResult.REJECTED_INSUFFICIENT_DATA

        # Check improvement threshold
        if request.switch_decision and request.alternative_scores:
            if not self._meets_improvement_threshold(request):
                return ValidationResult.REJECTED_NO_BETTER_ALTERNATIVE

        # Check risk increase
        if self._exceeds_risk_threshold(request):
            if self.config.require_manual_approval_for_high_risk:
                return ValidationResult.REJECTED_HIGH_RISK

        return ValidationResult.APPROVED

    def _is_manual_override_active(self) -> bool:
        """Check if manual override is currently active"""
        if not self.manual_override_active:
            return False

        if self.manual_override_until and datetime.now(UTC) > self.manual_override_until:
            # Override has expired
            self.manual_override_active = False
            self.manual_override_until = None
            self.manual_override_reason = None
            return False

        return True

    def _can_switch_now(self, trigger: SwitchTrigger) -> bool:
        """Check if enough time has passed since last switch

        Emergency stops bypass the cooling-off period entirely to allow
        immediate intervention in critical situations.
        Manual requests use the emergency interval for faster response.
        """
        if not self.last_switch_time:
            return True

        # Emergency stops bypass cooling-off period entirely
        if trigger == SwitchTrigger.EMERGENCY_STOP:
            return True

        # Manual requests use emergency interval for faster response
        if trigger == SwitchTrigger.MANUAL_REQUEST:
            min_interval = timedelta(hours=self.config.emergency_switch_interval_hours)
        else:
            min_interval = timedelta(hours=self.config.min_switch_interval_hours)

        return datetime.now(UTC) - self.last_switch_time >= min_interval

    def _within_switch_limits(self) -> bool:
        """Check if we're within daily/weekly switch limits"""
        now = datetime.now(UTC)

        # Check daily limit
        daily_cutoff = now - timedelta(days=1)
        daily_switches = sum(
            1
            for r in self.switch_history
            if r.request.requested_at >= daily_cutoff and r.status == SwitchStatus.COMPLETED
        )

        if daily_switches >= self.config.max_switches_per_day:
            return False

        # Check weekly limit
        weekly_cutoff = now - timedelta(days=7)
        weekly_switches = sum(
            1
            for r in self.switch_history
            if r.request.requested_at >= weekly_cutoff and r.status == SwitchStatus.COMPLETED
        )

        return weekly_switches < self.config.max_switches_per_week

    def _meets_improvement_threshold(self, request: SwitchRequest) -> bool:
        """Check if the proposed switch meets minimum improvement threshold"""
        if not request.alternative_scores or not request.switch_decision:
            return True  # Can't validate, allow switch

        target_score = next(
            (s for s in request.alternative_scores if s.strategy_id == request.to_strategy), None
        )

        if not target_score:
            return False

        # To properly compare, we need to calculate the current strategy's score
        # using the same methodology as the alternatives. However, since the current
        # strategy was excluded from the alternative_scores, we need to estimate it.

        # Use the degradation severity as a proxy for current performance
        # This is a reasonable heuristic since degradation severity is based on
        # actual performance analysis
        if hasattr(request.switch_decision, "degradation_severity"):
            severity = request.switch_decision.degradation_severity

            # Map degradation severity to estimated normalized score
            # These values represent the current strategy's performance level
            if severity.value == "critical":
                estimated_current_score = 0.15  # Very poor performance
            elif severity.value == "severe":
                estimated_current_score = 0.35  # Poor performance
            elif severity.value == "moderate":
                estimated_current_score = 0.55  # Below average
            else:
                estimated_current_score = 0.65  # Slightly below average
        else:
            # If no degradation severity info, assume moderate underperformance
            estimated_current_score = 0.55

        # Calculate the actual improvement
        improvement = target_score.total_score - estimated_current_score

        # Calculate relative improvement as percentage
        if estimated_current_score > 0:
            relative_improvement = improvement / estimated_current_score
        else:
            # If current score is 0 or negative, any positive improvement is significant
            relative_improvement = improvement if improvement > 0 else 0

        # Check if improvement meets the threshold
        meets_threshold = relative_improvement >= self.config.min_improvement_threshold

        self.logger.debug(
            f"Improvement threshold check: "
            f"current_estimated={estimated_current_score:.3f}, "
            f"target_score={target_score.total_score:.3f}, "
            f"absolute_improvement={improvement:.3f}, "
            f"relative_improvement={relative_improvement:.3f}, "
            f"threshold={self.config.min_improvement_threshold:.3f}, "
            f"meets_threshold={meets_threshold}"
        )

        return meets_threshold

    def _exceeds_risk_threshold(self, request: SwitchRequest) -> bool:
        """Check if the proposed switch exceeds risk thresholds"""
        if not request.alternative_scores:
            return False

        target_score = next(
            (s for s in request.alternative_scores if s.strategy_id == request.to_strategy), None
        )

        if not target_score:
            return False

        # Check risk metrics from the criteria scores
        criteria_scores = target_score.criteria_scores

        # High risk if drawdown score is low (meaning high drawdown)
        from .strategy_selector import SelectionCriteria

        drawdown_score = criteria_scores.get(SelectionCriteria.MAX_DRAWDOWN, 1.0)

        # High risk if volatility score is low (meaning high volatility)
        volatility_score = criteria_scores.get(SelectionCriteria.VOLATILITY, 1.0)

        # High risk if risk-adjusted score is low
        risk_adjusted_score = target_score.risk_adjusted_score

        # Consider high risk if any of these conditions are met
        is_high_risk = (
            drawdown_score < 0.3  # Poor drawdown control
            or volatility_score < 0.3  # Very high volatility
            or risk_adjusted_score < 0.4  # Poor risk-adjusted performance
        )

        return is_high_risk

    def _determine_priority(self, severity: DegradationSeverity) -> int:
        """Determine switch priority based on degradation severity"""
        if severity == DegradationSeverity.CRITICAL:
            return 4  # Emergency
        if severity == DegradationSeverity.SEVERE:
            return 3  # High
        if severity == DegradationSeverity.MODERATE:
            return 2  # Medium
        return 1  # Low

    def _capture_performance_snapshot(
        self, strategy_id: str, performance_tracker: PerformanceTracker | None = None
    ) -> dict[str, float]:
        """Capture performance snapshot before/after switch"""
        snapshot = {"timestamp": datetime.now(UTC).timestamp(), "strategy_id": strategy_id}

        # If performance tracker is provided, capture detailed metrics
        if performance_tracker:
            try:
                metrics = performance_tracker.get_performance_metrics()
                snapshot.update(
                    {
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "total_return_pct": metrics.total_return_pct,
                        "max_drawdown": metrics.max_drawdown,
                        "win_rate": metrics.win_rate,
                        "total_trades": metrics.total_trades,
                        "expectancy": metrics.expectancy,  # More robust measure of average expected profit/loss per trade
                        "volatility": metrics.volatility,
                    }
                )
            except (AttributeError, KeyError, TypeError) as e:
                self.logger.warning("Failed to capture detailed metrics: %s", e)

        return snapshot

    def _start_switch_performance_tracking(self, switch_record: SwitchRecord) -> None:
        """Start tracking performance after a switch"""
        if not switch_record.request.to_strategy:
            return

        tracking_info = {
            "switch_id": switch_record.switch_id,
            "start_time": datetime.now(UTC),
            "end_time": datetime.now(UTC)
            + timedelta(days=self.config.switch_performance_window_days),
            "from_strategy": switch_record.request.from_strategy,
            "to_strategy": switch_record.request.to_strategy,
            "pre_switch_performance": switch_record.pre_switch_performance,
        }

        self.switch_performance_tracking[switch_record.switch_id] = tracking_info

        self.logger.info(f"Started performance tracking for switch {switch_record.switch_id}")

    def update_switch_performance_tracking(
        self, performance_trackers: dict[str, PerformanceTracker]
    ) -> None:
        """Update performance tracking for recent switches"""
        now = datetime.now(UTC)
        completed_tracking = []

        for switch_id, tracking_info in self.switch_performance_tracking.items():
            if now >= tracking_info["end_time"]:
                # Tracking period completed
                to_strategy = tracking_info["to_strategy"]

                if to_strategy in performance_trackers:
                    # Capture post-switch performance with detailed metrics
                    post_switch_performance = self._capture_performance_snapshot(
                        to_strategy, performance_trackers[to_strategy]
                    )

                    # Calculate performance impact
                    performance_impact = self._calculate_performance_impact(
                        tracking_info["pre_switch_performance"], post_switch_performance
                    )

                    # Update switch record in history
                    for record in self.switch_history:
                        if record.switch_id == switch_id:
                            record.post_switch_performance = post_switch_performance
                            record.performance_impact = performance_impact
                            break

                    self.logger.info(f"Completed performance tracking for switch {switch_id}")

                completed_tracking.append(switch_id)

        # Remove completed tracking
        for switch_id in completed_tracking:
            del self.switch_performance_tracking[switch_id]

    def _calculate_performance_impact(
        self, pre_performance: dict[str, float], post_performance: dict[str, float]
    ) -> dict[str, float]:
        """Calculate performance impact of a switch"""
        impact = {"calculated_at": datetime.now(UTC).timestamp()}

        # Calculate changes in key metrics
        metrics_to_compare = [
            "sharpe_ratio",
            "total_return_pct",
            "max_drawdown",
            "win_rate",
            "expectancy",
            "volatility",
        ]

        for metric in metrics_to_compare:
            if metric in pre_performance and metric in post_performance:
                pre_value = pre_performance[metric]
                post_value = post_performance[metric]

                # Calculate absolute and relative changes
                absolute_change = post_value - pre_value

                # Calculate relative change with robust zero handling
                if abs(pre_value) > 0.0001:
                    # Meaningful baseline - use standard relative change
                    relative_change = (post_value - pre_value) / abs(pre_value)
                else:
                    # Near-zero baseline - skip relative change calculation to avoid misleading metrics
                    # Only report absolute change when baseline is near zero
                    relative_change = None  # Will be excluded from analysis

                impact[f"{metric}_change"] = absolute_change
                impact[f"{metric}_change_pct"] = relative_change

        # Calculate overall performance score change
        # Positive changes in sharpe, return, win_rate are good
        # Negative changes in drawdown and volatility are good
        positive_changes = []
        if "sharpe_ratio_change" in impact:
            positive_changes.append(impact["sharpe_ratio_change"])
        if "total_return_pct_change" in impact:
            positive_changes.append(impact["total_return_pct_change"])
        if "win_rate_change" in impact:
            positive_changes.append(impact["win_rate_change"])
        if "max_drawdown_change" in impact:
            positive_changes.append(-impact["max_drawdown_change"])  # Negative is good
        if "volatility_change" in impact:
            positive_changes.append(-impact["volatility_change"])  # Negative is good

        if positive_changes:
            impact["overall_performance_change"] = sum(positive_changes) / len(positive_changes)
        else:
            impact["overall_performance_change"] = 0.0

        return impact

    def _activate_circuit_breaker(
        self, reason: str, last_known_strategy: str | None = None
    ) -> None:
        """
        Activate circuit breaker to prevent further automatic switches

        Args:
            reason: Reason for circuit breaker activation
            last_known_strategy: Last strategy that was known to be active
        """
        self.circuit_breaker_active = True
        self.circuit_breaker_activated_at = datetime.now(UTC)
        self.circuit_breaker_reason = reason
        self.last_active_strategy = last_known_strategy

        self.logger.critical(
            f"CIRCUIT BREAKER ACTIVATED: {reason}. "
            f"Last known strategy: {last_known_strategy}. "
            "Automatic switching disabled. Manual intervention required."
        )

    def reset_circuit_breaker(self, reason: str) -> bool:
        """
        Reset circuit breaker after manual intervention

        Args:
            reason: Reason for resetting the circuit breaker

        Returns:
            True if reset was successful
        """
        if not self.circuit_breaker_active:
            self.logger.warning("Attempted to reset circuit breaker, but it's not active")
            return False

        self.circuit_breaker_active = False
        self.logger.warning(
            f"Circuit breaker reset by manual intervention: {reason}. "
            f"Was activated at {self.circuit_breaker_activated_at} due to: {self.circuit_breaker_reason}"
        )

        # Clear circuit breaker state
        self.circuit_breaker_activated_at = None
        self.circuit_breaker_reason = None

        return True

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get circuit breaker status and details"""
        return {
            "active": self.circuit_breaker_active,
            "activated_at": (
                self.circuit_breaker_activated_at.isoformat()
                if self.circuit_breaker_activated_at
                else None
            ),
            "reason": self.circuit_breaker_reason,
            "last_active_strategy": self.last_active_strategy,
        }
