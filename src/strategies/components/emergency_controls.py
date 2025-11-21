"""
Emergency Controls and Manual Override System

This module implements emergency controls, manual strategy switching capabilities,
conservative mode activation, approval workflows, and real-time monitoring
and alerting for strategy performance.
"""

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .performance_tracker import PerformanceMetrics, PerformanceTracker
from .regime_context import RegimeContext
from .strategy_switcher import (
    StrategySwitcher,
    SwitchRequest,
    SwitchTrigger,
    TimeoutError,
    execute_with_timeout,
)


class EmergencyLevel(Enum):
    """Emergency severity levels"""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class ConservativeMode(Enum):
    """Conservative mode settings"""

    DISABLED = "disabled"
    ENABLED = "enabled"
    EMERGENCY_ONLY = "emergency_only"


class ApprovalStatus(Enum):
    """Approval workflow status"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class AlertType(Enum):
    """Types of performance alerts"""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_DRAWDOWN = "high_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    STRATEGY_ERROR = "strategy_error"
    EMERGENCY_STOP = "emergency_stop"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


@dataclass
class EmergencyConfig:
    """Configuration for emergency controls"""

    # Emergency thresholds
    critical_drawdown_threshold: float = 0.25  # 25% drawdown triggers critical
    high_drawdown_threshold: float = 0.15  # 15% drawdown triggers high alert
    consecutive_loss_threshold: int = 5  # 5 consecutive losses triggers alert

    # Conservative mode settings
    conservative_position_size_multiplier: float = 0.5  # Reduce position sizes by 50%
    conservative_risk_multiplier: float = 0.3  # Reduce risk by 70%
    conservative_confidence_threshold: float = 0.8  # Higher confidence required

    # Approval workflow settings
    approval_timeout_hours: int = 24  # Approval requests expire after 24 hours
    require_approval_for_high_risk: bool = True
    require_approval_for_emergency: bool = False  # Emergency switches bypass approval

    # Alert settings
    alert_cooldown_minutes: int = 30  # Minimum time between similar alerts
    max_alerts_per_hour: int = 10  # Rate limiting for alerts

    # Monitoring settings
    performance_check_interval_minutes: int = 5
    emergency_check_interval_minutes: int = 1


@dataclass
class EmergencyAlert:
    """Emergency alert record"""

    alert_id: str
    alert_type: AlertType
    level: EmergencyLevel
    strategy_id: str
    message: str
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved: bool = False
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "level": self.level.name.lower(),  # Use name for string representation
            "strategy_id": self.strategy_id,
            "message": self.message,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class ApprovalRequest:
    """Approval request for strategy operations"""

    request_id: str
    operation_type: str  # "strategy_switch", "emergency_stop", etc.
    strategy_id: str
    requested_by: str
    requested_at: datetime
    reason: str
    priority: int
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING

    # Approval details
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejection_reason: str | None = None

    # Associated data
    switch_request: SwitchRequest | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "request_id": self.request_id,
            "operation_type": self.operation_type,
            "strategy_id": self.strategy_id,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "reason": self.reason,
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "switch_request": self.switch_request.to_dict() if self.switch_request else None,
        }


class EmergencyControls:
    """
    Emergency controls and manual override system

    This class implements emergency controls, manual strategy switching,
    conservative mode, approval workflows, and real-time monitoring.
    """

    def __init__(
        self, strategy_switcher: StrategySwitcher, config: EmergencyConfig | None = None
    ):
        """
        Initialize emergency controls

        Args:
            strategy_switcher: Strategy switcher for executing switches
            config: Configuration for emergency controls
        """
        self.strategy_switcher = strategy_switcher
        self.config = config or EmergencyConfig()
        self.logger = logging.getLogger("EmergencyControls")

        # Emergency state
        self.emergency_level = EmergencyLevel.NONE
        self.conservative_mode = ConservativeMode.DISABLED
        self.emergency_stop_active = False

        # Alert system
        self.active_alerts: dict[str, EmergencyAlert] = {}
        self.alert_history: deque[EmergencyAlert] = deque(maxlen=1000)
        self.alert_cooldowns: dict[str, datetime] = {}  # alert_type -> last_alert_time

        # Approval system
        self.pending_approvals: dict[str, ApprovalRequest] = {}
        self.approval_history: deque[ApprovalRequest] = deque(maxlen=500)

        # Monitoring state
        self.last_performance_check: datetime | None = None
        self.last_emergency_check: datetime | None = None

        # Callbacks for external integration
        self.alert_callbacks: list[Callable[[EmergencyAlert], None]] = []
        self.approval_callbacks: list[Callable[[ApprovalRequest], None]] = []

        # Performance tracking for emergency detection
        self.performance_snapshots: dict[str, list[dict[str, float]]] = {}
        self.active_strategies: set[str] = set()  # Track active strategies
        self.last_cleanup_time: datetime = datetime.now()

        self.logger.info("EmergencyControls initialized")

    def check_emergency_conditions(
        self,
        strategy_id: str,
        performance_tracker: PerformanceTracker,
        current_regime: RegimeContext | None = None,
    ) -> EmergencyLevel:
        """
        Check for emergency conditions in strategy performance

        Args:
            strategy_id: Strategy to check
            performance_tracker: Performance tracker for the strategy
            current_regime: Current market regime

        Returns:
            Emergency level detected
        """
        current_level = EmergencyLevel.NONE

        # Get current performance metrics
        metrics = performance_tracker.get_performance_metrics()

        # Check drawdown levels
        if metrics.max_drawdown >= self.config.critical_drawdown_threshold:
            current_level = EmergencyLevel.CRITICAL
            self._trigger_alert(
                AlertType.HIGH_DRAWDOWN,
                EmergencyLevel.CRITICAL,
                strategy_id,
                f"Critical drawdown: {metrics.max_drawdown:.1%}",
            )
        elif metrics.max_drawdown >= self.config.high_drawdown_threshold:
            current_level = max(current_level, EmergencyLevel.HIGH)
            self._trigger_alert(
                AlertType.HIGH_DRAWDOWN,
                EmergencyLevel.HIGH,
                strategy_id,
                f"High drawdown: {metrics.max_drawdown:.1%}",
            )

        # Check consecutive losses
        if metrics.consecutive_losses >= self.config.consecutive_loss_threshold:
            current_level = max(current_level, EmergencyLevel.MEDIUM)
            self._trigger_alert(
                AlertType.CONSECUTIVE_LOSSES,
                EmergencyLevel.MEDIUM,
                strategy_id,
                f"Consecutive losses: {metrics.consecutive_losses}",
            )

        # Check for performance degradation
        if self._detect_performance_degradation(strategy_id, metrics):
            current_level = max(current_level, EmergencyLevel.MEDIUM)
            self._trigger_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                EmergencyLevel.MEDIUM,
                strategy_id,
                "Significant performance degradation detected",
            )

        # Update emergency level
        if current_level > self.emergency_level:
            self.emergency_level = current_level
            self.logger.warning(f"Emergency level escalated to {current_level.value}")

            # Auto-activate conservative mode for high/critical levels
            if current_level in [EmergencyLevel.HIGH, EmergencyLevel.CRITICAL]:
                self.activate_conservative_mode(
                    f"Auto-activated due to {current_level.value} emergency"
                )

        return current_level

    def request_manual_strategy_switch(
        self,
        from_strategy: str,
        to_strategy: str,
        reason: str,
        requested_by: str,
        bypass_approval: bool = False,
    ) -> str:
        """
        Request manual strategy switch with optional approval workflow

        Args:
            from_strategy: Current strategy
            to_strategy: Target strategy
            reason: Reason for switch
            requested_by: Who requested the switch
            bypass_approval: Whether to bypass approval workflow

        Returns:
            Request ID for tracking
        """
        # Create switch request
        switch_request = SwitchRequest(
            request_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=from_strategy,
            to_strategy=to_strategy,
            reason=reason,
            requested_at=datetime.now(),
            requested_by=requested_by,
            priority=2,  # Medium priority for manual requests
        )

        # Check if approval is required
        if (
            self.config.require_approval_for_high_risk
            and not bypass_approval
            and not self.emergency_stop_active
        ):

            # Create approval request
            approval_request = ApprovalRequest(
                request_id=f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                operation_type="strategy_switch",
                strategy_id=from_strategy,
                requested_by=requested_by,
                requested_at=datetime.now(),
                reason=reason,
                priority=2,
                expires_at=datetime.now() + timedelta(hours=self.config.approval_timeout_hours),
                switch_request=switch_request,
            )

            self.pending_approvals[approval_request.request_id] = approval_request

            # Notify approval callbacks with timeout protection
            for i, callback in enumerate(self.approval_callbacks):
                try:
                    execute_with_timeout(
                        callback, 10, approval_request
                    )  # 10 second timeout for approval callbacks
                except TimeoutError as error:
                    self.logger.error(f"Approval callback #{i} timed out: {error}")
                except Exception as error:
                    self.logger.error(f"Approval callback #{i} error: {error}")

            self.logger.info(
                f"Manual switch request requires approval: {approval_request.request_id}"
            )
            return approval_request.request_id

        else:
            # Execute switch directly
            request_id = self.strategy_switcher.request_manual_switch(
                from_strategy, to_strategy, reason, requested_by
            )

            self.logger.info(f"Manual switch request submitted directly: {request_id}")
            return request_id

    def approve_request(
        self, request_id: str, approved_by: str, rejection_reason: str | None = None
    ) -> bool:
        """
        Approve or reject a pending approval request

        Args:
            request_id: Request to approve/reject
            approved_by: Who is approving/rejecting
            rejection_reason: Reason for rejection (if rejecting)

        Returns:
            True if request was processed, False if not found or already processed
        """
        if request_id not in self.pending_approvals:
            self.logger.warning(f"Approval request not found: {request_id}")
            return False

        approval_request = self.pending_approvals[request_id]

        # Check if request has expired
        if datetime.now() > approval_request.expires_at:
            approval_request.status = ApprovalStatus.EXPIRED
            self.logger.warning(f"Approval request expired: {request_id}")
            return False

        # Process approval/rejection
        if rejection_reason:
            # Reject the request
            approval_request.status = ApprovalStatus.REJECTED
            approval_request.approved_by = approved_by
            approval_request.approved_at = datetime.now()
            approval_request.rejection_reason = rejection_reason

            self.logger.info(
                f"Request rejected by {approved_by}: {request_id} - {rejection_reason}"
            )

        else:
            # Approve the request
            approval_request.status = ApprovalStatus.APPROVED
            approval_request.approved_by = approved_by
            approval_request.approved_at = datetime.now()

            # Execute the approved operation
            if approval_request.switch_request:
                switch_request_id = self.strategy_switcher.request_manual_switch(
                    approval_request.switch_request.from_strategy,
                    approval_request.switch_request.to_strategy,
                    f"Approved: {approval_request.switch_request.reason}",
                    approved_by,
                )
                self.logger.info(f"Approved switch request executed: {switch_request_id}")

            self.logger.info(f"Request approved by {approved_by}: {request_id}")

        # Move to history
        self.approval_history.append(approval_request)
        del self.pending_approvals[request_id]

        return True

    def activate_emergency_stop(self, reason: str, activated_by: str) -> bool:
        """
        Activate emergency stop to halt all trading

        Args:
            reason: Reason for emergency stop
            activated_by: Who activated the emergency stop

        Returns:
            True if emergency stop was activated
        """
        if self.emergency_stop_active:
            self.logger.warning("Emergency stop already active")
            return False

        self.emergency_stop_active = True
        self.emergency_level = EmergencyLevel.CRITICAL

        # Trigger emergency alert
        self._trigger_alert(
            AlertType.EMERGENCY_STOP,
            EmergencyLevel.CRITICAL,
            "system",
            f"Emergency stop activated by {activated_by}: {reason}",
        )

        # Activate conservative mode
        self.activate_conservative_mode(f"Emergency stop: {reason}")

        self.logger.critical(f"Emergency stop activated by {activated_by}: {reason}")

        return True

    def deactivate_emergency_stop(self, reason: str, deactivated_by: str) -> bool:
        """
        Deactivate emergency stop

        Args:
            reason: Reason for deactivation
            deactivated_by: Who deactivated the emergency stop

        Returns:
            True if emergency stop was deactivated
        """
        if not self.emergency_stop_active:
            self.logger.warning("Emergency stop not active")
            return False

        self.emergency_stop_active = False
        self.emergency_level = EmergencyLevel.NONE

        # Resolve emergency stop alert
        for alert in self.active_alerts.values():
            if alert.alert_type == AlertType.EMERGENCY_STOP:
                alert.resolved = True
                alert.resolved_at = datetime.now()

        self.logger.info(f"Emergency stop deactivated by {deactivated_by}: {reason}")

        return True

    def activate_conservative_mode(self, reason: str) -> None:
        """
        Activate conservative trading mode

        Args:
            reason: Reason for activation
        """
        if self.conservative_mode == ConservativeMode.ENABLED:
            return

        self.conservative_mode = ConservativeMode.ENABLED

        self.logger.info(f"Conservative mode activated: {reason}")

    def deactivate_conservative_mode(self, reason: str) -> None:
        """
        Deactivate conservative trading mode

        Args:
            reason: Reason for deactivation
        """
        if self.conservative_mode == ConservativeMode.DISABLED:
            return

        self.conservative_mode = ConservativeMode.DISABLED

        self.logger.info(f"Conservative mode deactivated: {reason}")

    def get_conservative_adjustments(self) -> dict[str, float]:
        """
        Get conservative mode adjustments for trading parameters

        Returns:
            Dictionary of adjustment multipliers
        """
        if self.conservative_mode != ConservativeMode.ENABLED:
            return {}

        return {
            "position_size_multiplier": self.config.conservative_position_size_multiplier,
            "risk_multiplier": self.config.conservative_risk_multiplier,
            "confidence_threshold": self.config.conservative_confidence_threshold,
        }

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an active alert

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if alert was acknowledged
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()

        self.logger.info(f"Alert acknowledged by {acknowledged_by}: {alert_id}")

        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert

        Args:
            alert_id: Alert to resolve

        Returns:
            True if alert was resolved
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()

        # Remove from active alerts (already in history from when it was triggered)
        del self.active_alerts[alert_id]

        self.logger.info(f"Alert resolved: {alert_id}")

        return True

    def get_system_status(self) -> dict[str, Any]:
        """
        Get comprehensive system status

        Returns:
            Dictionary with system status information
        """
        return {
            "emergency_level": self.emergency_level.name.lower(),
            "conservative_mode": self.conservative_mode.value,
            "emergency_stop_active": self.emergency_stop_active,
            "active_alerts": len(self.active_alerts),
            "pending_approvals": len(self.pending_approvals),
            "manual_override_active": self.strategy_switcher.manual_override_active,
            "last_performance_check": (
                self.last_performance_check.isoformat() if self.last_performance_check else None
            ),
            "last_emergency_check": (
                self.last_emergency_check.isoformat() if self.last_emergency_check else None
            ),
            "conservative_adjustments": self.get_conservative_adjustments(),
        }

    def get_active_alerts(self) -> list[EmergencyAlert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())

    def get_pending_approvals(self) -> list[ApprovalRequest]:
        """Get list of pending approval requests"""
        return list(self.pending_approvals.values())

    def add_alert_callback(self, callback: Callable[[EmergencyAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def add_approval_callback(self, callback: Callable[[ApprovalRequest], None]) -> None:
        """Add callback for approval notifications"""
        self.approval_callbacks.append(callback)

    def _trigger_alert(
        self, alert_type: AlertType, level: EmergencyLevel, strategy_id: str, message: str
    ) -> str | None:
        """Trigger an emergency alert"""
        # Check cooldown
        cooldown_key = f"{alert_type.value}_{strategy_id}"
        if cooldown_key in self.alert_cooldowns:
            last_alert = self.alert_cooldowns[cooldown_key]
            if datetime.now() - last_alert < timedelta(minutes=self.config.alert_cooldown_minutes):
                return None  # Still in cooldown

        # Check rate limiting
        recent_alerts = [
            alert
            for alert in self.alert_history
            if alert.triggered_at > datetime.now() - timedelta(hours=1)
        ]
        if len(recent_alerts) >= self.config.max_alerts_per_hour:
            self.logger.warning("Alert rate limit exceeded, suppressing alert")
            return None

        # Create alert
        alert = EmergencyAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            alert_type=alert_type,
            level=level,
            strategy_id=strategy_id,
            message=message,
            triggered_at=datetime.now(),
        )

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert

        # Add to alert history for rate limiting (immediately, not just on resolution)
        self.alert_history.append(alert)

        # Update cooldown
        self.alert_cooldowns[cooldown_key] = datetime.now()

        # Notify callbacks
        for i, callback in enumerate(self.alert_callbacks):
            try:
                execute_with_timeout(callback, 10, alert)  # 10 second timeout for alert callbacks
            except TimeoutError as error:
                self.logger.error(f"Alert callback #{i} timed out: {error}")
            except Exception as error:
                self.logger.error(f"Alert callback #{i} error: {error}")

        self.logger.warning(f"Alert triggered: {alert.alert_type.value} - {message}")

        return alert.alert_id

    def _detect_performance_degradation(
        self, strategy_id: str, current_metrics: PerformanceMetrics
    ) -> bool:
        """Detect significant performance degradation"""
        # Track active strategy
        self.active_strategies.add(strategy_id)

        # Store performance snapshot
        if strategy_id not in self.performance_snapshots:
            self.performance_snapshots[strategy_id] = []

        snapshot = {
            "timestamp": datetime.now().timestamp(),
            "sharpe_ratio": current_metrics.sharpe_ratio,
            "win_rate": current_metrics.win_rate,
            "max_drawdown": current_metrics.max_drawdown,
            "total_return_pct": current_metrics.total_return_pct,
        }

        self.performance_snapshots[strategy_id].append(snapshot)

        # Keep only recent snapshots (last 24 hours)
        cutoff_time = datetime.now().timestamp() - (24 * 3600)
        self.performance_snapshots[strategy_id] = [
            s for s in self.performance_snapshots[strategy_id] if s["timestamp"] > cutoff_time
        ]

        # Periodic cleanup of inactive strategies (every hour)
        if (datetime.now() - self.last_cleanup_time).total_seconds() > 3600:
            self._cleanup_inactive_strategies()
            self.last_cleanup_time = datetime.now()

        # Need at least 2 snapshots to detect degradation
        if len(self.performance_snapshots[strategy_id]) < 2:
            return False

        # Compare current performance to recent average
        recent_snapshots = self.performance_snapshots[strategy_id][-5:]  # Last 5 snapshots

        if len(recent_snapshots) < 2:
            return False

        # Calculate recent averages
        avg_sharpe = sum(s["sharpe_ratio"] for s in recent_snapshots) / len(recent_snapshots)
        avg_win_rate = sum(s["win_rate"] for s in recent_snapshots) / len(recent_snapshots)

        # Check for significant degradation with proper zero handling
        # Use consistent logic to avoid mixing ratios and absolute differences

        # Sharpe ratio degradation calculation with robust zero handling
        if abs(avg_sharpe) > 0.01:  # More precise threshold
            # Meaningful baseline - use relative degradation
            sharpe_degradation = (avg_sharpe - current_metrics.sharpe_ratio) / abs(avg_sharpe)
        elif avg_sharpe != 0:
            # Small but non-zero baseline - use normalized absolute difference
            sharpe_degradation = min(1.0, abs(avg_sharpe - current_metrics.sharpe_ratio) / 0.01)
        else:
            # Zero baseline - flag as degraded only if current is significantly negative
            sharpe_degradation = 0.0 if abs(current_metrics.sharpe_ratio) < 0.01 else 1.0

        # Win rate degradation calculation with robust zero handling
        if avg_win_rate > 0.01:  # More precise threshold
            # Meaningful baseline - use relative degradation
            win_rate_degradation = (avg_win_rate - current_metrics.win_rate) / avg_win_rate
        elif avg_win_rate != 0:
            # Small but non-zero baseline - use normalized absolute difference
            win_rate_degradation = min(1.0, abs(avg_win_rate - current_metrics.win_rate) / 0.01)
        else:
            # Zero baseline - flag as degraded only if current is significantly negative
            win_rate_degradation = 0.0 if abs(current_metrics.win_rate) < 0.01 else 1.0

        # Normalize degradation metrics to equivalent significance levels
        # Sharpe degradation of 0.3 means 30% worse performance
        # Win rate degradation of 0.2 means 20% worse performance
        # These thresholds are calibrated based on historical analysis:
        # - 30% Sharpe degradation indicates significant underperformance
        # - 20% win rate degradation indicates concerning trend change

        # Alternative: Use normalized thresholds for equivalent significance
        # normalized_sharpe_threshold = 0.3  # 30% degradation
        # normalized_win_rate_threshold = 0.2  # 20% degradation

        # Trigger if either metric has degraded significantly
        return sharpe_degradation > 0.3 or win_rate_degradation > 0.2

    def _cleanup_inactive_strategies(self) -> None:
        """Clean up performance snapshots for inactive strategies"""
        # Find strategies with snapshots but not seen recently
        inactive_strategies = []
        cutoff_time = datetime.now().timestamp() - (48 * 3600)  # 48 hours

        for strategy_id in list(self.performance_snapshots.keys()):
            if strategy_id not in self.active_strategies:
                # Check if strategy has any recent snapshots
                recent_snapshots = [
                    s
                    for s in self.performance_snapshots[strategy_id]
                    if s["timestamp"] > cutoff_time
                ]

                if not recent_snapshots:
                    inactive_strategies.append(strategy_id)

        # Remove inactive strategy snapshots
        for strategy_id in inactive_strategies:
            del self.performance_snapshots[strategy_id]
            self.logger.debug(f"Cleaned up snapshots for inactive strategy: {strategy_id}")

        # Clear active strategies set for next tracking period
        self.active_strategies.clear()

    def cleanup_expired_requests(self) -> None:
        """Clean up expired approval requests"""
        now = datetime.now()
        expired_requests = []

        for request_id, request in self.pending_approvals.items():
            if now > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                expired_requests.append(request_id)

        for request_id in expired_requests:
            request = self.pending_approvals[request_id]
            self.approval_history.append(request)
            del self.pending_approvals[request_id]
            self.logger.info(f"Approval request expired: {request_id}")

    def update_monitoring(
        self,
        strategy_performance: dict[str, PerformanceTracker],
        current_regime: RegimeContext | None = None,
    ) -> None:
        """
        Update monitoring checks for all strategies

        Args:
            strategy_performance: Dictionary of strategy performance trackers
            current_regime: Current market regime
        """
        now = datetime.now()

        # Check if it's time for performance monitoring
        if self.last_performance_check is None or now - self.last_performance_check >= timedelta(
            minutes=self.config.performance_check_interval_minutes
        ):

            for strategy_id, tracker in strategy_performance.items():
                self.check_emergency_conditions(strategy_id, tracker, current_regime)

            self.last_performance_check = now

        # Check if it's time for emergency monitoring
        if self.last_emergency_check is None or now - self.last_emergency_check >= timedelta(
            minutes=self.config.emergency_check_interval_minutes
        ):

            # Clean up expired requests
            self.cleanup_expired_requests()

            self.last_emergency_check = now
