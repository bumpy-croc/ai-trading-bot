"""
Automatic Strategy Switching System

This module implements safe automatic strategy switching with validation,
cooling-off periods, audit trails, and performance impact analysis.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd

from .performance_monitor import DegradationSeverity, PerformanceMonitor, SwitchDecision
from .performance_tracker import PerformanceTracker
from .regime_context import RegimeContext
from .strategy_selector import StrategyScore, StrategySelector


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
    to_strategy: Optional[str]  # None for auto-selection
    reason: str
    requested_at: datetime
    requested_by: str
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=emergency
    
    # Switch decision context
    switch_decision: Optional[SwitchDecision] = None
    alternative_scores: Optional[list[StrategyScore]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'request_id': self.request_id,
            'trigger': self.trigger.value,
            'from_strategy': self.from_strategy,
            'to_strategy': self.to_strategy,
            'reason': self.reason,
            'requested_at': self.requested_at.isoformat(),
            'requested_by': self.requested_by,
            'priority': self.priority,
            'switch_decision': self.switch_decision.to_dict() if self.switch_decision else None,
            'alternative_scores': [s.to_dict() for s in self.alternative_scores] if self.alternative_scores else None
        }


@dataclass
class SwitchRecord:
    """Record of completed strategy switch"""
    switch_id: str
    request: SwitchRequest
    validation_result: ValidationResult
    status: SwitchStatus
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Performance tracking
    pre_switch_performance: Optional[dict[str, float]] = None
    post_switch_performance: Optional[dict[str, float]] = None
    performance_impact: Optional[dict[str, float]] = None
    
    # Error information
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'switch_id': self.switch_id,
            'request': self.request.to_dict(),
            'validation_result': self.validation_result.value,
            'status': self.status.value,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'pre_switch_performance': self.pre_switch_performance,
            'post_switch_performance': self.post_switch_performance,
            'performance_impact': self.performance_impact,
            'error_message': self.error_message
        }


class StrategySwitcher:
    """
    Automatic strategy switching system with safety controls
    
    This class implements safe automatic strategy switching with validation,
    cooling-off periods, audit trails, and performance impact analysis.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor,
                 strategy_selector: StrategySelector,
                 config: Optional[SwitchConfig] = None):
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
        self.last_switch_time: Optional[datetime] = None
        
        # Manual override controls
        self.manual_override_active = False
        self.manual_override_until: Optional[datetime] = None
        self.manual_override_reason: Optional[str] = None
        
        # Switch callbacks
        self.pre_switch_callbacks: list[Callable[[str, str], bool]] = []
        self.post_switch_callbacks: list[Callable[[str, str, bool], None]] = []
        
        # Performance tracking
        self.switch_performance_tracking: dict[str, dict[str, Any]] = {}
        
        self.logger.info("StrategySwitcher initialized")
    
    def evaluate_switch_need(self, current_strategy_id: str,
                           performance_tracker: PerformanceTracker,
                           available_strategies: dict[str, PerformanceTracker],
                           market_data: pd.DataFrame,
                           current_regime: Optional[RegimeContext] = None) -> Optional[SwitchRequest]:
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
            request_id=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger=SwitchTrigger.PERFORMANCE_DEGRADATION,
            from_strategy=current_strategy_id,
            to_strategy=alternative_scores[0].strategy_id,
            reason=switch_decision.reason,
            requested_at=datetime.now(),
            requested_by="automatic_evaluation",
            priority=self._determine_priority(switch_decision.degradation_severity),
            switch_decision=switch_decision,
            alternative_scores=alternative_scores
        )
        
        self.logger.info(f"Switch evaluation recommends switching from {current_strategy_id} "
                        f"to {request.to_strategy}: {request.reason}")
        
        return request
    
    def request_manual_switch(self, from_strategy: str, to_strategy: str,
                            reason: str, requested_by: str) -> str:
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
            request_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=from_strategy,
            to_strategy=to_strategy,
            reason=reason,
            requested_at=datetime.now(),
            requested_by=requested_by,
            priority=2  # Medium priority for manual requests
        )
        
        self.pending_requests[request.request_id] = request
        
        self.logger.info(f"Manual switch requested: {from_strategy} -> {to_strategy} "
                        f"by {requested_by}: {reason}")
        
        return request.request_id
    
    def execute_switch(self, request: SwitchRequest,
                      strategy_activation_callback: Callable[[str], bool]) -> SwitchRecord:
        """
        Execute a strategy switch request
        
        Args:
            request: Switch request to execute
            strategy_activation_callback: Callback to activate the new strategy
            
        Returns:
            Switch record with execution results
        """
        switch_record = SwitchRecord(
            switch_id=f"switch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            request=request,
            validation_result=ValidationResult.APPROVED,  # Will be updated
            status=SwitchStatus.PENDING
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
            switch_record.executed_at = datetime.now()
            
            # Capture pre-switch performance
            switch_record.pre_switch_performance = self._capture_performance_snapshot(request.from_strategy)
            
            # Execute pre-switch callbacks
            for callback in self.pre_switch_callbacks:
                try:
                    if not callback(request.from_strategy, request.to_strategy):
                        switch_record.status = SwitchStatus.FAILED
                        switch_record.error_message = "Pre-switch callback failed"
                        return switch_record
                except Exception as e:
                    self.logger.error(f"Pre-switch callback error: {e}")
                    switch_record.status = SwitchStatus.FAILED
                    switch_record.error_message = f"Pre-switch callback error: {e}"
                    return switch_record
            
            # Activate new strategy
            success = strategy_activation_callback(request.to_strategy)
            
            if not success:
                switch_record.status = SwitchStatus.FAILED
                switch_record.error_message = "Strategy activation failed"
                return switch_record
            
            # Update switch state
            self.last_switch_time = datetime.now()
            switch_record.status = SwitchStatus.COMPLETED
            switch_record.completed_at = datetime.now()
            
            # Execute post-switch callbacks
            for callback in self.post_switch_callbacks:
                try:
                    callback(request.from_strategy, request.to_strategy, True)
                except Exception as e:
                    self.logger.error(f"Post-switch callback error: {e}")
            
            # Start performance tracking
            if self.config.track_switch_performance:
                self._start_switch_performance_tracking(switch_record)
            
            self.logger.info(f"Strategy switch completed: {request.from_strategy} -> {request.to_strategy}")
            
        except Exception as e:
            switch_record.status = SwitchStatus.FAILED
            switch_record.error_message = str(e)
            self.logger.error(f"Strategy switch failed: {e}")
            
            # Execute post-switch callbacks with failure flag
            for callback in self.post_switch_callbacks:
                try:
                    callback(request.from_strategy, request.to_strategy, False)
                except Exception as callback_error:
                    self.logger.error(f"Post-switch callback error: {callback_error}")
        
        finally:
            # Add to history
            self.switch_history.append(switch_record)
            
            # Remove from pending requests
            if request.request_id in self.pending_requests:
                del self.pending_requests[request.request_id]
        
        return switch_record
    
    def set_manual_override(self, active: bool, duration_hours: Optional[int] = None,
                          reason: Optional[str] = None) -> None:
        """
        Set manual override to prevent/allow automatic switching
        
        Args:
            active: Whether to activate manual override
            duration_hours: How long to keep override active (None = indefinite)
            reason: Reason for override
        """
        self.manual_override_active = active
        
        if active and duration_hours:
            self.manual_override_until = datetime.now() + timedelta(hours=duration_hours)
        else:
            self.manual_override_until = None
        
        self.manual_override_reason = reason
        
        status = "activated" if active else "deactivated"
        duration_str = f" for {duration_hours} hours" if duration_hours else ""
        reason_str = f": {reason}" if reason else ""
        
        self.logger.info(f"Manual override {status}{duration_str}{reason_str}")
    
    def get_switch_history(self, days: int = 30, strategy_id: Optional[str] = None) -> list[SwitchRecord]:
        """
        Get switch history for analysis
        
        Args:
            days: Number of days to look back
            strategy_id: Optional filter by strategy ID
            
        Returns:
            List of switch records
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_history = []
        for record in self.switch_history:
            if record.request.requested_at < cutoff_date:
                continue
            
            if strategy_id and (record.request.from_strategy != strategy_id and 
                              record.request.to_strategy != strategy_id):
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
                'total_switches': 0,
                'successful_switches': 0,
                'failed_switches': 0,
                'success_rate': 0.0,
                'avg_switches_per_day': 0.0,
                'triggers': {},
                'most_switched_from': None,
                'most_switched_to': None
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
        
        most_switched_from = max(set(from_strategies), key=from_strategies.count) if from_strategies else None
        most_switched_to = max(set(to_strategies), key=to_strategies.count) if to_strategies else None
        
        return {
            'total_switches': total_switches,
            'successful_switches': successful_switches,
            'failed_switches': failed_switches,
            'success_rate': success_rate,
            'avg_switches_per_day': total_switches / days,
            'triggers': trigger_counts,
            'most_switched_from': most_switched_from,
            'most_switched_to': most_switched_to,
            'manual_override_active': self.manual_override_active,
            'last_switch_time': self.last_switch_time.isoformat() if self.last_switch_time else None
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
        
        # Check if we have sufficient data for the target strategy
        if request.alternative_scores:
            target_score = next((s for s in request.alternative_scores 
                               if s.strategy_id == request.to_strategy), None)
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
        
        if self.manual_override_until and datetime.now() > self.manual_override_until:
            # Override has expired
            self.manual_override_active = False
            self.manual_override_until = None
            self.manual_override_reason = None
            return False
        
        return True
    
    def _can_switch_now(self, trigger: SwitchTrigger) -> bool:
        """Check if enough time has passed since last switch"""
        if not self.last_switch_time:
            return True
        
        if trigger == SwitchTrigger.EMERGENCY_STOP:
            min_interval = timedelta(hours=self.config.emergency_switch_interval_hours)
        else:
            min_interval = timedelta(hours=self.config.min_switch_interval_hours)
        
        return datetime.now() - self.last_switch_time >= min_interval
    
    def _within_switch_limits(self) -> bool:
        """Check if we're within daily/weekly switch limits"""
        now = datetime.now()
        
        # Check daily limit
        daily_cutoff = now - timedelta(days=1)
        daily_switches = sum(1 for r in self.switch_history 
                           if r.request.requested_at >= daily_cutoff and 
                           r.status == SwitchStatus.COMPLETED)
        
        if daily_switches >= self.config.max_switches_per_day:
            return False
        
        # Check weekly limit
        weekly_cutoff = now - timedelta(days=7)
        weekly_switches = sum(1 for r in self.switch_history 
                            if r.request.requested_at >= weekly_cutoff and 
                            r.status == SwitchStatus.COMPLETED)
        
        if weekly_switches >= self.config.max_switches_per_week:
            return False
        
        return True
    
    def _meets_improvement_threshold(self, request: SwitchRequest) -> bool:
        """Check if the proposed switch meets minimum improvement threshold"""
        if not request.alternative_scores or not request.switch_decision:
            return True  # Can't validate, allow switch
        
        target_score = next((s for s in request.alternative_scores 
                           if s.strategy_id == request.to_strategy), None)
        
        if not target_score:
            return False
        
        # For now, use a simple threshold check
        # In a full implementation, you'd compare against current strategy performance
        return target_score.total_score >= self.config.min_improvement_threshold
    
    def _exceeds_risk_threshold(self, request: SwitchRequest) -> bool:
        """Check if the proposed switch exceeds risk thresholds"""
        if not request.alternative_scores:
            return False
        
        target_score = next((s for s in request.alternative_scores 
                           if s.strategy_id == request.to_strategy), None)
        
        if not target_score:
            return False
        
        # Check if target strategy has high drawdown risk
        # This is a simplified check - in practice you'd compare risk metrics
        return False  # Placeholder implementation
    
    def _determine_priority(self, severity: DegradationSeverity) -> int:
        """Determine switch priority based on degradation severity"""
        if severity == DegradationSeverity.CRITICAL:
            return 4  # Emergency
        elif severity == DegradationSeverity.SEVERE:
            return 3  # High
        elif severity == DegradationSeverity.MODERATE:
            return 2  # Medium
        else:
            return 1  # Low
    
    def _capture_performance_snapshot(self, strategy_id: str) -> dict[str, float]:
        """Capture performance snapshot before switch"""
        # Placeholder implementation
        # In practice, you'd capture key metrics from the performance tracker
        return {
            'timestamp': datetime.now().timestamp(),
            'strategy_id': strategy_id
        }
    
    def _start_switch_performance_tracking(self, switch_record: SwitchRecord) -> None:
        """Start tracking performance after a switch"""
        if not switch_record.request.to_strategy:
            return
        
        tracking_info = {
            'switch_id': switch_record.switch_id,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(days=self.config.switch_performance_window_days),
            'from_strategy': switch_record.request.from_strategy,
            'to_strategy': switch_record.request.to_strategy,
            'pre_switch_performance': switch_record.pre_switch_performance
        }
        
        self.switch_performance_tracking[switch_record.switch_id] = tracking_info
        
        self.logger.info(f"Started performance tracking for switch {switch_record.switch_id}")
    
    def update_switch_performance_tracking(self, performance_trackers: dict[str, PerformanceTracker]) -> None:
        """Update performance tracking for recent switches"""
        now = datetime.now()
        completed_tracking = []
        
        for switch_id, tracking_info in self.switch_performance_tracking.items():
            if now >= tracking_info['end_time']:
                # Tracking period completed
                to_strategy = tracking_info['to_strategy']
                
                if to_strategy in performance_trackers:
                    # Capture post-switch performance
                    post_switch_performance = self._capture_performance_snapshot(to_strategy)
                    
                    # Calculate performance impact
                    performance_impact = self._calculate_performance_impact(
                        tracking_info['pre_switch_performance'],
                        post_switch_performance
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
    
    def _calculate_performance_impact(self, pre_performance: dict[str, float],
                                    post_performance: dict[str, float]) -> dict[str, float]:
        """Calculate performance impact of a switch"""
        # Placeholder implementation
        # In practice, you'd calculate meaningful performance differences
        return {
            'performance_change': 0.0,  # Placeholder
            'calculated_at': datetime.now().timestamp()
        }