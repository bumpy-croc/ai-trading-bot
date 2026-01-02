"""
Performance Monitoring System Integration

This module demonstrates how to integrate all performance monitoring and
automatic strategy switching components together in a complete system.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .emergency_controls import EmergencyConfig, EmergencyControls
from .performance_monitor import PerformanceDegradationConfig, PerformanceMonitor
from .performance_tracker import PerformanceTracker
from .regime_context import RegimeContext
from .strategy_selector import SelectionConfig, StrategySelector
from .strategy_switcher import StrategySwitcher, SwitchConfig


class PerformanceMonitoringSystem:
    """
    Integrated performance monitoring and automatic strategy switching system

    This class orchestrates all components to provide comprehensive performance
    monitoring, strategy selection, automatic switching, and emergency controls.
    """

    def __init__(
        self,
        performance_config: PerformanceDegradationConfig | None = None,
        selection_config: SelectionConfig | None = None,
        switch_config: SwitchConfig | None = None,
        emergency_config: EmergencyConfig | None = None,
    ):
        """
        Initialize the performance monitoring system

        Args:
            performance_config: Configuration for performance degradation detection
            selection_config: Configuration for strategy selection
            switch_config: Configuration for strategy switching
            emergency_config: Configuration for emergency controls
        """
        self.logger = logging.getLogger("PerformanceMonitoringSystem")

        # Initialize components
        self.performance_monitor = PerformanceMonitor(performance_config)
        self.strategy_selector = StrategySelector(selection_config)
        self.strategy_switcher = StrategySwitcher(
            self.performance_monitor, self.strategy_selector, switch_config
        )
        self.emergency_controls = EmergencyControls(self.strategy_switcher, emergency_config)

        # System state
        self.current_strategy_id: str | None = None
        self.available_strategies: dict[str, PerformanceTracker] = {}
        self.strategy_activation_callback: Callable[[str], bool] | None = None

        # Monitoring state
        self.last_monitoring_update = datetime.now(UTC)
        self.monitoring_enabled = True

        self.logger.info("PerformanceMonitoringSystem initialized")

    def register_strategy(self, strategy_id: str, performance_tracker: PerformanceTracker) -> None:
        """
        Register a strategy with the monitoring system

        Args:
            strategy_id: Unique identifier for the strategy
            performance_tracker: Performance tracker for the strategy
        """
        self.available_strategies[strategy_id] = performance_tracker

        # Update performance baseline
        self.performance_monitor.update_performance_baseline(strategy_id, performance_tracker)

        self.logger.info(f"Registered strategy: {strategy_id}")

    def set_current_strategy(self, strategy_id: str) -> bool:
        """
        Set the currently active strategy

        Args:
            strategy_id: Strategy to set as current

        Returns:
            True if strategy was set successfully
        """
        if strategy_id not in self.available_strategies:
            self.logger.error(f"Strategy not found: {strategy_id}")
            return False

        self.current_strategy_id = strategy_id
        self.logger.info(f"Current strategy set to: {strategy_id}")
        return True

    def set_strategy_activation_callback(self, callback: Callable[[str], bool]) -> None:
        """
        Set callback function for activating strategies

        Args:
            callback: Function that takes strategy_id and returns success boolean
        """
        self.strategy_activation_callback = callback

    def update_monitoring(
        self, market_data: pd.DataFrame, current_regime: RegimeContext | None = None
    ) -> dict[str, Any]:
        """
        Update all monitoring components and check for necessary actions

        Args:
            market_data: Recent market data for analysis
            current_regime: Current market regime context

        Returns:
            Dictionary with monitoring results and actions taken
        """
        if not self.monitoring_enabled:
            return {"monitoring_enabled": False}

        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "current_strategy": self.current_strategy_id,
            "monitoring_enabled": True,
            "actions_taken": [],
            "alerts_triggered": [],
            "switch_recommendations": [],
        }

        try:
            # Update emergency controls monitoring
            self.emergency_controls.update_monitoring(self.available_strategies, current_regime)

            # Check emergency conditions for current strategy
            if self.current_strategy_id:
                current_tracker = self.available_strategies.get(self.current_strategy_id)
                if current_tracker:
                    emergency_level = self.emergency_controls.check_emergency_conditions(
                        self.current_strategy_id, current_tracker, current_regime
                    )
                    results["emergency_level"] = emergency_level.name.lower()

            # Evaluate need for strategy switching
            if (
                self.current_strategy_id
                and not self.emergency_controls.emergency_stop_active
                and self.strategy_activation_callback
            ):

                current_tracker = self.available_strategies.get(self.current_strategy_id)
                if current_tracker:
                    switch_request = self.strategy_switcher.evaluate_switch_need(
                        self.current_strategy_id,
                        current_tracker,
                        self.available_strategies,
                        market_data,
                        current_regime,
                    )

                    if switch_request:
                        results["switch_recommendations"].append(
                            {
                                "request_id": switch_request.request_id,
                                "from_strategy": switch_request.from_strategy,
                                "to_strategy": switch_request.to_strategy,
                                "reason": switch_request.reason,
                                "confidence": (
                                    switch_request.switch_decision.confidence
                                    if switch_request.switch_decision
                                    else 0.0
                                ),
                            }
                        )

                        # Execute automatic switch if conditions are met
                        switch_record = self.strategy_switcher.execute_switch(
                            switch_request, self.strategy_activation_callback
                        )

                        if switch_record.status.value == "completed":
                            self.current_strategy_id = switch_request.to_strategy
                            results["actions_taken"].append(
                                {
                                    "action": "strategy_switch",
                                    "from_strategy": switch_request.from_strategy,
                                    "to_strategy": switch_request.to_strategy,
                                    "reason": switch_request.reason,
                                }
                            )

                            self.logger.info(
                                f"Automatic strategy switch completed: "
                                f"{switch_request.from_strategy} -> {switch_request.to_strategy}"
                            )

            # Update performance baselines
            for strategy_id, tracker in self.available_strategies.items():
                regime_key = None
                if current_regime:
                    regime_key = f"{current_regime.trend.value}_{current_regime.volatility.value}"

                self.performance_monitor.update_performance_baseline(
                    strategy_id, tracker, regime_key
                )

            # Get system status
            results["system_status"] = self.emergency_controls.get_system_status()
            results["active_alerts"] = [
                alert.to_dict() for alert in self.emergency_controls.get_active_alerts()
            ]
            results["pending_approvals"] = [
                req.to_dict() for req in self.emergency_controls.get_pending_approvals()
            ]

            # Get switch statistics
            results["switch_statistics"] = self.strategy_switcher.get_switch_statistics()

            self.last_monitoring_update = datetime.now(UTC)

        except (ValueError, KeyError, AttributeError) as e:
            self.logger.exception("Monitoring update failed: %s", e)
            results["error"] = str(e)

        return results

    def request_manual_switch(
        self, to_strategy: str, reason: str, requested_by: str, bypass_approval: bool = False
    ) -> str:
        """
        Request manual strategy switch

        Args:
            to_strategy: Target strategy ID
            reason: Reason for switch
            requested_by: Who requested the switch
            bypass_approval: Whether to bypass approval workflow

        Returns:
            Request ID for tracking
        """
        if not self.current_strategy_id:
            raise ValueError("No current strategy set")

        return self.emergency_controls.request_manual_strategy_switch(
            self.current_strategy_id, to_strategy, reason, requested_by, bypass_approval
        )

    def approve_request(
        self, request_id: str, approved_by: str, rejection_reason: str | None = None
    ) -> bool:
        """
        Approve or reject a pending request

        Args:
            request_id: Request to approve/reject
            approved_by: Who is approving/rejecting
            rejection_reason: Reason for rejection (if rejecting)

        Returns:
            True if request was processed successfully
        """
        return self.emergency_controls.approve_request(request_id, approved_by, rejection_reason)

    def activate_emergency_stop(self, reason: str, activated_by: str) -> bool:
        """
        Activate emergency stop

        Args:
            reason: Reason for emergency stop
            activated_by: Who activated the emergency stop

        Returns:
            True if emergency stop was activated
        """
        return self.emergency_controls.activate_emergency_stop(reason, activated_by)

    def deactivate_emergency_stop(self, reason: str, deactivated_by: str) -> bool:
        """
        Deactivate emergency stop

        Args:
            reason: Reason for deactivation
            deactivated_by: Who deactivated the emergency stop

        Returns:
            True if emergency stop was deactivated
        """
        return self.emergency_controls.deactivate_emergency_stop(reason, deactivated_by)

    def set_manual_override(
        self, active: bool, duration_hours: int | None = None, reason: str | None = None
    ) -> None:
        """
        Set manual override for automatic switching

        Args:
            active: Whether to activate manual override
            duration_hours: How long to keep override active
            reason: Reason for override
        """
        self.strategy_switcher.set_manual_override(active, duration_hours, reason)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an active alert

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if alert was acknowledged
        """
        return self.emergency_controls.acknowledge_alert(alert_id, acknowledged_by)

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert

        Args:
            alert_id: Alert to resolve

        Returns:
            True if alert was resolved
        """
        return self.emergency_controls.resolve_alert(alert_id)

    def get_strategy_rankings(self, current_regime: RegimeContext | None = None) -> list:
        """
        Get current strategy rankings

        Args:
            current_regime: Current market regime for regime-specific ranking

        Returns:
            List of strategy scores sorted by performance
        """
        return self.strategy_selector.rank_strategies(self.available_strategies, current_regime)

    def get_comprehensive_status(self) -> dict[str, Any]:
        """
        Get comprehensive system status

        Returns:
            Dictionary with complete system status
        """
        return {
            "system_info": {
                "current_strategy": self.current_strategy_id,
                "available_strategies": list(self.available_strategies.keys()),
                "monitoring_enabled": self.monitoring_enabled,
                "last_monitoring_update": self.last_monitoring_update.isoformat(),
            },
            "emergency_status": self.emergency_controls.get_system_status(),
            "switch_statistics": self.strategy_switcher.get_switch_statistics(),
            "active_alerts": [
                alert.to_dict() for alert in self.emergency_controls.get_active_alerts()
            ],
            "pending_approvals": [
                req.to_dict() for req in self.emergency_controls.get_pending_approvals()
            ],
            "performance_baselines": len(self.performance_monitor.performance_baselines),
        }

    def enable_monitoring(self) -> None:
        """Enable automatic monitoring"""
        self.monitoring_enabled = True
        self.logger.info("Monitoring enabled")

    def disable_monitoring(self) -> None:
        """Disable automatic monitoring"""
        self.monitoring_enabled = False
        self.logger.info("Monitoring disabled")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for alert notifications"""
        self.emergency_controls.add_alert_callback(callback)

    def add_approval_callback(self, callback: Callable) -> None:
        """Add callback for approval notifications"""
        self.emergency_controls.add_approval_callback(callback)

    def add_switch_callback(
        self, pre_switch: Callable | None = None, post_switch: Callable | None = None
    ) -> None:
        """Add callbacks for strategy switches"""
        if pre_switch:
            self.strategy_switcher.add_pre_switch_callback(pre_switch)
        if post_switch:
            self.strategy_switcher.add_post_switch_callback(post_switch)
