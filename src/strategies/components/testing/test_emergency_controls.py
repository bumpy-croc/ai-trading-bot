"""
Unit tests for EmergencyControls

Tests the emergency controls and manual override system including
emergency detection, conservative mode, approval workflows, and alerting.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.strategies.components.emergency_controls import (
    AlertType,
    ApprovalRequest,
    ApprovalStatus,
    ConservativeMode,
    EmergencyAlert,
    EmergencyConfig,
    EmergencyControls,
    EmergencyLevel,
)
from src.strategies.components.performance_tracker import (
    PerformanceMetrics,
    PerformancePeriod,
    PerformanceTracker,
)
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.strategy_switcher import StrategySwitcher


class TestEmergencyControls(unittest.TestCase):
    """Test cases for EmergencyControls"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy_switcher = Mock(spec=StrategySwitcher)
        self.config = EmergencyConfig(
            critical_drawdown_threshold=0.25,
            high_drawdown_threshold=0.15,
            consecutive_loss_threshold=5,
            approval_timeout_hours=1  # Short timeout for testing
        )
        
        self.emergency_controls = EmergencyControls(self.strategy_switcher, self.config)
        
        # Mock performance tracker
        self.performance_tracker = Mock(spec=PerformanceTracker)
        
        # Mock regime context
        self.regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=20,
            strength=0.7
        )
    
    def _create_performance_metrics(self, max_drawdown: float = 0.1,
                                  consecutive_losses: int = 2,
                                  sharpe_ratio: float = 1.5,
                                  win_rate: float = 0.6) -> PerformanceMetrics:
        """Create mock performance metrics"""
        return PerformanceMetrics(
            total_return=500.0, total_return_pct=0.25, annualized_return=0.91,
            volatility=0.12, sharpe_ratio=sharpe_ratio, sortino_ratio=2.0, calmar_ratio=4.5,
            max_drawdown=max_drawdown, var_95=-0.02, total_trades=50, winning_trades=35,
            losing_trades=15, win_rate=win_rate, avg_win=0.03, avg_loss=-0.01,
            profit_factor=2.1, expectancy=0.005, avg_trade_duration=20.0,
            trades_per_day=1.2, hit_rate=0.7, max_drawdown_duration=3.0,
            current_drawdown=0.05, drawdown_recovery_time=1.0, best_trade=0.06,
            worst_trade=-0.02, consecutive_wins=8, consecutive_losses=consecutive_losses,
            period_start=datetime.now() - timedelta(days=60),
            period_end=datetime.now(), period_type=PerformancePeriod.ALL_TIME
        )
    
    def test_initialization(self):
        """Test EmergencyControls initialization"""
        controls = EmergencyControls(self.strategy_switcher)
        
        self.assertIsNotNone(controls.config)
        self.assertEqual(controls.emergency_level, EmergencyLevel.NONE)
        self.assertEqual(controls.conservative_mode, ConservativeMode.DISABLED)
        self.assertFalse(controls.emergency_stop_active)
        self.assertEqual(len(controls.active_alerts), 0)
        self.assertEqual(len(controls.pending_approvals), 0)
    
    def test_check_emergency_conditions_none(self):
        """Test emergency condition check with normal performance"""
        metrics = self._create_performance_metrics(
            max_drawdown=0.08,  # Below thresholds
            consecutive_losses=2
        )
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        level = self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        self.assertEqual(level, EmergencyLevel.NONE)
        self.assertEqual(len(self.emergency_controls.active_alerts), 0)
    
    def test_check_emergency_conditions_high_drawdown(self):
        """Test emergency condition check with high drawdown"""
        metrics = self._create_performance_metrics(
            max_drawdown=0.18,  # Above high threshold
            consecutive_losses=2
        )
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        level = self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        self.assertEqual(level, EmergencyLevel.HIGH)
        self.assertGreater(len(self.emergency_controls.active_alerts), 0)
        
        # Should have triggered high drawdown alert
        alert_types = [alert.alert_type for alert in self.emergency_controls.active_alerts.values()]
        self.assertIn(AlertType.HIGH_DRAWDOWN, alert_types)
    
    def test_check_emergency_conditions_critical_drawdown(self):
        """Test emergency condition check with critical drawdown"""
        metrics = self._create_performance_metrics(
            max_drawdown=0.28,  # Above critical threshold
            consecutive_losses=2
        )
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        level = self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        self.assertEqual(level, EmergencyLevel.CRITICAL)
        self.assertGreater(len(self.emergency_controls.active_alerts), 0)
        
        # Should have activated conservative mode
        self.assertEqual(self.emergency_controls.conservative_mode, ConservativeMode.ENABLED)
    
    def test_check_emergency_conditions_consecutive_losses(self):
        """Test emergency condition check with consecutive losses"""
        metrics = self._create_performance_metrics(
            max_drawdown=0.08,  # Normal drawdown
            consecutive_losses=6  # Above threshold
        )
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        level = self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        self.assertEqual(level, EmergencyLevel.MEDIUM)
        self.assertGreater(len(self.emergency_controls.active_alerts), 0)
        
        # Should have triggered consecutive losses alert
        alert_types = [alert.alert_type for alert in self.emergency_controls.active_alerts.values()]
        self.assertIn(AlertType.CONSECUTIVE_LOSSES, alert_types)
    
    def test_request_manual_strategy_switch_direct(self):
        """Test manual strategy switch request without approval"""
        # Configure to not require approval
        self.emergency_controls.config.require_approval_for_high_risk = False
        
        self.strategy_switcher.request_manual_switch.return_value = "switch_request_123"
        
        request_id = self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        
        self.assertEqual(request_id, "switch_request_123")
        self.strategy_switcher.request_manual_switch.assert_called_once_with(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        self.assertEqual(len(self.emergency_controls.pending_approvals), 0)
    
    def test_request_manual_strategy_switch_with_approval(self):
        """Test manual strategy switch request requiring approval"""
        # Configure to require approval
        self.emergency_controls.config.require_approval_for_high_risk = True
        
        request_id = self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        
        # Should create approval request
        self.assertIsNotNone(request_id)
        self.assertEqual(len(self.emergency_controls.pending_approvals), 1)
        self.assertIn(request_id, self.emergency_controls.pending_approvals)
        
        # Should not call strategy switcher directly
        self.strategy_switcher.request_manual_switch.assert_not_called()
    
    def test_request_manual_strategy_switch_bypass_approval(self):
        """Test manual strategy switch request bypassing approval"""
        # Configure to require approval but bypass it
        self.emergency_controls.config.require_approval_for_high_risk = True
        
        self.strategy_switcher.request_manual_switch.return_value = "switch_request_123"
        
        request_id = self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Emergency switch", "test_user",
            bypass_approval=True
        )
        
        self.assertEqual(request_id, "switch_request_123")
        self.strategy_switcher.request_manual_switch.assert_called_once()
        self.assertEqual(len(self.emergency_controls.pending_approvals), 0)
    
    def test_approve_request_approval(self):
        """Test approving a pending request"""
        # Create approval request
        request_id = self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        
        self.strategy_switcher.request_manual_switch.return_value = "switch_request_123"
        
        # Approve the request
        result = self.emergency_controls.approve_request(request_id, "approver")
        
        self.assertTrue(result)
        self.assertEqual(len(self.emergency_controls.pending_approvals), 0)
        self.assertGreater(len(self.emergency_controls.approval_history), 0)
        
        # Should have called strategy switcher
        self.strategy_switcher.request_manual_switch.assert_called_once()
    
    def test_approve_request_rejection(self):
        """Test rejecting a pending request"""
        # Create approval request
        request_id = self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        
        # Reject the request
        result = self.emergency_controls.approve_request(
            request_id, "approver", "Not suitable"
        )
        
        self.assertTrue(result)
        self.assertEqual(len(self.emergency_controls.pending_approvals), 0)
        self.assertGreater(len(self.emergency_controls.approval_history), 0)
        
        # Should not have called strategy switcher
        self.strategy_switcher.request_manual_switch.assert_not_called()
        
        # Check rejection details
        approved_request = self.emergency_controls.approval_history[-1]
        self.assertEqual(approved_request.status, ApprovalStatus.REJECTED)
        self.assertEqual(approved_request.rejection_reason, "Not suitable")
    
    def test_approve_request_not_found(self):
        """Test approving non-existent request"""
        result = self.emergency_controls.approve_request("nonexistent", "approver")
        self.assertFalse(result)
    
    def test_approve_request_expired(self):
        """Test approving expired request"""
        # Create approval request with past expiry
        approval_request = ApprovalRequest(
            request_id="expired_request",
            operation_type="strategy_switch",
            strategy_id="test_strategy",
            requested_by="test_user",
            requested_at=datetime.now() - timedelta(hours=2),
            reason="Test",
            priority=1,
            expires_at=datetime.now() - timedelta(hours=1)  # Expired
        )
        
        self.emergency_controls.pending_approvals["expired_request"] = approval_request
        
        result = self.emergency_controls.approve_request("expired_request", "approver")
        
        self.assertFalse(result)
        self.assertEqual(approval_request.status, ApprovalStatus.EXPIRED)
    
    def test_activate_emergency_stop(self):
        """Test activating emergency stop"""
        result = self.emergency_controls.activate_emergency_stop("System failure", "admin")
        
        self.assertTrue(result)
        self.assertTrue(self.emergency_controls.emergency_stop_active)
        self.assertEqual(self.emergency_controls.emergency_level, EmergencyLevel.CRITICAL)
        self.assertEqual(self.emergency_controls.conservative_mode, ConservativeMode.ENABLED)
        
        # Should have triggered emergency alert
        alert_types = [alert.alert_type for alert in self.emergency_controls.active_alerts.values()]
        self.assertIn(AlertType.EMERGENCY_STOP, alert_types)
    
    def test_activate_emergency_stop_already_active(self):
        """Test activating emergency stop when already active"""
        # First activation
        self.emergency_controls.activate_emergency_stop("System failure", "admin")
        
        # Second activation should return False
        result = self.emergency_controls.activate_emergency_stop("Another failure", "admin")
        self.assertFalse(result)
    
    def test_deactivate_emergency_stop(self):
        """Test deactivating emergency stop"""
        # First activate
        self.emergency_controls.activate_emergency_stop("System failure", "admin")
        
        # Then deactivate
        result = self.emergency_controls.deactivate_emergency_stop("Issue resolved", "admin")
        
        self.assertTrue(result)
        self.assertFalse(self.emergency_controls.emergency_stop_active)
        self.assertEqual(self.emergency_controls.emergency_level, EmergencyLevel.NONE)
    
    def test_deactivate_emergency_stop_not_active(self):
        """Test deactivating emergency stop when not active"""
        result = self.emergency_controls.deactivate_emergency_stop("Not needed", "admin")
        self.assertFalse(result)
    
    def test_activate_conservative_mode(self):
        """Test activating conservative mode"""
        self.emergency_controls.activate_conservative_mode("High volatility")
        
        self.assertEqual(self.emergency_controls.conservative_mode, ConservativeMode.ENABLED)
    
    def test_activate_conservative_mode_already_active(self):
        """Test activating conservative mode when already active"""
        self.emergency_controls.activate_conservative_mode("High volatility")
        
        # Second activation should not change anything
        self.emergency_controls.activate_conservative_mode("Another reason")
        
        self.assertEqual(self.emergency_controls.conservative_mode, ConservativeMode.ENABLED)
    
    def test_deactivate_conservative_mode(self):
        """Test deactivating conservative mode"""
        # First activate
        self.emergency_controls.activate_conservative_mode("High volatility")
        
        # Then deactivate
        self.emergency_controls.deactivate_conservative_mode("Volatility normalized")
        
        self.assertEqual(self.emergency_controls.conservative_mode, ConservativeMode.DISABLED)
    
    def test_get_conservative_adjustments_disabled(self):
        """Test getting conservative adjustments when disabled"""
        adjustments = self.emergency_controls.get_conservative_adjustments()
        self.assertEqual(adjustments, {})
    
    def test_get_conservative_adjustments_enabled(self):
        """Test getting conservative adjustments when enabled"""
        self.emergency_controls.activate_conservative_mode("Test")
        
        adjustments = self.emergency_controls.get_conservative_adjustments()
        
        expected_keys = ['position_size_multiplier', 'risk_multiplier', 'confidence_threshold']
        for key in expected_keys:
            self.assertIn(key, adjustments)
            self.assertIsInstance(adjustments[key], float)
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert"""
        # Trigger an alert first
        metrics = self._create_performance_metrics(max_drawdown=0.18)
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        # Get the alert ID
        alert_id = list(self.emergency_controls.active_alerts.keys())[0]
        
        # Acknowledge the alert
        result = self.emergency_controls.acknowledge_alert(alert_id, "operator")
        
        self.assertTrue(result)
        
        alert = self.emergency_controls.active_alerts[alert_id]
        self.assertTrue(alert.acknowledged)
        self.assertEqual(alert.acknowledged_by, "operator")
        self.assertIsNotNone(alert.acknowledged_at)
    
    def test_acknowledge_alert_not_found(self):
        """Test acknowledging non-existent alert"""
        result = self.emergency_controls.acknowledge_alert("nonexistent", "operator")
        self.assertFalse(result)
    
    def test_resolve_alert(self):
        """Test resolving an alert"""
        # Trigger an alert first
        metrics = self._create_performance_metrics(max_drawdown=0.18)
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        # Get the alert ID
        alert_id = list(self.emergency_controls.active_alerts.keys())[0]
        
        # Resolve the alert
        result = self.emergency_controls.resolve_alert(alert_id)
        
        self.assertTrue(result)
        self.assertNotIn(alert_id, self.emergency_controls.active_alerts)
        self.assertGreater(len(self.emergency_controls.alert_history), 0)
    
    def test_resolve_alert_not_found(self):
        """Test resolving non-existent alert"""
        result = self.emergency_controls.resolve_alert("nonexistent")
        self.assertFalse(result)
    
    def test_get_system_status(self):
        """Test getting system status"""
        status = self.emergency_controls.get_system_status()
        
        expected_keys = [
            'emergency_level', 'conservative_mode', 'emergency_stop_active',
            'active_alerts', 'pending_approvals', 'manual_override_active',
            'last_performance_check', 'last_emergency_check', 'conservative_adjustments'
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        # Initially no alerts
        alerts = self.emergency_controls.get_active_alerts()
        self.assertEqual(len(alerts), 0)
        
        # Trigger an alert
        metrics = self._create_performance_metrics(max_drawdown=0.18)
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        # Should have alerts now
        alerts = self.emergency_controls.get_active_alerts()
        self.assertGreater(len(alerts), 0)
        self.assertIsInstance(alerts[0], EmergencyAlert)
    
    def test_get_pending_approvals(self):
        """Test getting pending approvals"""
        # Initially no approvals
        approvals = self.emergency_controls.get_pending_approvals()
        self.assertEqual(len(approvals), 0)
        
        # Create approval request
        self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        
        # Should have approvals now
        approvals = self.emergency_controls.get_pending_approvals()
        self.assertGreater(len(approvals), 0)
        self.assertIsInstance(approvals[0], ApprovalRequest)
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism"""
        metrics = self._create_performance_metrics(max_drawdown=0.18)
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        # First check should trigger alert
        self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        initial_alert_count = len(self.emergency_controls.active_alerts)
        self.assertGreater(initial_alert_count, 0)
        
        # Second check immediately should not trigger new alert (cooldown)
        self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        # Should still have same number of alerts
        self.assertEqual(len(self.emergency_controls.active_alerts), initial_alert_count)
    
    def test_cleanup_expired_requests(self):
        """Test cleanup of expired approval requests"""
        # Create expired approval request
        expired_request = ApprovalRequest(
            request_id="expired_request",
            operation_type="strategy_switch",
            strategy_id="test_strategy",
            requested_by="test_user",
            requested_at=datetime.now() - timedelta(hours=2),
            reason="Test",
            priority=1,
            expires_at=datetime.now() - timedelta(hours=1)  # Expired
        )
        
        self.emergency_controls.pending_approvals["expired_request"] = expired_request
        
        # Run cleanup
        self.emergency_controls.cleanup_expired_requests()
        
        # Should be moved to history
        self.assertEqual(len(self.emergency_controls.pending_approvals), 0)
        self.assertGreater(len(self.emergency_controls.approval_history), 0)
        self.assertEqual(expired_request.status, ApprovalStatus.EXPIRED)
    
    def test_add_callbacks(self):
        """Test adding alert and approval callbacks"""
        alert_callback = Mock()
        approval_callback = Mock()
        
        self.emergency_controls.add_alert_callback(alert_callback)
        self.emergency_controls.add_approval_callback(approval_callback)
        
        self.assertIn(alert_callback, self.emergency_controls.alert_callbacks)
        self.assertIn(approval_callback, self.emergency_controls.approval_callbacks)
    
    def test_callback_execution(self):
        """Test that callbacks are executed"""
        alert_callback = Mock()
        approval_callback = Mock()
        
        self.emergency_controls.add_alert_callback(alert_callback)
        self.emergency_controls.add_approval_callback(approval_callback)
        
        # Trigger alert
        metrics = self._create_performance_metrics(max_drawdown=0.18)
        self.performance_tracker.get_performance_metrics.return_value = metrics
        
        self.emergency_controls.check_emergency_conditions(
            "test_strategy", self.performance_tracker, self.regime
        )
        
        # Alert callback should be called
        alert_callback.assert_called_once()
        
        # Create approval request
        self.emergency_controls.request_manual_strategy_switch(
            "strategy_a", "strategy_b", "Manual test", "test_user"
        )
        
        # Approval callback should be called
        approval_callback.assert_called_once()


if __name__ == '__main__':
    unittest.main()