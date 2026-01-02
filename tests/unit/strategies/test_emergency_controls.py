"""Tests for strategies.components.emergency_controls module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

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


class TestEmergencyLevel:
    """Tests for EmergencyLevel enum."""

    def test_level_ordering(self):
        """Test that emergency levels are properly ordered."""
        assert EmergencyLevel.NONE < EmergencyLevel.LOW
        assert EmergencyLevel.LOW < EmergencyLevel.MEDIUM
        assert EmergencyLevel.MEDIUM < EmergencyLevel.HIGH
        assert EmergencyLevel.HIGH < EmergencyLevel.CRITICAL

    def test_level_comparison(self):
        """Test level comparison operations."""
        assert EmergencyLevel.HIGH > EmergencyLevel.LOW
        assert EmergencyLevel.MEDIUM >= EmergencyLevel.MEDIUM
        assert EmergencyLevel.LOW <= EmergencyLevel.HIGH

    def test_level_values(self):
        """Test that level values are correct."""
        assert EmergencyLevel.NONE.value == 0
        assert EmergencyLevel.LOW.value == 1
        assert EmergencyLevel.MEDIUM.value == 2
        assert EmergencyLevel.HIGH.value == 3
        assert EmergencyLevel.CRITICAL.value == 4


class TestConservativeMode:
    """Tests for ConservativeMode enum."""

    def test_mode_values(self):
        """Test conservative mode values."""
        assert ConservativeMode.DISABLED.value == "disabled"
        assert ConservativeMode.ENABLED.value == "enabled"
        assert ConservativeMode.EMERGENCY_ONLY.value == "emergency_only"


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_status_values(self):
        """Test approval status values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.EXPIRED.value == "expired"
        assert ApprovalStatus.CANCELLED.value == "cancelled"


class TestEmergencyConfig:
    """Tests for EmergencyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmergencyConfig()

        assert config.critical_drawdown_threshold == 0.25
        assert config.high_drawdown_threshold == 0.15
        assert config.consecutive_loss_threshold == 5
        assert config.conservative_position_size_multiplier == 0.5
        assert config.conservative_risk_multiplier == 0.3
        assert config.conservative_confidence_threshold == 0.8
        assert config.approval_timeout_hours == 24
        assert config.require_approval_for_high_risk is True
        assert config.require_approval_for_emergency is False
        assert config.alert_cooldown_minutes == 30
        assert config.max_alerts_per_hour == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmergencyConfig(
            critical_drawdown_threshold=0.30,
            consecutive_loss_threshold=10,
        )

        assert config.critical_drawdown_threshold == 0.30
        assert config.consecutive_loss_threshold == 10


class TestEmergencyAlert:
    """Tests for EmergencyAlert dataclass."""

    def test_to_dict(self):
        """Test alert serialization to dictionary."""
        now = datetime.now(UTC)
        alert = EmergencyAlert(
            alert_id="alert_123",
            alert_type=AlertType.HIGH_DRAWDOWN,
            level=EmergencyLevel.HIGH,
            strategy_id="ml_basic",
            message="High drawdown detected",
            triggered_at=now,
        )

        result = alert.to_dict()

        assert result["alert_id"] == "alert_123"
        assert result["alert_type"] == "high_drawdown"
        assert result["level"] == "high"
        assert result["strategy_id"] == "ml_basic"
        assert result["message"] == "High drawdown detected"
        assert result["acknowledged"] is False
        assert result["resolved"] is False

    def test_acknowledged_alert_serialization(self):
        """Test serialization of acknowledged alert."""
        now = datetime.now(UTC)
        ack_time = now + timedelta(minutes=5)

        alert = EmergencyAlert(
            alert_id="alert_123",
            alert_type=AlertType.CONSECUTIVE_LOSSES,
            level=EmergencyLevel.MEDIUM,
            strategy_id="ml_basic",
            message="5 consecutive losses",
            triggered_at=now,
            acknowledged=True,
            acknowledged_by="admin",
            acknowledged_at=ack_time,
        )

        result = alert.to_dict()

        assert result["acknowledged"] is True
        assert result["acknowledged_by"] == "admin"
        assert result["acknowledged_at"] is not None


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_to_dict(self):
        """Test approval request serialization."""
        now = datetime.now(UTC)
        request = ApprovalRequest(
            request_id="approval_123",
            operation_type="strategy_switch",
            strategy_id="ml_basic",
            requested_by="user1",
            requested_at=now,
            reason="Performance degradation",
            priority=2,
            expires_at=now + timedelta(hours=24),
        )

        result = request.to_dict()

        assert result["request_id"] == "approval_123"
        assert result["operation_type"] == "strategy_switch"
        assert result["status"] == "pending"
        assert result["priority"] == 2


class TestEmergencyControls:
    """Tests for EmergencyControls class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_switcher = MagicMock()
        self.mock_switcher.manual_override_active = False
        self.controls = EmergencyControls(self.mock_switcher)

    def test_initialization(self):
        """Test proper initialization."""
        assert self.controls.emergency_level == EmergencyLevel.NONE
        assert self.controls.conservative_mode == ConservativeMode.DISABLED
        assert self.controls.emergency_stop_active is False
        assert len(self.controls.active_alerts) == 0
        assert len(self.controls.pending_approvals) == 0

    def test_check_emergency_conditions_critical_drawdown(self):
        """Test detection of critical drawdown."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.30  # 30% - above critical threshold
        mock_metrics.consecutive_losses = 0
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        level = self.controls.check_emergency_conditions("ml_basic", mock_tracker)

        assert level == EmergencyLevel.CRITICAL
        assert self.controls.emergency_level == EmergencyLevel.CRITICAL
        assert self.controls.conservative_mode == ConservativeMode.ENABLED

    def test_check_emergency_conditions_high_drawdown(self):
        """Test detection of high drawdown."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.18  # 18% - above high threshold
        mock_metrics.consecutive_losses = 0
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        level = self.controls.check_emergency_conditions("ml_basic", mock_tracker)

        assert level == EmergencyLevel.HIGH
        assert self.controls.conservative_mode == ConservativeMode.ENABLED

    def test_check_emergency_conditions_consecutive_losses(self):
        """Test detection of consecutive losses."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.05
        mock_metrics.consecutive_losses = 6  # Above threshold of 5
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        level = self.controls.check_emergency_conditions("ml_basic", mock_tracker)

        assert level == EmergencyLevel.MEDIUM

    def test_check_emergency_conditions_no_issues(self):
        """Test when no emergency conditions detected."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.05
        mock_metrics.consecutive_losses = 2
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        level = self.controls.check_emergency_conditions("ml_basic", mock_tracker)

        assert level == EmergencyLevel.NONE

    def test_activate_emergency_stop(self):
        """Test emergency stop activation."""
        result = self.controls.activate_emergency_stop("Critical failure", "admin")

        assert result is True
        assert self.controls.emergency_stop_active is True
        assert self.controls.emergency_level == EmergencyLevel.CRITICAL
        assert self.controls.conservative_mode == ConservativeMode.ENABLED

    def test_activate_emergency_stop_already_active(self):
        """Test that activating already active emergency stop returns False."""
        self.controls.activate_emergency_stop("First activation", "admin")
        result = self.controls.activate_emergency_stop("Second activation", "admin")

        assert result is False

    def test_deactivate_emergency_stop(self):
        """Test emergency stop deactivation."""
        self.controls.activate_emergency_stop("Test", "admin")
        result = self.controls.deactivate_emergency_stop("Issue resolved", "admin")

        assert result is True
        assert self.controls.emergency_stop_active is False
        assert self.controls.emergency_level == EmergencyLevel.NONE

    def test_deactivate_emergency_stop_not_active(self):
        """Test deactivating when not active returns False."""
        result = self.controls.deactivate_emergency_stop("Not needed", "admin")

        assert result is False

    def test_activate_conservative_mode(self):
        """Test conservative mode activation."""
        self.controls.activate_conservative_mode("High volatility")

        assert self.controls.conservative_mode == ConservativeMode.ENABLED

    def test_deactivate_conservative_mode(self):
        """Test conservative mode deactivation."""
        self.controls.activate_conservative_mode("Test")
        self.controls.deactivate_conservative_mode("Normal conditions")

        assert self.controls.conservative_mode == ConservativeMode.DISABLED

    def test_get_conservative_adjustments_when_enabled(self):
        """Test getting conservative adjustments when mode is enabled."""
        self.controls.activate_conservative_mode("Test")

        adjustments = self.controls.get_conservative_adjustments()

        assert "position_size_multiplier" in adjustments
        assert "risk_multiplier" in adjustments
        assert "confidence_threshold" in adjustments
        assert adjustments["position_size_multiplier"] == 0.5
        assert adjustments["risk_multiplier"] == 0.3

    def test_get_conservative_adjustments_when_disabled(self):
        """Test getting empty adjustments when mode is disabled."""
        adjustments = self.controls.get_conservative_adjustments()

        assert adjustments == {}

    def test_acknowledge_alert(self):
        """Test alert acknowledgement."""
        # Trigger an alert first
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.30
        mock_metrics.consecutive_losses = 0
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        self.controls.check_emergency_conditions("ml_basic", mock_tracker)

        # Get alert ID
        alerts = self.controls.get_active_alerts()
        assert len(alerts) > 0

        alert_id = alerts[0].alert_id
        result = self.controls.acknowledge_alert(alert_id, "admin")

        assert result is True
        assert alerts[0].acknowledged is True
        assert alerts[0].acknowledged_by == "admin"

    def test_acknowledge_nonexistent_alert(self):
        """Test acknowledging non-existent alert returns False."""
        result = self.controls.acknowledge_alert("fake_id", "admin")

        assert result is False

    def test_resolve_alert(self):
        """Test alert resolution."""
        # Trigger an alert first
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.30
        mock_metrics.consecutive_losses = 0
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        self.controls.check_emergency_conditions("ml_basic", mock_tracker)

        alerts = self.controls.get_active_alerts()
        alert_id = alerts[0].alert_id

        result = self.controls.resolve_alert(alert_id)

        assert result is True
        assert len(self.controls.get_active_alerts()) == 0

    def test_get_system_status(self):
        """Test system status retrieval."""
        status = self.controls.get_system_status()

        assert "emergency_level" in status
        assert "conservative_mode" in status
        assert "emergency_stop_active" in status
        assert "active_alerts" in status
        assert "pending_approvals" in status

    def test_request_manual_strategy_switch_with_approval(self):
        """Test manual switch request requiring approval."""
        request_id = self.controls.request_manual_strategy_switch(
            from_strategy="ml_basic",
            to_strategy="ml_adaptive",
            reason="Better performance",
            requested_by="user1",
        )

        assert request_id is not None
        assert len(self.controls.pending_approvals) == 1

    def test_request_manual_strategy_switch_bypass_approval(self):
        """Test manual switch request bypassing approval."""
        self.mock_switcher.request_manual_switch.return_value = "switch_123"

        request_id = self.controls.request_manual_strategy_switch(
            from_strategy="ml_basic",
            to_strategy="ml_adaptive",
            reason="Emergency switch",
            requested_by="admin",
            bypass_approval=True,
        )

        assert request_id == "switch_123"
        assert len(self.controls.pending_approvals) == 0

    def test_approve_request(self):
        """Test approving a pending request."""
        self.mock_switcher.request_manual_switch.return_value = "switch_123"

        # Create a pending request
        request_id = self.controls.request_manual_strategy_switch(
            from_strategy="ml_basic",
            to_strategy="ml_adaptive",
            reason="Test",
            requested_by="user1",
        )

        # Approve it
        result = self.controls.approve_request(request_id, "admin")

        assert result is True
        assert len(self.controls.pending_approvals) == 0

    def test_reject_request(self):
        """Test rejecting a pending request."""
        request_id = self.controls.request_manual_strategy_switch(
            from_strategy="ml_basic",
            to_strategy="ml_adaptive",
            reason="Test",
            requested_by="user1",
        )

        result = self.controls.approve_request(
            request_id, "admin", rejection_reason="Not approved"
        )

        assert result is True
        assert len(self.controls.pending_approvals) == 0

    def test_add_alert_callback(self):
        """Test adding alert callback."""
        callback = MagicMock()
        self.controls.add_alert_callback(callback)

        assert callback in self.controls.alert_callbacks

    def test_add_approval_callback(self):
        """Test adding approval callback."""
        callback = MagicMock()
        self.controls.add_approval_callback(callback)

        assert callback in self.controls.approval_callbacks


@pytest.mark.fast
class TestEmergencyControlsIntegration:
    """Integration tests for EmergencyControls."""

    def test_full_emergency_workflow(self):
        """Test complete emergency detection and response workflow."""
        mock_switcher = MagicMock()
        mock_switcher.manual_override_active = False
        controls = EmergencyControls(mock_switcher)

        # Simulate deteriorating performance
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        # Phase 1: Normal conditions
        mock_metrics.max_drawdown = 0.05
        mock_metrics.consecutive_losses = 1
        level = controls.check_emergency_conditions("ml_basic", mock_tracker)
        assert level == EmergencyLevel.NONE

        # Phase 2: High drawdown
        mock_metrics.max_drawdown = 0.18
        level = controls.check_emergency_conditions("ml_basic", mock_tracker)
        assert level == EmergencyLevel.HIGH
        assert controls.conservative_mode == ConservativeMode.ENABLED

        # Phase 3: Critical drawdown
        mock_metrics.max_drawdown = 0.30
        level = controls.check_emergency_conditions("ml_basic", mock_tracker)
        assert level == EmergencyLevel.CRITICAL

        # Verify system status
        status = controls.get_system_status()
        assert status["emergency_level"] == "critical"
        assert status["conservative_mode"] == "enabled"

    def test_alert_cooldown(self):
        """Test that alert cooldown prevents duplicate alerts."""
        mock_switcher = MagicMock()
        controls = EmergencyControls(mock_switcher)

        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.max_drawdown = 0.30
        mock_metrics.consecutive_losses = 0
        mock_metrics.sharpe_ratio = 1.0
        mock_metrics.win_rate = 0.5
        mock_metrics.total_return_pct = 0.0
        mock_tracker.get_performance_metrics.return_value = mock_metrics

        # First check - should create alert
        controls.check_emergency_conditions("ml_basic", mock_tracker)
        initial_alerts = len(controls.get_active_alerts())

        # Second check - should not create duplicate (cooldown)
        controls.check_emergency_conditions("ml_basic", mock_tracker)
        final_alerts = len(controls.get_active_alerts())

        # Alert count should be same due to cooldown
        assert initial_alerts == final_alerts
