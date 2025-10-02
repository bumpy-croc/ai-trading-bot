"""
Unit tests for StrategySwitcher

Tests the automatic strategy switching system with validation,
cooling-off periods, audit trails, and performance impact analysis.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.strategies.components.performance_monitor import (
    DegradationSeverity,
    PerformanceMonitor,
    SwitchDecision,
)
from src.strategies.components.performance_tracker import PerformanceTracker
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.strategy_selector import StrategyScore, StrategySelector
from src.strategies.components.strategy_switcher import (
    StrategySwitcher,
    SwitchConfig,
    SwitchRecord,
    SwitchRequest,
    SwitchStatus,
    SwitchTrigger,
    ValidationResult,
)


class TestStrategySwitcher(unittest.TestCase):
    """Test cases for StrategySwitcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.performance_monitor = Mock(spec=PerformanceMonitor)
        self.strategy_selector = Mock(spec=StrategySelector)
        
        self.config = SwitchConfig(
            min_switch_interval_hours=1,  # Short interval for testing
            max_switches_per_day=5,
            max_switches_per_week=20
        )
        
        self.switcher = StrategySwitcher(
            self.performance_monitor,
            self.strategy_selector,
            self.config
        )
        
        # Mock strategies
        self.current_strategy = "current_strategy"
        self.alternative_strategy = "alternative_strategy"
        
        # Mock performance tracker
        self.performance_tracker = Mock(spec=PerformanceTracker)
        self.available_strategies = {
            self.current_strategy: self.performance_tracker,
            self.alternative_strategy: Mock(spec=PerformanceTracker)
        }
        
        # Mock regime context
        self.regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=20,
            strength=0.7
        )
    
    def test_initialization(self):
        """Test StrategySwitcher initialization"""
        switcher = StrategySwitcher(self.performance_monitor, self.strategy_selector)
        
        self.assertIsNotNone(switcher.config)
        self.assertEqual(len(switcher.switch_history), 0)
        self.assertEqual(len(switcher.pending_requests), 0)
        self.assertIsNone(switcher.last_switch_time)
        self.assertFalse(switcher.manual_override_active)
    
    def test_evaluate_switch_need_no_switch_needed(self):
        """Test switch evaluation when no switch is needed"""
        # Mock performance monitor to return no switch needed
        self.performance_monitor.should_switch_strategy.return_value = SwitchDecision(
            should_switch=False,
            reason="Performance is acceptable",
            confidence=0.3
        )
        
        result = self.switcher.evaluate_switch_need(
            self.current_strategy,
            self.performance_tracker,
            self.available_strategies,
            Mock(),  # market_data
            self.regime
        )
        
        self.assertIsNone(result)
        self.performance_monitor.should_switch_strategy.assert_called_once()
    
    def test_evaluate_switch_need_switch_recommended(self):
        """Test switch evaluation when switch is recommended"""
        # Mock performance monitor to recommend switch
        switch_decision = SwitchDecision(
            should_switch=True,
            reason="Performance degradation detected",
            confidence=0.8,
            degradation_severity=DegradationSeverity.MODERATE
        )
        self.performance_monitor.should_switch_strategy.return_value = switch_decision
        
        # Mock strategy selector to return alternatives
        alternative_scores = [
            StrategyScore(
                strategy_id=self.alternative_strategy,
                total_score=0.85,
                criteria_scores={},
                regime_scores={},
                risk_adjusted_score=0.8,
                correlation_penalty=0.1
            )
        ]
        self.strategy_selector.rank_strategies.return_value = alternative_scores
        
        result = self.switcher.evaluate_switch_need(
            self.current_strategy,
            self.performance_tracker,
            self.available_strategies,
            Mock(),  # market_data
            self.regime
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SwitchRequest)
        self.assertEqual(result.from_strategy, self.current_strategy)
        self.assertEqual(result.to_strategy, self.alternative_strategy)
        self.assertEqual(result.trigger, SwitchTrigger.PERFORMANCE_DEGRADATION)
        self.assertEqual(result.priority, 2)  # Moderate severity
    
    def test_evaluate_switch_need_manual_override_active(self):
        """Test switch evaluation when manual override is active"""
        # Activate manual override
        self.switcher.set_manual_override(True, reason="Testing")
        
        result = self.switcher.evaluate_switch_need(
            self.current_strategy,
            self.performance_tracker,
            self.available_strategies,
            Mock(),  # market_data
            self.regime
        )
        
        self.assertIsNone(result)
        # Performance monitor should not be called due to override
        self.performance_monitor.should_switch_strategy.assert_not_called()
    
    def test_evaluate_switch_need_cooling_off_period(self):
        """Test switch evaluation during cooling-off period"""
        # Set last switch time to recent
        self.switcher.last_switch_time = datetime.now() - timedelta(minutes=30)
        
        result = self.switcher.evaluate_switch_need(
            self.current_strategy,
            self.performance_tracker,
            self.available_strategies,
            Mock(),  # market_data
            self.regime
        )
        
        self.assertIsNone(result)
        # Performance monitor should not be called due to cooling-off
        self.performance_monitor.should_switch_strategy.assert_not_called()
    
    def test_request_manual_switch(self):
        """Test manual switch request"""
        request_id = self.switcher.request_manual_switch(
            self.current_strategy,
            self.alternative_strategy,
            "Manual testing",
            "test_user"
        )
        
        self.assertIsNotNone(request_id)
        self.assertIn(request_id, self.switcher.pending_requests)
        
        request = self.switcher.pending_requests[request_id]
        self.assertEqual(request.from_strategy, self.current_strategy)
        self.assertEqual(request.to_strategy, self.alternative_strategy)
        self.assertEqual(request.trigger, SwitchTrigger.MANUAL_REQUEST)
        self.assertEqual(request.requested_by, "test_user")
    
    def test_execute_switch_successful(self):
        """Test successful strategy switch execution"""
        # Create switch request
        request = SwitchRequest(
            request_id="test_request",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=self.current_strategy,
            to_strategy=self.alternative_strategy,
            reason="Test switch",
            requested_at=datetime.now(),
            requested_by="test_user"
        )
        
        # Mock successful activation callback
        activation_callback = Mock(return_value=True)
        
        # Execute switch
        result = self.switcher.execute_switch(request, activation_callback)
        
        self.assertIsInstance(result, SwitchRecord)
        self.assertEqual(result.status, SwitchStatus.COMPLETED)
        self.assertEqual(result.validation_result, ValidationResult.APPROVED)
        self.assertIsNotNone(result.executed_at)
        self.assertIsNotNone(result.completed_at)
        
        # Check that activation callback was called
        activation_callback.assert_called_once_with(self.alternative_strategy)
        
        # Check that switch was added to history
        self.assertEqual(len(self.switcher.switch_history), 1)
        self.assertEqual(self.switcher.switch_history[0], result)
    
    def test_execute_switch_activation_failed(self):
        """Test strategy switch execution with activation failure"""
        request = SwitchRequest(
            request_id="test_request",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=self.current_strategy,
            to_strategy=self.alternative_strategy,
            reason="Test switch",
            requested_at=datetime.now(),
            requested_by="test_user"
        )
        
        # Mock failed activation callback
        activation_callback = Mock(return_value=False)
        
        # Execute switch
        result = self.switcher.execute_switch(request, activation_callback)
        
        self.assertEqual(result.status, SwitchStatus.FAILED)
        self.assertEqual(result.error_message, "Strategy activation failed")
    
    def test_execute_switch_validation_rejected(self):
        """Test strategy switch execution with validation rejection"""
        # Activate manual override to cause validation rejection
        self.switcher.set_manual_override(True, reason="Testing")
        
        request = SwitchRequest(
            request_id="test_request",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=self.current_strategy,
            to_strategy=self.alternative_strategy,
            reason="Test switch",
            requested_at=datetime.now(),
            requested_by="test_user"
        )
        
        activation_callback = Mock(return_value=True)
        
        # Execute switch
        result = self.switcher.execute_switch(request, activation_callback)
        
        self.assertEqual(result.status, SwitchStatus.REJECTED)
        self.assertEqual(result.validation_result, ValidationResult.REJECTED_MANUAL_OVERRIDE)
        
        # Activation callback should not be called
        activation_callback.assert_not_called()
    
    def test_set_manual_override_temporary(self):
        """Test setting temporary manual override"""
        self.switcher.set_manual_override(True, duration_hours=2, reason="Maintenance")
        
        self.assertTrue(self.switcher.manual_override_active)
        self.assertIsNotNone(self.switcher.manual_override_until)
        self.assertEqual(self.switcher.manual_override_reason, "Maintenance")
        
        # Check that override is active
        self.assertTrue(self.switcher._is_manual_override_active())
    
    def test_set_manual_override_permanent(self):
        """Test setting permanent manual override"""
        self.switcher.set_manual_override(True, reason="Long-term testing")
        
        self.assertTrue(self.switcher.manual_override_active)
        self.assertIsNone(self.switcher.manual_override_until)
        self.assertEqual(self.switcher.manual_override_reason, "Long-term testing")
    
    def test_set_manual_override_deactivate(self):
        """Test deactivating manual override"""
        # First activate
        self.switcher.set_manual_override(True, reason="Testing")
        self.assertTrue(self.switcher.manual_override_active)
        
        # Then deactivate
        self.switcher.set_manual_override(False)
        self.assertFalse(self.switcher.manual_override_active)
        self.assertIsNone(self.switcher.manual_override_until)
        self.assertIsNone(self.switcher.manual_override_reason)
    
    def test_manual_override_expiry(self):
        """Test manual override automatic expiry"""
        # Set override to expire in the past
        self.switcher.manual_override_active = True
        self.switcher.manual_override_until = datetime.now() - timedelta(hours=1)
        
        # Check that override is no longer active
        self.assertFalse(self.switcher._is_manual_override_active())
        self.assertFalse(self.switcher.manual_override_active)
    
    def test_can_switch_now_no_previous_switch(self):
        """Test switch timing check with no previous switches"""
        result = self.switcher._can_switch_now(SwitchTrigger.PERFORMANCE_DEGRADATION)
        self.assertTrue(result)
    
    def test_can_switch_now_within_cooling_off(self):
        """Test switch timing check within cooling-off period"""
        self.switcher.last_switch_time = datetime.now() - timedelta(minutes=30)
        
        result = self.switcher._can_switch_now(SwitchTrigger.PERFORMANCE_DEGRADATION)
        self.assertFalse(result)
    
    def test_can_switch_now_after_cooling_off(self):
        """Test switch timing check after cooling-off period"""
        self.switcher.last_switch_time = datetime.now() - timedelta(hours=2)
        
        result = self.switcher._can_switch_now(SwitchTrigger.PERFORMANCE_DEGRADATION)
        self.assertTrue(result)
    
    def test_can_switch_now_emergency_trigger(self):
        """Test switch timing check with emergency trigger"""
        self.switcher.last_switch_time = datetime.now() - timedelta(minutes=30)
        
        # Emergency switches have shorter cooling-off period
        result = self.switcher._can_switch_now(SwitchTrigger.EMERGENCY_STOP)
        self.assertTrue(result)
    
    def test_within_switch_limits_under_limit(self):
        """Test switch limits check when under limits"""
        result = self.switcher._within_switch_limits()
        self.assertTrue(result)
    
    def test_within_switch_limits_daily_limit_exceeded(self):
        """Test switch limits check when daily limit is exceeded"""
        # Add switches to exceed daily limit
        for i in range(self.config.max_switches_per_day):
            record = SwitchRecord(
                switch_id=f"switch_{i}",
                request=SwitchRequest(
                    request_id=f"req_{i}",
                    trigger=SwitchTrigger.MANUAL_REQUEST,
                    from_strategy="strategy_a",
                    to_strategy="strategy_b",
                    reason="Test",
                    requested_at=datetime.now() - timedelta(hours=i),
                    requested_by="test"
                ),
                validation_result=ValidationResult.APPROVED,
                status=SwitchStatus.COMPLETED
            )
            self.switcher.switch_history.append(record)
        
        result = self.switcher._within_switch_limits()
        self.assertFalse(result)
    
    def test_get_switch_history_all(self):
        """Test getting all switch history"""
        # Add some test records
        for i in range(3):
            record = SwitchRecord(
                switch_id=f"switch_{i}",
                request=SwitchRequest(
                    request_id=f"req_{i}",
                    trigger=SwitchTrigger.MANUAL_REQUEST,
                    from_strategy="strategy_a",
                    to_strategy="strategy_b",
                    reason="Test",
                    requested_at=datetime.now() - timedelta(days=i),
                    requested_by="test"
                ),
                validation_result=ValidationResult.APPROVED,
                status=SwitchStatus.COMPLETED
            )
            self.switcher.switch_history.append(record)
        
        history = self.switcher.get_switch_history(days=30)
        self.assertEqual(len(history), 3)
    
    def test_get_switch_history_filtered_by_time(self):
        """Test getting switch history filtered by time"""
        # Add records with different ages
        recent_record = SwitchRecord(
            switch_id="recent",
            request=SwitchRequest(
                request_id="recent_req",
                trigger=SwitchTrigger.MANUAL_REQUEST,
                from_strategy="strategy_a",
                to_strategy="strategy_b",
                reason="Test",
                requested_at=datetime.now() - timedelta(days=1),
                requested_by="test"
            ),
            validation_result=ValidationResult.APPROVED,
            status=SwitchStatus.COMPLETED
        )
        
        old_record = SwitchRecord(
            switch_id="old",
            request=SwitchRequest(
                request_id="old_req",
                trigger=SwitchTrigger.MANUAL_REQUEST,
                from_strategy="strategy_a",
                to_strategy="strategy_b",
                reason="Test",
                requested_at=datetime.now() - timedelta(days=40),
                requested_by="test"
            ),
            validation_result=ValidationResult.APPROVED,
            status=SwitchStatus.COMPLETED
        )
        
        self.switcher.switch_history.extend([recent_record, old_record])
        
        # Get history for last 30 days
        history = self.switcher.get_switch_history(days=30)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].switch_id, "recent")
    
    def test_get_switch_history_filtered_by_strategy(self):
        """Test getting switch history filtered by strategy"""
        # Add records with different strategies
        target_record = SwitchRecord(
            switch_id="target",
            request=SwitchRequest(
                request_id="target_req",
                trigger=SwitchTrigger.MANUAL_REQUEST,
                from_strategy="target_strategy",
                to_strategy="other_strategy",
                reason="Test",
                requested_at=datetime.now() - timedelta(days=1),
                requested_by="test"
            ),
            validation_result=ValidationResult.APPROVED,
            status=SwitchStatus.COMPLETED
        )
        
        other_record = SwitchRecord(
            switch_id="other",
            request=SwitchRequest(
                request_id="other_req",
                trigger=SwitchTrigger.MANUAL_REQUEST,
                from_strategy="other_strategy",
                to_strategy="different_strategy",
                reason="Test",
                requested_at=datetime.now() - timedelta(days=1),
                requested_by="test"
            ),
            validation_result=ValidationResult.APPROVED,
            status=SwitchStatus.COMPLETED
        )
        
        self.switcher.switch_history.extend([target_record, other_record])
        
        # Get history for specific strategy
        history = self.switcher.get_switch_history(days=30, strategy_id="target_strategy")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].switch_id, "target")
    
    def test_get_switch_statistics_empty_history(self):
        """Test getting switch statistics with empty history"""
        stats = self.switcher.get_switch_statistics(days=30)
        
        expected_stats = {
            'total_switches': 0,
            'successful_switches': 0,
            'failed_switches': 0,
            'success_rate': 0.0,
            'avg_switches_per_day': 0.0,
            'triggers': {},
            'most_switched_from': None,
            'most_switched_to': None
        }
        
        for key, value in expected_stats.items():
            self.assertEqual(stats[key], value)
    
    def test_get_switch_statistics_with_history(self):
        """Test getting switch statistics with history"""
        # Add test records
        successful_record = SwitchRecord(
            switch_id="success",
            request=SwitchRequest(
                request_id="success_req",
                trigger=SwitchTrigger.PERFORMANCE_DEGRADATION,
                from_strategy="strategy_a",
                to_strategy="strategy_b",
                reason="Test",
                requested_at=datetime.now() - timedelta(days=1),
                requested_by="test"
            ),
            validation_result=ValidationResult.APPROVED,
            status=SwitchStatus.COMPLETED
        )
        
        failed_record = SwitchRecord(
            switch_id="failed",
            request=SwitchRequest(
                request_id="failed_req",
                trigger=SwitchTrigger.MANUAL_REQUEST,
                from_strategy="strategy_a",
                to_strategy="strategy_c",
                reason="Test",
                requested_at=datetime.now() - timedelta(days=2),
                requested_by="test"
            ),
            validation_result=ValidationResult.APPROVED,
            status=SwitchStatus.FAILED
        )
        
        self.switcher.switch_history.extend([successful_record, failed_record])
        
        stats = self.switcher.get_switch_statistics(days=30)
        
        self.assertEqual(stats['total_switches'], 2)
        self.assertEqual(stats['successful_switches'], 1)
        self.assertEqual(stats['failed_switches'], 1)
        self.assertEqual(stats['success_rate'], 0.5)
        self.assertEqual(stats['avg_switches_per_day'], 2/30)
        self.assertEqual(stats['most_switched_from'], 'strategy_a')
        self.assertIn('performance_degradation', stats['triggers'])
        self.assertIn('manual_request', stats['triggers'])
    
    def test_add_callbacks(self):
        """Test adding pre and post switch callbacks"""
        pre_callback = Mock(return_value=True)
        post_callback = Mock()
        
        self.switcher.add_pre_switch_callback(pre_callback)
        self.switcher.add_post_switch_callback(post_callback)
        
        self.assertIn(pre_callback, self.switcher.pre_switch_callbacks)
        self.assertIn(post_callback, self.switcher.post_switch_callbacks)
    
    def test_callbacks_execution_during_switch(self):
        """Test that callbacks are executed during switch"""
        pre_callback = Mock(return_value=True)
        post_callback = Mock()
        
        self.switcher.add_pre_switch_callback(pre_callback)
        self.switcher.add_post_switch_callback(post_callback)
        
        request = SwitchRequest(
            request_id="test_request",
            trigger=SwitchTrigger.MANUAL_REQUEST,
            from_strategy=self.current_strategy,
            to_strategy=self.alternative_strategy,
            reason="Test switch",
            requested_at=datetime.now(),
            requested_by="test_user"
        )
        
        activation_callback = Mock(return_value=True)
        
        # Execute switch
        result = self.switcher.execute_switch(request, activation_callback)
        
        # Check that callbacks were called
        pre_callback.assert_called_once_with(self.current_strategy, self.alternative_strategy)
        post_callback.assert_called_once_with(self.current_strategy, self.alternative_strategy, True)
        
        self.assertEqual(result.status, SwitchStatus.COMPLETED)


if __name__ == '__main__':
    unittest.main()