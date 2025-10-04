"""
Integration tests for Performance Monitoring System

Tests the complete integration of all performance monitoring and
automatic strategy switching components.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd

from src.strategies.components.performance_monitoring_system import PerformanceMonitoringSystem
from src.strategies.components.performance_tracker import PerformanceTracker, TradeResult
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel


class TestPerformanceMonitoringSystemIntegration(unittest.TestCase):
    """Integration test cases for PerformanceMonitoringSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitoring_system = PerformanceMonitoringSystem()
        
        # Create mock strategies with performance trackers
        self.strategy_a = PerformanceTracker("strategy_a")
        self.strategy_b = PerformanceTracker("strategy_b")
        
        # Populate with some test data
        self._populate_strategy_performance(self.strategy_a, good_performance=False)
        self._populate_strategy_performance(self.strategy_b, good_performance=True)
        
        # Register strategies
        self.monitoring_system.register_strategy("strategy_a", self.strategy_a)
        self.monitoring_system.register_strategy("strategy_b", self.strategy_b)
        
        # Set current strategy
        self.monitoring_system.set_current_strategy("strategy_a")
        
        # Mock strategy activation callback
        self.activation_callback = Mock(return_value=True)
        self.monitoring_system.set_strategy_activation_callback(self.activation_callback)
        
        # Create mock regime context
        self.regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=20,
            strength=0.7
        )
        
        # Create mock market data
        self.market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'close': [50000 + i * 10 for i in range(100)]
        })
    
    def _populate_strategy_performance(self, tracker: PerformanceTracker, good_performance: bool = True):
        """Populate a strategy with test performance data"""
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(20):
            timestamp = base_time + timedelta(days=i)
            
            if good_performance:
                # Good performance: mostly winning trades
                pnl = 100.0 if i % 4 != 0 else -50.0  # 75% win rate
                pnl_percent = 0.002 if i % 4 != 0 else -0.001
            else:
                # Poor performance: mostly losing trades, high drawdown
                pnl = 50.0 if i % 4 == 0 else -100.0  # 25% win rate
                pnl_percent = 0.001 if i % 4 == 0 else -0.002
            
            trade = TradeResult(
                timestamp=timestamp,
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                exit_price=50000.0 + pnl,
                quantity=0.001,
                pnl=pnl,
                pnl_percent=pnl_percent,
                duration_hours=24.0,
                strategy_id=tracker.strategy_id,
                confidence=0.7,
                regime="trend_up_low_vol"
            )
            
            tracker.record_trade(trade)
    
    def test_system_initialization(self):
        """Test that the monitoring system initializes correctly"""
        system = PerformanceMonitoringSystem()
        
        self.assertIsNotNone(system.performance_monitor)
        self.assertIsNotNone(system.strategy_selector)
        self.assertIsNotNone(system.strategy_switcher)
        self.assertIsNotNone(system.emergency_controls)
        self.assertTrue(system.monitoring_enabled)
    
    def test_strategy_registration(self):
        """Test strategy registration"""
        system = PerformanceMonitoringSystem()
        tracker = PerformanceTracker("test_strategy")
        
        system.register_strategy("test_strategy", tracker)
        
        self.assertIn("test_strategy", system.available_strategies)
        self.assertEqual(system.available_strategies["test_strategy"], tracker)
    
    def test_current_strategy_setting(self):
        """Test setting current strategy"""
        system = PerformanceMonitoringSystem()
        tracker = PerformanceTracker("test_strategy")
        system.register_strategy("test_strategy", tracker)
        
        result = system.set_current_strategy("test_strategy")
        
        self.assertTrue(result)
        self.assertEqual(system.current_strategy_id, "test_strategy")
    
    def test_current_strategy_setting_invalid(self):
        """Test setting invalid current strategy"""
        system = PerformanceMonitoringSystem()
        
        result = system.set_current_strategy("nonexistent_strategy")
        
        self.assertFalse(result)
        self.assertIsNone(system.current_strategy_id)
    
    def test_monitoring_update_basic(self):
        """Test basic monitoring update"""
        results = self.monitoring_system.update_monitoring(self.market_data, self.regime)
        
        self.assertIn('timestamp', results)
        self.assertIn('current_strategy', results)
        self.assertIn('monitoring_enabled', results)
        self.assertIn('system_status', results)
        self.assertTrue(results['monitoring_enabled'])
        self.assertEqual(results['current_strategy'], 'strategy_a')
    
    def test_monitoring_update_with_poor_performance(self):
        """Test monitoring update with poor performing strategy"""
        # The strategy_a has poor performance, so it might trigger alerts or switches
        results = self.monitoring_system.update_monitoring(self.market_data, self.regime)
        
        # Should have some monitoring activity
        self.assertIn('emergency_level', results)
        self.assertIn('system_status', results)
        
        # May have triggered alerts or switch recommendations
        if results.get('switch_recommendations'):
            self.assertIsInstance(results['switch_recommendations'], list)
        
        if results.get('actions_taken'):
            self.assertIsInstance(results['actions_taken'], list)
    
    def test_manual_switch_request(self):
        """Test manual strategy switch request"""
        request_id = self.monitoring_system.request_manual_switch(
            "strategy_b", "Manual test", "test_user"
        )
        
        self.assertIsNotNone(request_id)
        self.assertIsInstance(request_id, str)
    
    def test_emergency_stop_activation(self):
        """Test emergency stop activation"""
        result = self.monitoring_system.activate_emergency_stop("Test emergency", "admin")
        
        self.assertTrue(result)
        
        # Check system status
        status = self.monitoring_system.get_comprehensive_status()
        self.assertTrue(status['emergency_status']['emergency_stop_active'])
    
    def test_emergency_stop_deactivation(self):
        """Test emergency stop deactivation"""
        # First activate
        self.monitoring_system.activate_emergency_stop("Test emergency", "admin")
        
        # Then deactivate
        result = self.monitoring_system.deactivate_emergency_stop("Issue resolved", "admin")
        
        self.assertTrue(result)
        
        # Check system status
        status = self.monitoring_system.get_comprehensive_status()
        self.assertFalse(status['emergency_status']['emergency_stop_active'])
    
    def test_manual_override(self):
        """Test manual override functionality"""
        self.monitoring_system.set_manual_override(True, duration_hours=1, reason="Testing")
        
        # Check that override is active
        status = self.monitoring_system.get_comprehensive_status()
        self.assertTrue(status['emergency_status']['manual_override_active'])
        
        # Deactivate override
        self.monitoring_system.set_manual_override(False)
        
        # Check that override is inactive
        status = self.monitoring_system.get_comprehensive_status()
        self.assertFalse(status['emergency_status']['manual_override_active'])
    
    def test_strategy_rankings(self):
        """Test getting strategy rankings"""
        rankings = self.monitoring_system.get_strategy_rankings(self.regime)
        
        self.assertIsInstance(rankings, list)
        
        if rankings:
            # Should have strategy scores
            for ranking in rankings:
                self.assertIn('strategy_id', ranking.__dict__)
                self.assertIn('total_score', ranking.__dict__)
    
    def test_comprehensive_status(self):
        """Test getting comprehensive system status"""
        status = self.monitoring_system.get_comprehensive_status()
        
        expected_keys = [
            'system_info', 'emergency_status', 'switch_statistics',
            'active_alerts', 'pending_approvals', 'performance_baselines'
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)
        
        # Check system info
        self.assertIn('current_strategy', status['system_info'])
        self.assertIn('available_strategies', status['system_info'])
        self.assertIn('monitoring_enabled', status['system_info'])
    
    def test_monitoring_enable_disable(self):
        """Test enabling and disabling monitoring"""
        # Initially enabled
        self.assertTrue(self.monitoring_system.monitoring_enabled)
        
        # Disable monitoring
        self.monitoring_system.disable_monitoring()
        self.assertFalse(self.monitoring_system.monitoring_enabled)
        
        # Update should indicate monitoring is disabled
        results = self.monitoring_system.update_monitoring(self.market_data, self.regime)
        self.assertFalse(results['monitoring_enabled'])
        
        # Re-enable monitoring
        self.monitoring_system.enable_monitoring()
        self.assertTrue(self.monitoring_system.monitoring_enabled)
    
    def test_callback_registration(self):
        """Test callback registration"""
        alert_callback = Mock()
        approval_callback = Mock()
        pre_switch_callback = Mock(return_value=True)
        post_switch_callback = Mock()
        
        # Add callbacks
        self.monitoring_system.add_alert_callback(alert_callback)
        self.monitoring_system.add_approval_callback(approval_callback)
        self.monitoring_system.add_switch_callback(pre_switch_callback, post_switch_callback)
        
        # Callbacks should be registered (we can't easily test execution without triggering actual events)
        self.assertIn(alert_callback, self.monitoring_system.emergency_controls.alert_callbacks)
        self.assertIn(approval_callback, self.monitoring_system.emergency_controls.approval_callbacks)
        self.assertIn(pre_switch_callback, self.monitoring_system.strategy_switcher.pre_switch_callbacks)
        self.assertIn(post_switch_callback, self.monitoring_system.strategy_switcher.post_switch_callbacks)


if __name__ == '__main__':
    unittest.main()