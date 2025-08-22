"""Unit tests for dynamic risk management"""

from unittest.mock import Mock

import pytest

from src.position_management.dynamic_risk import (
    DynamicRiskConfig,
    DynamicRiskManager,
    RiskAdjustments,
)
from src.risk.risk_manager import RiskParameters


class TestDynamicRiskConfig:
    """Test dynamic risk configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DynamicRiskConfig()
        
        assert config.enabled is True
        assert config.performance_window_days == 30
        assert config.drawdown_thresholds == [0.05, 0.10, 0.15]
        assert config.risk_reduction_factors == [0.8, 0.6, 0.4]
        assert config.recovery_thresholds == [0.02, 0.05]
        assert config.volatility_adjustment_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DynamicRiskConfig(
            enabled=False,
            performance_window_days=60,
            drawdown_thresholds=[0.03, 0.08],
            risk_reduction_factors=[0.9, 0.5]
        )
        
        assert config.enabled is False
        assert config.performance_window_days == 60
        assert config.drawdown_thresholds == [0.03, 0.08]
        assert config.risk_reduction_factors == [0.9, 0.5]
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Mismatched lengths
        with pytest.raises(ValueError):
            DynamicRiskConfig(
                drawdown_thresholds=[0.05, 0.10],
                risk_reduction_factors=[0.8, 0.6, 0.4]
            )
        
        # Invalid risk reduction factors
        with pytest.raises(ValueError):
            DynamicRiskConfig(
                risk_reduction_factors=[1.5, 0.5]  # > 1.0
            )
        
        # Invalid drawdown thresholds
        with pytest.raises(ValueError):
            DynamicRiskConfig(
                drawdown_thresholds=[-0.05, 0.10]  # negative
            )


class TestRiskAdjustments:
    """Test risk adjustments container"""
    
    def test_default_adjustments(self):
        """Test default adjustment values"""
        adj = RiskAdjustments()
        
        assert adj.position_size_factor == 1.0
        assert adj.stop_loss_tightening == 1.0
        assert adj.daily_risk_factor == 1.0
        assert adj.primary_reason == "normal"
        assert adj.adjustment_details == {}
    
    def test_custom_adjustments(self):
        """Test custom adjustment values"""
        adj = RiskAdjustments(
            position_size_factor=0.8,
            stop_loss_tightening=1.2,
            daily_risk_factor=0.7,
            primary_reason="drawdown_10%"
        )
        
        assert adj.position_size_factor == 0.8
        assert adj.stop_loss_tightening == 1.2
        assert adj.daily_risk_factor == 0.7
        assert adj.primary_reason == "drawdown_10%"


class TestDynamicRiskManager:
    """Test dynamic risk manager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = DynamicRiskConfig()
        self.db_manager = Mock()
        self.manager = DynamicRiskManager(self.config, self.db_manager)
    
    def test_disabled_config(self):
        """Test behavior when dynamic risk is disabled"""
        config = DynamicRiskConfig(enabled=False)
        manager = DynamicRiskManager(config)
        
        adjustments = manager.calculate_dynamic_risk_adjustments(
            current_balance=9000,
            peak_balance=10000
        )
        
        assert adjustments.primary_reason == "disabled"
        assert adjustments.position_size_factor == 1.0
        assert adjustments.stop_loss_tightening == 1.0
        assert adjustments.daily_risk_factor == 1.0
    
    def test_no_drawdown(self):
        """Test behavior with no drawdown"""
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000
        )
        
        # Should be normal adjustments when no drawdown
        assert adjustments.position_size_factor >= 0.8  # Could be adjusted by performance/volatility
        assert adjustments.stop_loss_tightening >= 0.8
        assert adjustments.daily_risk_factor >= 0.8
    
    def test_small_drawdown(self):
        """Test behavior with small drawdown (< first threshold)"""
        # 3% drawdown (below 5% threshold)
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=9700,
            peak_balance=10000
        )
        
        # Should have some adjustment but not severe
        assert 0.8 <= adjustments.position_size_factor <= 1.0
    
    def test_large_drawdown(self):
        """Test behavior with large drawdown"""
        # Mock database to return minimal performance data
        self.db_manager.get_performance_metrics.return_value = {
            "total_trades": 5,  # Below minimum for reliable adjustment
            "win_rate": 0.4,
            "profit_factor": 0.8
        }
        
        # 12% drawdown (exceeds 10% threshold)
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=8800,
            peak_balance=10000,
            session_id=1
        )
        
        # Should have significant risk reduction
        assert adjustments.position_size_factor <= 0.6
        assert adjustments.stop_loss_tightening >= 1.2
        assert adjustments.daily_risk_factor <= 0.6
        assert "drawdown" in adjustments.primary_reason
    
    def test_poor_performance_adjustment(self):
        """Test adjustment based on poor performance"""
        # Mock database to return poor performance data
        self.db_manager.get_performance_metrics.return_value = {
            "total_trades": 20,  # Sufficient data
            "win_rate": 0.2,     # Poor win rate
            "profit_factor": 0.5  # Poor profit factor
        }
        
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000,
            session_id=1
        )
        
        # Should reduce risk due to poor performance
        assert adjustments.position_size_factor <= 0.7
        assert adjustments.stop_loss_tightening >= 1.1
    
    def test_good_performance_adjustment(self):
        """Test adjustment based on good performance"""
        # Mock database to return good performance data
        self.db_manager.get_performance_metrics.return_value = {
            "total_trades": 20,  # Sufficient data
            "win_rate": 0.8,     # Good win rate
            "profit_factor": 2.5  # Good profit factor
        }
        
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000,
            session_id=1
        )
        
        # Should allow increased risk due to good performance
        assert adjustments.position_size_factor >= 1.0
    
    def test_apply_risk_adjustments(self):
        """Test applying adjustments to risk parameters"""
        original_params = RiskParameters(
            base_risk_per_trade=0.02,
            max_risk_per_trade=0.03,
            max_position_size=0.25,
            max_daily_risk=0.06
        )
        
        adjustments = RiskAdjustments(
            position_size_factor=0.8,
            stop_loss_tightening=1.2,
            daily_risk_factor=0.7
        )
        
        adjusted_params = self.manager.apply_risk_adjustments(original_params, adjustments)
        
        assert adjusted_params.base_risk_per_trade == pytest.approx(0.016)  # 0.02 * 0.8
        assert adjusted_params.max_risk_per_trade == pytest.approx(0.024)   # 0.03 * 0.8
        assert adjusted_params.max_position_size == pytest.approx(0.2)      # 0.25 * 0.8
        assert adjusted_params.max_daily_risk == pytest.approx(0.042)       # 0.06 * 0.7
        assert adjusted_params.position_size_atr_multiplier == pytest.approx(1.2)  # 1.0 * 1.2
    
    def test_current_drawdown_calculation(self):
        """Test drawdown calculation"""
        # Test normal drawdown
        drawdown = self.manager._calculate_current_drawdown(9000, 10000)
        assert drawdown == 0.1  # 10%
        
        # Test no drawdown
        drawdown = self.manager._calculate_current_drawdown(10000, 10000)
        assert drawdown == 0.0
        
        # Test gain (should return 0)
        drawdown = self.manager._calculate_current_drawdown(11000, 10000)
        assert drawdown == 0.0
        
        # Test zero peak balance
        drawdown = self.manager._calculate_current_drawdown(5000, 0)
        assert drawdown == 0.0
    
    def test_insufficient_data_handling(self):
        """Test behavior with insufficient performance data"""
        # Mock database to return insufficient data
        self.db_manager.get_performance_metrics.return_value = {
            "total_trades": 5,  # Below minimum threshold
            "win_rate": 0.6,
            "profit_factor": 1.5
        }
        
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000,
            session_id=1
        )
        
        # Performance adjustment should indicate insufficient data
        perf_adj = self.manager._calculate_performance_adjustment(
            {"total_trades": 5, "win_rate": 0.6, "profit_factor": 1.5}
        )
        assert perf_adj.primary_reason == "insufficient_data"
    
    def test_database_error_handling(self):
        """Test handling of database errors"""
        # Mock database to raise an exception
        self.db_manager.get_performance_metrics.side_effect = Exception("Database error")
        
        # Should not crash and should handle gracefully
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=10000,
            peak_balance=10000,
            session_id=1
        )
        
        # Should still provide valid adjustments
        assert isinstance(adjustments, RiskAdjustments)
        assert adjustments.position_size_factor > 0
        assert adjustments.stop_loss_tightening > 0
        assert adjustments.daily_risk_factor > 0
    
    def test_extreme_scenarios(self):
        """Test extreme market scenarios"""
        # Test extreme drawdown
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=5000,
            peak_balance=10000  # 50% drawdown
        )
        
        # Should apply maximum risk reduction
        assert adjustments.position_size_factor <= 0.4
        assert adjustments.stop_loss_tightening >= 1.4
        
        # Test zero balance scenario
        adjustments = self.manager.calculate_dynamic_risk_adjustments(
            current_balance=0,
            peak_balance=10000
        )
        
        # Should handle gracefully
        assert isinstance(adjustments, RiskAdjustments)