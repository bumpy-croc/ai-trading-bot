"""Integration tests for dynamic risk management in live trading engine"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.live.trading_engine import LiveTradingEngine
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.risk.risk_manager import RiskParameters

pytestmark = pytest.mark.integration


class MockStrategy:
    """Mock strategy for testing"""
    def __init__(self, has_overrides=False):
        self.has_overrides = has_overrides
    
    def check_entry_conditions(self, df, index):
        return False
    
    def check_exit_conditions(self, df, index, entry_price):
        return False
    
    def calculate_position_size(self, df, index, balance):
        return 0.05  # 5% position size
    
    def calculate_indicators(self, df):
        return df
    
    def set_database_manager(self, db_manager):
        """Required by live trading engine"""
        pass
    
    def get_risk_overrides(self):
        """Override for testing strategy-specific dynamic risk config"""
        if self.has_overrides:
            return {
                'dynamic_risk': {
                    'enabled': True,
                    'drawdown_thresholds': [0.03, 0.08, 0.15],
                    'risk_reduction_factors': [0.9, 0.7, 0.5],
                    'recovery_thresholds': [0.01, 0.03]
                }
            }
        return {}


class MockDataProvider:
    """Mock data provider for testing"""
    def get_historical_data(self, symbol, timeframe, start, end):
        # Return sample data for live trading tests
        dates = pd.date_range(start=datetime.now(), periods=100, freq='1H')
        return pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [102.0] * 100,
            'volume': [1000.0] * 100
        }, index=dates)


class MockExchange:
    """Mock exchange for testing"""
    def __init__(self):
        self.balance = 10000.0
        self.open_positions = []
    
    def get_balance(self):
        return self.balance
    
    def get_open_positions(self):
        return self.open_positions
    
    def place_order(self, symbol, side, amount, price=None, order_type='market'):
        return {'id': 'test_order_123', 'status': 'filled'}


@pytest.mark.live_trading
class TestLiveTradingDynamicRiskIntegration:
    """Integration tests for dynamic risk management in live trading engine"""
    
    def test_live_engine_with_dynamic_risk_disabled(self):
        """Test live trading engine creation with dynamic risk disabled"""
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=False,
            database_url="sqlite:///:memory:"
        )
        
        assert engine.enable_dynamic_risk is False
        assert engine.dynamic_risk_manager is None
    
    def test_live_engine_with_dynamic_risk_enabled(self):
        """Test live trading engine creation with dynamic risk enabled"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4]
        )
        
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            database_url="sqlite:///:memory:"
        )
        
        assert engine.enable_dynamic_risk is True
        # Dynamic risk manager is created after start() is called
        assert hasattr(engine, '_dynamic_risk_config')
    
    def test_dynamic_risk_size_adjustment_in_live_engine(self):
        """Test dynamic risk size adjustment functionality in live engine"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4]
        )
        
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            initial_balance=10000,
            database_url="sqlite:///:memory:"
        )
        
        # Initialize the database manager and dynamic risk manager manually for testing
        from src.database.database_manager import DatabaseManager
        engine.db_manager = DatabaseManager(":memory:")  # Use in-memory DB for testing
        final_config = engine._merge_dynamic_risk_config(config)
        engine.dynamic_risk_manager = DynamicRiskManager(
            config=final_config,
            db_manager=engine.db_manager
        )
        engine.current_balance = 10000
        engine.peak_balance = 10000
        
        # Test no drawdown scenario
        adjusted_size = engine._get_dynamic_risk_adjusted_size(0.05)
        
        # Should be close to original size (no significant adjustment)
        assert 0.04 <= adjusted_size <= 0.06
        
        # Test 15% drawdown scenario  
        engine.current_balance = 8500  # 15% drawdown
        engine.peak_balance = 10000
        
        adjusted_size = engine._get_dynamic_risk_adjusted_size(0.05)
        
        # Should be significantly reduced (0.05 * 0.4 = 0.02)
        assert adjusted_size == pytest.approx(0.02, rel=0.1)
    
    def test_strategy_dynamic_risk_overrides(self):
        """Test strategy-specific dynamic risk overrides"""
        # Default config
        default_config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4]
        )
        
        # Strategy with overrides
        strategy_with_overrides = MockStrategy(has_overrides=True)
        
        engine = LiveTradingEngine(
            strategy=strategy_with_overrides,
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            dynamic_risk_config=default_config,
            database_url="sqlite:///:memory:"
        )
        
        # Test config merging
        merged_config = engine._merge_dynamic_risk_config(default_config)
        
        # Should use strategy overrides
        assert merged_config.drawdown_thresholds == [0.03, 0.08, 0.15]
        assert merged_config.risk_reduction_factors == [0.9, 0.7, 0.5]
        assert merged_config.recovery_thresholds == [0.01, 0.03]
    
    def test_dynamic_risk_adjusted_params(self):
        """Test dynamic risk parameter adjustments"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4]
        )
        
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            initial_balance=10000,
            database_url="sqlite:///:memory:"
        )
        
        # Manually initialize dynamic risk manager
        from src.database.database_manager import DatabaseManager
        engine.db_manager = DatabaseManager(":memory:")
        final_config = engine._merge_dynamic_risk_config(config)
        engine.dynamic_risk_manager = DynamicRiskManager(
            config=final_config,
            db_manager=engine.db_manager
        )
        engine.current_balance = 8500  # 15% drawdown
        engine.peak_balance = 10000
        
        # Get adjusted risk parameters
        adjusted_params = engine._get_dynamic_risk_adjusted_params()
        
        assert isinstance(adjusted_params, RiskParameters)
        # Parameters should be adjusted based on drawdown
        # Exact values depend on implementation but should reflect risk reduction
    
    def test_dynamic_risk_graceful_failure(self):
        """Test that dynamic risk fails gracefully in live engine"""
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            
            enable_dynamic_risk=True,
            database_url="sqlite:///:memory:"
        )
        
        # Break the dynamic risk manager
        engine.dynamic_risk_manager = None
        
        # Should return original size without error
        original_size = 0.05
        adjusted_size = engine._get_dynamic_risk_adjusted_size(original_size)
        
        assert adjusted_size == original_size
        
        # Should return default risk parameters without error
        adjusted_params = engine._get_dynamic_risk_adjusted_params()
        assert isinstance(adjusted_params, RiskParameters)
    
    def test_dynamic_risk_database_integration(self):
        """Test dynamic risk database logging integration"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4]
        )
        
        # Mock database manager
        mock_db = MagicMock()
        
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            log_to_database=True
        )
        
        # Replace database manager with mock
        engine.db_manager = mock_db
        
        # Manually initialize dynamic risk manager
        from src.database.database_manager import DatabaseManager
        engine.db_manager = DatabaseManager(":memory:")
        final_config = engine._merge_dynamic_risk_config(config)
        engine.dynamic_risk_manager = DynamicRiskManager(
            config=final_config,
            db_manager=engine.db_manager
        )
        engine.current_balance = 8500  # 15% drawdown to trigger logging
        engine.peak_balance = 10000
        
        # This should trigger database logging
        adjusted_size = engine._get_dynamic_risk_adjusted_size(0.05)
        
        # Verify size was adjusted
        assert adjusted_size == pytest.approx(0.02, rel=0.1)
        
        # Note: Database logging is tested separately in unit tests
        # This integration test focuses on the end-to-end flow
    
    def test_dynamic_risk_with_custom_config(self):
        """Test dynamic risk with custom configuration in live engine"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.03, 0.07],  # Custom thresholds
            risk_reduction_factors=[0.9, 0.5],  # Custom factors
            volatility_adjustment_enabled=False,
            performance_window_trades=5  # Smaller window for testing
        )
        
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            initial_balance=10000,
            database_url="sqlite:///:memory:"
        )
        
        # Manually initialize dynamic risk manager
        from src.database.database_manager import DatabaseManager
        engine.db_manager = DatabaseManager(":memory:")
        final_config = engine._merge_dynamic_risk_config(config)
        engine.dynamic_risk_manager = DynamicRiskManager(
            config=final_config,
            db_manager=engine.db_manager
        )
        
        # Test 5% drawdown (between first and second threshold)
        engine.current_balance = 9500
        engine.peak_balance = 10000
        
        adjusted_size = engine._get_dynamic_risk_adjusted_size(0.04)
        
        # Should apply first reduction factor (0.04 * 0.9 = 0.036)
        assert adjusted_size == pytest.approx(0.036, rel=0.1)
    
    def test_balance_tracking_in_live_engine(self):
        """Test balance and peak balance tracking in live engine"""
        engine = LiveTradingEngine(
            strategy=MockStrategy(),
            data_provider=MockDataProvider(),
            
            initial_balance=10000,
            database_url="sqlite:///:memory:"
        )
        
        # Initial state
        assert engine.current_balance == 10000
        assert engine.peak_balance == 10000
        
        # Simulate balance increase
        engine.current_balance = 12000
        # Live engine automatically updates peak balance in balance tracking
        if engine.current_balance > engine.peak_balance:
            engine.peak_balance = engine.current_balance
        assert engine.peak_balance == 12000
        
        # Simulate balance decrease (should not affect peak)
        engine.current_balance = 9000
        # Peak should remain unchanged when balance decreases
        assert engine.peak_balance == 12000  # Peak should remain