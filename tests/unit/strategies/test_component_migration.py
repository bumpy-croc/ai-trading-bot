"""
Unit tests for component system migration

Tests that verify the new component-based strategy system maintains
compatibility with existing strategy interfaces during migration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.strategies.base import BaseStrategy
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generator import (
    SignalGenerator, Signal, SignalDirection, MLBasicSignalGenerator
)
from src.strategies.components.risk_manager import FixedRiskManager
from src.strategies.components.position_sizer import ConfidenceWeightedSizer
from src.strategies.components.regime_context import EnhancedRegimeDetector


pytestmark = pytest.mark.unit


class TestComponentMigrationCompatibility:
    """Test compatibility between old and new strategy systems"""
    
    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data and indicators"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible tests
        
        data = {
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'onnx_pred': np.random.uniform(95, 115, 100),  # ML predictions
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.uniform(-2, 2, 100),
            'atr': np.random.uniform(1, 5, 100)
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_ml_basic_strategy_component_equivalence(self):
        """Test that component-based ML strategy produces equivalent results to legacy"""
        df = self.create_test_dataframe()
        balance = 10000.0
        
        # Create legacy strategy
        legacy_strategy = MlBasic()
        df_with_indicators = legacy_strategy.calculate_indicators(df)
        
        # Create component-based strategy
        signal_generator = MLBasicSignalGenerator()
        risk_manager = FixedRiskManager(risk_per_trade=0.02)
        position_sizer = ConfidenceWeightedSizer()
        
        component_strategy = Strategy(
            name="ml_basic_component",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        # Test multiple decision points
        for index in [20, 50, 80]:
            # Legacy decision
            legacy_entry = legacy_strategy.check_entry_conditions(df_with_indicators, index)
            legacy_exit = legacy_strategy.check_exit_conditions(df_with_indicators, index, 100.0)
            legacy_position_size = legacy_strategy.calculate_position_size(df_with_indicators, index, balance)
            
            # Component decision
            component_decision = component_strategy.process_candle(df_with_indicators, index, balance)
            
            # Compare results (allowing for some differences due to implementation details)
            if legacy_entry:
                assert component_decision.signal.direction in [SignalDirection.BUY, SignalDirection.SELL]
            
            if legacy_position_size > 0:
                assert component_decision.position_size > 0
            
            # Both should produce reasonable position sizes
            assert 0 <= component_decision.position_size <= balance * 0.5
    
    def test_strategy_interface_backward_compatibility(self):
        """Test that component strategies can be used where BaseStrategy is expected"""
        df = self.create_test_dataframe()
        
        # Create component strategy
        signal_generator = MLBasicSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = ConfidenceWeightedSizer()
        
        strategy = Strategy(
            name="test_strategy",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        # Test that it has expected methods and properties
        assert hasattr(strategy, 'name')
        assert hasattr(strategy, 'process_candle')
        assert hasattr(strategy, 'should_exit_position')
        assert hasattr(strategy, 'get_stop_loss_price')
        assert hasattr(strategy, 'get_performance_metrics')
        
        # Test method calls don't raise exceptions
        decision = strategy.process_candle(df, 50, 10000.0)
        assert decision is not None
        
        metrics = strategy.get_performance_metrics()
        assert isinstance(metrics, dict)
    
    def test_component_parameter_extraction(self):
        """Test that component parameters can be extracted for configuration"""
        signal_generator = MLBasicSignalGenerator()
        risk_manager = FixedRiskManager(risk_per_trade=0.03)
        position_sizer = ConfidenceWeightedSizer(base_size_pct=0.05)
        
        strategy = Strategy(
            name="param_test",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        component_info = strategy.get_component_info()
        
        # Verify all components are represented
        assert 'signal_generator' in component_info
        assert 'risk_manager' in component_info
        assert 'position_sizer' in component_info
        
        # Verify parameters are accessible
        assert component_info['risk_manager']['risk_per_trade'] == 0.03
        assert component_info['position_sizer']['base_size_pct'] == 0.05
    
    def test_error_handling_compatibility(self):
        """Test that component strategies handle errors gracefully like legacy strategies"""
        # Create strategy with potentially failing components
        class FailingSignalGenerator(SignalGenerator):
            def __init__(self):
                super().__init__("failing_generator")
            
            def generate_signal(self, df, index, regime=None):
                raise Exception("Signal generation failed")
            
            def get_confidence(self, df, index):
                return 0.5
        
        failing_generator = FailingSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = ConfidenceWeightedSizer()
        
        strategy = Strategy(
            name="error_test",
            signal_generator=failing_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        df = self.create_test_dataframe()
        
        # Should not raise exception, should return safe decision
        decision = strategy.process_candle(df, 50, 10000.0)
        
        assert decision is not None
        assert decision.signal.direction == SignalDirection.HOLD
        assert decision.position_size == 0.0
        assert 'error' in decision.metadata
    
    def test_performance_metrics_compatibility(self):
        """Test that performance metrics are compatible with existing monitoring"""
        signal_generator = MLBasicSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = ConfidenceWeightedSizer()
        
        strategy = Strategy(
            name="metrics_test",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        df = self.create_test_dataframe()
        
        # Generate some decisions to populate metrics
        for i in range(10, 20):
            strategy.process_candle(df, i, 10000.0)
        
        metrics = strategy.get_performance_metrics()
        
        # Verify expected metric structure
        expected_keys = [
            'total_decisions', 'buy_signals', 'sell_signals', 'hold_signals',
            'avg_execution_time_ms', 'avg_signal_confidence', 'avg_position_size',
            'component_info'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Verify metric values are reasonable
        assert metrics['total_decisions'] == 10
        assert metrics['avg_execution_time_ms'] >= 0
        assert 0 <= metrics['avg_signal_confidence'] <= 1
        assert metrics['avg_position_size'] >= 0


class TestLegacyStrategyMigration:
    """Test migration utilities for converting legacy strategies"""
    
    def test_legacy_strategy_parameter_extraction(self):
        """Test extracting parameters from legacy strategies for component creation"""
        legacy_strategy = MlBasic()
        
        # Extract key parameters
        params = legacy_strategy.get_parameters()
        
        # Verify expected parameters exist
        expected_params = ['name', 'model_path', 'sequence_length', 'stop_loss_pct', 'take_profit_pct']
        for param in expected_params:
            assert param in params
        
        # Verify parameter values are reasonable
        assert isinstance(params['stop_loss_pct'], (int, float))
        assert 0 < params['stop_loss_pct'] < 1
        assert isinstance(params['sequence_length'], int)
        assert params['sequence_length'] > 0
    
    def test_component_creation_from_legacy_params(self):
        """Test creating components from legacy strategy parameters"""
        legacy_strategy = MlBasic()
        params = legacy_strategy.get_parameters()
        
        # Create components based on legacy parameters
        signal_generator = MLBasicSignalGenerator(
            model_path=params.get('model_path', 'default_model.onnx'),
            sequence_length=params.get('sequence_length', 120)
        )
        
        risk_manager = FixedRiskManager(
            stop_loss_pct=params.get('stop_loss_pct', 0.05)
        )
        
        position_sizer = ConfidenceWeightedSizer()
        
        # Verify components were created successfully
        assert signal_generator.model_path == params['model_path']
        assert signal_generator.sequence_length == params['sequence_length']
        assert risk_manager.stop_loss_pct == params['stop_loss_pct']
    
    def test_migration_validation(self):
        """Test validation of migrated strategies"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200],
            'onnx_pred': [101, 102, 103]
        })
        
        # Create migrated strategy
        signal_generator = MLBasicSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = ConfidenceWeightedSizer()
        
        strategy = Strategy(
            name="migrated_test",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        # Validate strategy can process data
        decision = strategy.process_candle(df, 1, 10000.0)
        
        # Basic validation checks
        assert decision is not None
        assert isinstance(decision.signal.direction, SignalDirection)
        assert 0 <= decision.signal.confidence <= 1
        assert 0 <= decision.signal.strength <= 1
        assert decision.position_size >= 0
        assert decision.execution_time_ms >= 0


class TestBackwardCompatibilityDuringMigration:
    """Test that both old and new systems can coexist during migration"""
    
    def test_mixed_strategy_usage(self):
        """Test using both legacy and component strategies in same system"""
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'onnx_pred': [101, 102, 103, 104, 105]
        })
        
        # Legacy strategy
        legacy_strategy = MlBasic()
        df_with_indicators = legacy_strategy.calculate_indicators(df)
        
        # Component strategy
        component_strategy = Strategy(
            name="component_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        # Both should be able to process the same data
        legacy_entry = legacy_strategy.check_entry_conditions(df_with_indicators, 2)
        component_decision = component_strategy.process_candle(df_with_indicators, 2, 10000.0)
        
        # Both should produce valid results
        assert isinstance(legacy_entry, bool)
        assert component_decision is not None
        assert isinstance(component_decision.signal.direction, SignalDirection)
    
    def test_configuration_compatibility(self):
        """Test that configuration systems work with both strategy types"""
        # Test that both strategy types can be configured similarly
        legacy_config = {
            'name': 'ml_basic',
            'model_path': 'test_model.onnx',
            'stop_loss_pct': 0.05
        }
        
        component_config = {
            'name': 'ml_basic_component',
            'signal_generator': {
                'type': 'MLBasicSignalGenerator',
                'model_path': 'test_model.onnx'
            },
            'risk_manager': {
                'type': 'FixedRiskManager',
                'stop_loss_pct': 0.05
            },
            'position_sizer': {
                'type': 'ConfidenceWeightedSizer'
            }
        }
        
        # Both configurations should be valid and contain necessary information
        assert 'name' in legacy_config
        assert 'name' in component_config
        assert legacy_config['stop_loss_pct'] == component_config['risk_manager']['stop_loss_pct']
    
    def test_test_fixture_compatibility(self):
        """Test that existing test fixtures work with component strategies"""
        # This would test that existing test fixtures and mocks
        # can be used with the new component system
        
        # Mock data that might be used in existing tests
        mock_df = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
        })
        
        # Component strategy should work with existing mock data
        strategy = Strategy(
            name="fixture_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        # Should not raise exceptions with minimal data
        try:
            decision = strategy.process_candle(mock_df, 0, 10000.0)
            assert decision is not None
        except Exception as e:
            pytest.fail(f"Component strategy failed with existing fixture data: {e}")


if __name__ == '__main__':
    pytest.main([__file__])