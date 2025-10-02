"""
Tests for AdapterFactory and related utilities

This module tests the AdapterFactory class and utility functions for creating
and managing LegacyStrategyAdapter instances.
"""

from unittest.mock import Mock

import pytest

from src.strategies.adapters.adapter_factory import (
    AdapterFactory,
    AdapterValidationUtils,
    MigrationHelper,
    adapter_factory,
)
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.base import BaseStrategy
from src.strategies.components.position_sizer import (
    ConfidenceWeightedSizer,
    FixedFractionSizer,
)
from src.strategies.components.risk_manager import FixedRiskManager
from src.strategies.components.signal_generator import (
    HoldSignalGenerator,
    RandomSignalGenerator,
)


class TestAdapterFactory:
    """Test cases for AdapterFactory"""
    
    @pytest.fixture
    def factory(self):
        """Create AdapterFactory instance for testing"""
        return AdapterFactory()
    
    def test_factory_initialization(self, factory):
        """Test factory initialization"""
        assert isinstance(factory, AdapterFactory)
        
        # Check component registries are populated
        components = factory.get_available_components()
        assert 'signal_generators' in components
        assert 'risk_managers' in components
        assert 'position_sizers' in components
        assert 'templates' in components
        
        # Check default components are registered
        assert 'hold' in components['signal_generators']
        assert 'random' in components['signal_generators']
        assert 'fixed' in components['risk_managers']
        assert 'fixed_fraction' in components['position_sizers']
        assert 'confidence_weighted' in components['position_sizers']
    
    def test_create_from_template_conservative(self, factory):
        """Test creating adapter from conservative template"""
        adapter = factory.create_from_template('conservative', name='test_conservative')
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.name == 'test_conservative'
        assert isinstance(adapter.signal_generator, HoldSignalGenerator)
        assert isinstance(adapter.risk_manager, FixedRiskManager)
        assert isinstance(adapter.position_sizer, FixedFractionSizer)
    
    def test_create_from_template_moderate(self, factory):
        """Test creating adapter from moderate template"""
        adapter = factory.create_from_template('moderate')
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.name == 'template_moderate'
        assert isinstance(adapter.signal_generator, RandomSignalGenerator)
        assert isinstance(adapter.risk_manager, FixedRiskManager)
        assert isinstance(adapter.position_sizer, FixedFractionSizer)
    
    def test_create_from_template_aggressive(self, factory):
        """Test creating adapter from aggressive template"""
        adapter = factory.create_from_template('aggressive')
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert isinstance(adapter.signal_generator, RandomSignalGenerator)
        assert isinstance(adapter.risk_manager, FixedRiskManager)
        assert isinstance(adapter.position_sizer, ConfidenceWeightedSizer)
    
    def test_create_from_template_invalid(self, factory):
        """Test creating adapter from invalid template"""
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            factory.create_from_template('invalid')
    
    def test_create_from_config_valid(self, factory):
        """Test creating adapter from valid configuration"""
        config = {
            'signal_generator': {'type': 'hold', 'params': {}},
            'risk_manager': {'type': 'fixed', 'params': {'risk_per_trade': 0.03}},
            'position_sizer': {'type': 'fixed_fraction', 'params': {'fraction': 0.025}},
            'regime_detector': {'params': {'stability_threshold': 15}}
        }
        
        adapter = factory.create_from_config(config, name='config_test')
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.name == 'config_test'
        assert isinstance(adapter.signal_generator, HoldSignalGenerator)
        assert isinstance(adapter.risk_manager, FixedRiskManager)
        assert adapter.risk_manager.risk_per_trade == 0.03
        assert isinstance(adapter.position_sizer, FixedFractionSizer)
        assert adapter.position_sizer.fraction == 0.025
    
    def test_create_from_config_missing_required_key(self, factory):
        """Test creating adapter from config missing required keys"""
        config = {
            'signal_generator': {'type': 'hold', 'params': {}},
            # Missing risk_manager and position_sizer
        }
        
        with pytest.raises(ValueError, match="Missing required config key"):
            factory.create_from_config(config)
    
    def test_create_from_config_missing_type(self, factory):
        """Test creating adapter from config missing component type"""
        config = {
            'signal_generator': {'params': {}},  # Missing 'type'
            'risk_manager': {'type': 'fixed', 'params': {}},
            'position_sizer': {'type': 'fixed_fraction', 'params': {}}
        }
        
        with pytest.raises(ValueError, match="Missing 'type' in signal_generator config"):
            factory.create_from_config(config)
    
    def test_create_from_config_invalid_component_type(self, factory):
        """Test creating adapter with invalid component type"""
        config = {
            'signal_generator': {'type': 'invalid_type', 'params': {}},
            'risk_manager': {'type': 'fixed', 'params': {}},
            'position_sizer': {'type': 'fixed_fraction', 'params': {}}
        }
        
        with pytest.raises(ValueError, match="Signal generator 'invalid_type' not found"):
            factory.create_from_config(config)
    
    def test_create_from_components(self, factory):
        """Test creating adapter from component instances"""
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = FixedFractionSizer()
        
        adapter = factory.create_from_components(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            name='component_test'
        )
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.name == 'component_test'
        assert adapter.signal_generator == signal_generator
        assert adapter.risk_manager == risk_manager
        assert adapter.position_sizer == position_sizer
    
    def test_convert_legacy_strategy(self, factory):
        """Test converting legacy strategy to adapter"""
        # Create mock legacy strategy
        legacy_strategy = Mock(spec=BaseStrategy)
        legacy_strategy.name = "test_legacy"
        legacy_strategy.get_trading_pair.return_value = "BTCUSDT"
        legacy_strategy.get_risk_overrides.return_value = None
        
        adapter = factory.convert_legacy_strategy(legacy_strategy)
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.name == "converted_test_legacy"
        assert adapter.get_trading_pair() == "BTCUSDT"
    
    def test_convert_legacy_strategy_with_risk_overrides(self, factory):
        """Test converting legacy strategy with risk overrides"""
        legacy_strategy = Mock(spec=BaseStrategy)
        legacy_strategy.name = "test_legacy_with_overrides"
        legacy_strategy.get_trading_pair.return_value = "ETHUSDT"
        legacy_strategy.get_risk_overrides.return_value = {
            'position_sizer': 'fixed_fraction',
            'base_fraction': 0.03,
            'stop_loss_pct': 0.04
        }
        
        conversion_config = {
            'risk_manager': {'params': {'risk_per_trade': 0.025}}
        }
        
        adapter = factory.convert_legacy_strategy(legacy_strategy, conversion_config)
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.get_trading_pair() == "ETHUSDT"
    
    def test_register_custom_components(self, factory):
        """Test registering custom component types"""
        # Create mock custom components
        CustomSignalGenerator = Mock()
        CustomRiskManager = Mock()
        CustomPositionSizer = Mock()
        
        # Register custom components
        factory.register_signal_generator('custom_signal', CustomSignalGenerator)
        factory.register_risk_manager('custom_risk', CustomRiskManager)
        factory.register_position_sizer('custom_sizer', CustomPositionSizer)
        
        # Check they are available
        components = factory.get_available_components()
        assert 'custom_signal' in components['signal_generators']
        assert 'custom_risk' in components['risk_managers']
        assert 'custom_sizer' in components['position_sizers']
    
    def test_component_creation_with_invalid_params(self, factory):
        """Test component creation with invalid parameters"""
        config = {
            'signal_generator': {'type': 'random', 'params': {'buy_prob': 1.5}},  # Invalid prob > 1
            'risk_manager': {'type': 'fixed', 'params': {}},
            'position_sizer': {'type': 'fixed_fraction', 'params': {}}
        }
        
        with pytest.raises(ValueError, match="Failed to create signal generator"):
            factory.create_from_config(config)


class TestAdapterValidationUtils:
    """Test cases for AdapterValidationUtils"""
    
    @pytest.fixture
    def valid_adapter(self):
        """Create valid adapter for testing"""
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = FixedFractionSizer()
        
        return LegacyStrategyAdapter(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            name='valid_adapter'
        )
    
    def test_validate_adapter_compatibility_valid(self, valid_adapter):
        """Test validation of valid adapter"""
        results = AdapterValidationUtils.validate_adapter_compatibility(valid_adapter)
        
        assert isinstance(results, dict)
        assert results['has_calculate_indicators'] is True
        assert results['has_check_entry_conditions'] is True
        assert results['has_check_exit_conditions'] is True
        assert results['has_calculate_position_size'] is True
        assert results['has_calculate_stop_loss'] is True
        assert results['has_get_parameters'] is True
        assert results['is_base_strategy'] is True
        assert results['components_valid'] is True
    
    def test_validate_adapter_compatibility_invalid(self):
        """Test validation of invalid adapter"""
        # Create adapter with missing components
        invalid_adapter = Mock(spec=LegacyStrategyAdapter)
        invalid_adapter.signal_generator = None
        
        results = AdapterValidationUtils.validate_adapter_compatibility(invalid_adapter)
        
        assert results['components_valid'] is False
    
    def test_test_adapter_methods_success(self, valid_adapter):
        """Test adapter methods testing with valid adapter"""
        results = AdapterValidationUtils.test_adapter_methods(valid_adapter)
        
        assert isinstance(results, dict)
        assert results['calculate_indicators'] is True
        assert results['check_entry_conditions'] is True
        assert results['check_exit_conditions'] is True
        assert results['calculate_position_size'] is True
        assert results['calculate_stop_loss'] is True
        assert results['get_parameters'] is True
    
    def test_test_adapter_methods_with_custom_data(self, valid_adapter):
        """Test adapter methods testing with custom test data"""
        import pandas as pd
        
        custom_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        results = AdapterValidationUtils.test_adapter_methods(valid_adapter, custom_data)
        
        assert isinstance(results, dict)
        assert results['calculate_indicators'] is True
    
    def test_test_adapter_methods_with_errors(self):
        """Test adapter methods testing with adapter that raises errors"""
        # Create adapter that raises errors
        error_adapter = Mock(spec=LegacyStrategyAdapter)
        error_adapter.calculate_indicators.side_effect = Exception("Test error")
        error_adapter.check_entry_conditions.side_effect = Exception("Test error")
        error_adapter.check_exit_conditions.side_effect = Exception("Test error")
        error_adapter.calculate_position_size.side_effect = Exception("Test error")
        error_adapter.calculate_stop_loss.side_effect = Exception("Test error")
        error_adapter.get_parameters.side_effect = Exception("Test error")
        
        results = AdapterValidationUtils.test_adapter_methods(error_adapter)
        
        assert results['calculate_indicators'] is False
        assert 'calculate_indicators_error' in results
        assert results['check_entry_conditions'] is False
        assert results['check_exit_conditions'] is False
        assert results['calculate_position_size'] is False
        assert results['calculate_stop_loss'] is False
        assert results['get_parameters'] is False


class TestMigrationHelper:
    """Test cases for MigrationHelper"""
    
    @pytest.fixture
    def mock_legacy_strategy(self):
        """Create mock legacy strategy for testing"""
        strategy = Mock(spec=BaseStrategy)
        strategy.name = "test_strategy"
        strategy.__class__.__name__ = "TestStrategy"
        strategy.get_trading_pair.return_value = "BTCUSDT"
        strategy.get_risk_overrides.return_value = None
        return strategy
    
    @pytest.fixture
    def mock_legacy_strategy_with_overrides(self):
        """Create mock legacy strategy with risk overrides"""
        strategy = Mock(spec=BaseStrategy)
        strategy.name = "test_strategy_overrides"
        strategy.__class__.__name__ = "TestStrategyWithOverrides"
        strategy.get_trading_pair.return_value = "ETHUSDT"
        strategy.get_risk_overrides.return_value = {
            'position_sizer': 'confidence_weighted',
            'base_fraction': 0.04,
            'stop_loss_pct': 0.06
        }
        return strategy
    
    def test_analyze_legacy_strategy_no_overrides(self, mock_legacy_strategy):
        """Test analyzing legacy strategy without risk overrides"""
        analysis = MigrationHelper.analyze_legacy_strategy(mock_legacy_strategy)
        
        assert isinstance(analysis, dict)
        assert analysis['strategy_name'] == "test_strategy"
        assert analysis['strategy_class'] == "TestStrategy"
        assert analysis['trading_pair'] == "BTCUSDT"
        assert analysis['has_risk_overrides'] is False
        
        # Should have default suggested components
        assert 'suggested_components' in analysis
        assert 'signal_generator' in analysis['suggested_components']
        assert 'risk_manager' in analysis['suggested_components']
        assert 'position_sizer' in analysis['suggested_components']
    
    def test_analyze_legacy_strategy_with_overrides(self, mock_legacy_strategy_with_overrides):
        """Test analyzing legacy strategy with risk overrides"""
        analysis = MigrationHelper.analyze_legacy_strategy(mock_legacy_strategy_with_overrides)
        
        assert analysis['strategy_name'] == "test_strategy_overrides"
        assert analysis['has_risk_overrides'] is True
        assert 'risk_overrides' in analysis
        
        # Should suggest components based on overrides
        suggested = analysis['suggested_components']
        assert suggested['position_sizer']['type'] == 'confidence_weighted'
        assert suggested['position_sizer']['params']['base_fraction'] == 0.04
        assert suggested['risk_manager']['params']['stop_loss_pct'] == 0.06
    
    def test_create_migration_plan_single_strategy(self, mock_legacy_strategy):
        """Test creating migration plan for single strategy"""
        strategies = [mock_legacy_strategy]
        plan = MigrationHelper.create_migration_plan(strategies)
        
        assert isinstance(plan, dict)
        assert plan['total_strategies'] == 1
        assert len(plan['strategy_analyses']) == 1
        assert 'migration_steps' in plan
        assert plan['estimated_effort'] == 'low'  # No complex strategies
    
    def test_create_migration_plan_multiple_strategies(self, mock_legacy_strategy, 
                                                     mock_legacy_strategy_with_overrides):
        """Test creating migration plan for multiple strategies"""
        strategies = [mock_legacy_strategy, mock_legacy_strategy_with_overrides]
        plan = MigrationHelper.create_migration_plan(strategies)
        
        assert plan['total_strategies'] == 2
        assert len(plan['strategy_analyses']) == 2
        assert plan['estimated_effort'] in ['low', 'medium', 'high']
    
    def test_create_migration_plan_high_complexity(self, mock_legacy_strategy_with_overrides):
        """Test migration plan with high complexity strategies"""
        # Create multiple strategies with overrides (high complexity)
        strategies = [mock_legacy_strategy_with_overrides] * 5
        plan = MigrationHelper.create_migration_plan(strategies)
        
        assert plan['total_strategies'] == 5
        # Should be high effort since all strategies have overrides
        assert plan['estimated_effort'] == 'high'


class TestGlobalFactoryInstance:
    """Test the global adapter_factory instance"""
    
    def test_global_factory_exists(self):
        """Test that global factory instance exists and is usable"""
        assert isinstance(adapter_factory, AdapterFactory)
        
        # Test basic functionality
        components = adapter_factory.get_available_components()
        assert 'signal_generators' in components
        assert 'templates' in components
    
    def test_global_factory_create_template(self):
        """Test creating adapter using global factory"""
        adapter = adapter_factory.create_from_template('conservative')
        
        assert isinstance(adapter, LegacyStrategyAdapter)
        assert adapter.name == 'template_conservative'


class TestFactoryErrorHandling:
    """Test error handling in factory methods"""
    
    @pytest.fixture
    def factory(self):
        """Create AdapterFactory instance for testing"""
        return AdapterFactory()
    
    def test_create_signal_generator_invalid_params(self, factory):
        """Test creating signal generator with invalid parameters"""
        with pytest.raises(ValueError, match="Failed to create signal generator"):
            factory._create_signal_generator('random', {'buy_prob': 2.0})  # Invalid probability
    
    def test_create_risk_manager_invalid_params(self, factory):
        """Test creating risk manager with invalid parameters"""
        with pytest.raises(ValueError, match="Failed to create risk manager"):
            factory._create_risk_manager('fixed', {'risk_per_trade': -0.1})  # Negative risk
    
    def test_create_position_sizer_invalid_params(self, factory):
        """Test creating position sizer with invalid parameters"""
        with pytest.raises(ValueError, match="Failed to create position sizer"):
            factory._create_position_sizer('fixed_fraction', {'fraction': 2.0})  # Invalid fraction
    
    def test_config_validation_comprehensive(self, factory):
        """Test comprehensive config validation"""
        # Test missing signal_generator type
        config = {
            'signal_generator': {'params': {}},
            'risk_manager': {'type': 'fixed', 'params': {}},
            'position_sizer': {'type': 'fixed_fraction', 'params': {}}
        }
        
        with pytest.raises(ValueError, match="Missing 'type' in signal_generator config"):
            factory._validate_config(config)
        
        # Test missing risk_manager entirely
        config = {
            'signal_generator': {'type': 'hold', 'params': {}},
            'position_sizer': {'type': 'fixed_fraction', 'params': {}}
        }
        
        with pytest.raises(ValueError, match="Missing required config key: risk_manager"):
            factory._validate_config(config)