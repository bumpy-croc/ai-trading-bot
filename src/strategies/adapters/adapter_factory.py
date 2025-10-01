"""
Adapter Factory and Utilities

This module provides factory methods and utilities for creating LegacyStrategyAdapter
instances from existing strategies and component configurations.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from src.strategies.base import BaseStrategy
from src.strategies.components.signal_generator import (
    SignalGenerator, HoldSignalGenerator, RandomSignalGenerator
)
from src.strategies.components.risk_manager import RiskManager, FixedRiskManager
from src.strategies.components.position_sizer import (
    PositionSizer, FixedFractionSizer, ConfidenceWeightedSizer
)
from src.strategies.components.regime_context import EnhancedRegimeDetector
from .legacy_adapter import LegacyStrategyAdapter


class AdapterFactory:
    """
    Factory class for creating LegacyStrategyAdapter instances
    
    Provides methods to create adapters from existing strategies, component
    configurations, and predefined templates.
    """
    
    def __init__(self):
        """Initialize the adapter factory"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Registry of available components
        self._signal_generators: Dict[str, Type[SignalGenerator]] = {
            'hold': HoldSignalGenerator,
            'random': RandomSignalGenerator,
        }
        
        self._risk_managers: Dict[str, Type[RiskManager]] = {
            'fixed': FixedRiskManager,
        }
        
        self._position_sizers: Dict[str, Type[PositionSizer]] = {
            'fixed_fraction': FixedFractionSizer,
            'confidence_weighted': ConfidenceWeightedSizer,
        }
        
        # Predefined strategy templates
        self._templates = {
            'conservative': {
                'signal_generator': {'type': 'hold', 'params': {}},
                'risk_manager': {'type': 'fixed', 'params': {'risk_per_trade': 0.01, 'stop_loss_pct': 0.03}},
                'position_sizer': {'type': 'fixed_fraction', 'params': {'fraction': 0.01}}
            },
            'moderate': {
                'signal_generator': {'type': 'random', 'params': {'buy_prob': 0.3, 'sell_prob': 0.3}},
                'risk_manager': {'type': 'fixed', 'params': {'risk_per_trade': 0.02, 'stop_loss_pct': 0.05}},
                'position_sizer': {'type': 'fixed_fraction', 'params': {'fraction': 0.02}}
            },
            'aggressive': {
                'signal_generator': {'type': 'random', 'params': {'buy_prob': 0.4, 'sell_prob': 0.4}},
                'risk_manager': {'type': 'fixed', 'params': {'risk_per_trade': 0.03, 'stop_loss_pct': 0.07}},
                'position_sizer': {'type': 'confidence_weighted', 'params': {'base_fraction': 0.05}}
            }
        }
    
    def create_from_template(self, template_name: str, name: Optional[str] = None) -> LegacyStrategyAdapter:
        """
        Create adapter from predefined template
        
        Args:
            template_name: Name of the template to use
            name: Optional custom name for the adapter
            
        Returns:
            LegacyStrategyAdapter instance
            
        Raises:
            ValueError: If template name is not found
        """
        if template_name not in self._templates:
            available = list(self._templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        template = self._templates[template_name]
        
        # Create components from template
        signal_generator = self._create_signal_generator(
            template['signal_generator']['type'],
            template['signal_generator']['params']
        )
        
        risk_manager = self._create_risk_manager(
            template['risk_manager']['type'],
            template['risk_manager']['params']
        )
        
        position_sizer = self._create_position_sizer(
            template['position_sizer']['type'],
            template['position_sizer']['params']
        )
        
        regime_detector = EnhancedRegimeDetector()
        
        # Create adapter
        adapter_name = name or f"template_{template_name}"
        adapter = LegacyStrategyAdapter(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=adapter_name
        )
        
        self.logger.info(f"Created adapter from template '{template_name}': {adapter}")
        return adapter
    
    def create_from_config(self, config: Dict[str, Any], name: Optional[str] = None) -> LegacyStrategyAdapter:
        """
        Create adapter from configuration dictionary
        
        Args:
            config: Configuration dictionary with component specifications
            name: Optional custom name for the adapter
            
        Returns:
            LegacyStrategyAdapter instance
            
        Example config:
            {
                'signal_generator': {'type': 'random', 'params': {'buy_prob': 0.3}},
                'risk_manager': {'type': 'fixed', 'params': {'risk_per_trade': 0.02}},
                'position_sizer': {'type': 'fixed_fraction', 'params': {'fraction': 0.02}},
                'regime_detector': {'params': {'stability_threshold': 15}}
            }
        """
        self._validate_config(config)
        
        # Create signal generator
        sg_config = config['signal_generator']
        signal_generator = self._create_signal_generator(sg_config['type'], sg_config.get('params', {}))
        
        # Create risk manager
        rm_config = config['risk_manager']
        risk_manager = self._create_risk_manager(rm_config['type'], rm_config.get('params', {}))
        
        # Create position sizer
        ps_config = config['position_sizer']
        position_sizer = self._create_position_sizer(ps_config['type'], ps_config.get('params', {}))
        
        # Create regime detector
        rd_params = config.get('regime_detector', {}).get('params', {})
        regime_detector = EnhancedRegimeDetector(**rd_params)
        
        # Create adapter
        adapter_name = name or "config_adapter"
        adapter = LegacyStrategyAdapter(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=adapter_name
        )
        
        self.logger.info(f"Created adapter from config: {adapter}")
        return adapter
    
    def create_from_components(self,
                             signal_generator: SignalGenerator,
                             risk_manager: RiskManager,
                             position_sizer: PositionSizer,
                             regime_detector: Optional[EnhancedRegimeDetector] = None,
                             name: Optional[str] = None) -> LegacyStrategyAdapter:
        """
        Create adapter from component instances
        
        Args:
            signal_generator: Signal generator component
            risk_manager: Risk manager component
            position_sizer: Position sizer component
            regime_detector: Optional regime detector
            name: Optional custom name for the adapter
            
        Returns:
            LegacyStrategyAdapter instance
        """
        regime_detector = regime_detector or EnhancedRegimeDetector()
        
        adapter = LegacyStrategyAdapter(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=name
        )
        
        self.logger.info(f"Created adapter from components: {adapter}")
        return adapter
    
    def convert_legacy_strategy(self, legacy_strategy: BaseStrategy, 
                              conversion_config: Optional[Dict[str, Any]] = None) -> LegacyStrategyAdapter:
        """
        Convert existing legacy strategy to component-based adapter
        
        Args:
            legacy_strategy: Existing BaseStrategy instance
            conversion_config: Optional configuration for conversion process
            
        Returns:
            LegacyStrategyAdapter instance
            
        Note:
            This is a placeholder implementation. Full conversion would require
            analyzing the legacy strategy's logic and extracting components.
        """
        self.logger.warning(f"Converting legacy strategy '{legacy_strategy.name}' using default components")
        
        # For now, create a conservative adapter as placeholder
        # In a full implementation, this would analyze the legacy strategy
        # and extract signal generation, risk management, and position sizing logic
        
        config = conversion_config or {}
        
        # Use moderate template as default for legacy conversion
        template_config = self._templates['moderate'].copy()
        
        # Override with any provided conversion config
        if 'signal_generator' in config:
            template_config['signal_generator'].update(config['signal_generator'])
        if 'risk_manager' in config:
            template_config['risk_manager'].update(config['risk_manager'])
        if 'position_sizer' in config:
            template_config['position_sizer'].update(config['position_sizer'])
        
        # Create adapter with legacy strategy's name
        adapter = self.create_from_config(template_config, name=f"converted_{legacy_strategy.name}")
        
        # Copy trading pair and other settings
        adapter.set_trading_pair(legacy_strategy.get_trading_pair())
        
        self.logger.info(f"Converted legacy strategy '{legacy_strategy.name}' to adapter")
        return adapter
    
    def register_signal_generator(self, name: str, generator_class: Type[SignalGenerator]) -> None:
        """Register a new signal generator type"""
        self._signal_generators[name] = generator_class
        self.logger.info(f"Registered signal generator: {name}")
    
    def register_risk_manager(self, name: str, manager_class: Type[RiskManager]) -> None:
        """Register a new risk manager type"""
        self._risk_managers[name] = manager_class
        self.logger.info(f"Registered risk manager: {name}")
    
    def register_position_sizer(self, name: str, sizer_class: Type[PositionSizer]) -> None:
        """Register a new position sizer type"""
        self._position_sizers[name] = sizer_class
        self.logger.info(f"Registered position sizer: {name}")
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get list of available component types"""
        return {
            'signal_generators': list(self._signal_generators.keys()),
            'risk_managers': list(self._risk_managers.keys()),
            'position_sizers': list(self._position_sizers.keys()),
            'templates': list(self._templates.keys())
        }
    
    def _create_signal_generator(self, generator_type: str, params: Dict[str, Any]) -> SignalGenerator:
        """Create signal generator instance"""
        if generator_type not in self._signal_generators:
            available = list(self._signal_generators.keys())
            raise ValueError(f"Signal generator '{generator_type}' not found. Available: {available}")
        
        generator_class = self._signal_generators[generator_type]
        
        try:
            return generator_class(**params)
        except Exception as e:
            self.logger.error(f"Error creating signal generator '{generator_type}': {e}")
            raise ValueError(f"Failed to create signal generator '{generator_type}': {e}")
    
    def _create_risk_manager(self, manager_type: str, params: Dict[str, Any]) -> RiskManager:
        """Create risk manager instance"""
        if manager_type not in self._risk_managers:
            available = list(self._risk_managers.keys())
            raise ValueError(f"Risk manager '{manager_type}' not found. Available: {available}")
        
        manager_class = self._risk_managers[manager_type]
        
        try:
            return manager_class(**params)
        except Exception as e:
            self.logger.error(f"Error creating risk manager '{manager_type}': {e}")
            raise ValueError(f"Failed to create risk manager '{manager_type}': {e}")
    
    def _create_position_sizer(self, sizer_type: str, params: Dict[str, Any]) -> PositionSizer:
        """Create position sizer instance"""
        if sizer_type not in self._position_sizers:
            available = list(self._position_sizers.keys())
            raise ValueError(f"Position sizer '{sizer_type}' not found. Available: {available}")
        
        sizer_class = self._position_sizers[sizer_type]
        
        try:
            return sizer_class(**params)
        except Exception as e:
            self.logger.error(f"Error creating position sizer '{sizer_type}': {e}")
            raise ValueError(f"Failed to create position sizer '{sizer_type}': {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration dictionary"""
        required_keys = ['signal_generator', 'risk_manager', 'position_sizer']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            
            if 'type' not in config[key]:
                raise ValueError(f"Missing 'type' in {key} config")


class AdapterValidationUtils:
    """
    Utilities for validating and testing adapter functionality
    """
    
    @staticmethod
    def validate_adapter_compatibility(adapter: LegacyStrategyAdapter) -> Dict[str, bool]:
        """
        Validate that adapter properly implements BaseStrategy interface
        
        Args:
            adapter: LegacyStrategyAdapter to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'has_calculate_indicators': hasattr(adapter, 'calculate_indicators'),
            'has_check_entry_conditions': hasattr(adapter, 'check_entry_conditions'),
            'has_check_exit_conditions': hasattr(adapter, 'check_exit_conditions'),
            'has_calculate_position_size': hasattr(adapter, 'calculate_position_size'),
            'has_calculate_stop_loss': hasattr(adapter, 'calculate_stop_loss'),
            'has_get_parameters': hasattr(adapter, 'get_parameters'),
            'is_base_strategy': isinstance(adapter, BaseStrategy),
            'components_valid': True
        }
        
        # Validate components exist
        try:
            assert adapter.signal_generator is not None
            assert adapter.risk_manager is not None
            assert adapter.position_sizer is not None
            assert adapter.regime_detector is not None
        except (AttributeError, AssertionError):
            results['components_valid'] = False
        
        return results
    
    @staticmethod
    def test_adapter_methods(adapter: LegacyStrategyAdapter, test_data: Optional[Any] = None) -> Dict[str, bool]:
        """
        Test adapter methods with sample data
        
        Args:
            adapter: LegacyStrategyAdapter to test
            test_data: Optional test DataFrame (creates sample if None)
            
        Returns:
            Dictionary with test results
        """
        import pandas as pd
        import numpy as np
        
        # Create sample test data if not provided
        if test_data is None:
            dates = pd.date_range('2024-01-01', periods=100, freq='1H')
            test_data = pd.DataFrame({
                'open': np.random.uniform(100, 110, 100),
                'high': np.random.uniform(110, 120, 100),
                'low': np.random.uniform(90, 100, 100),
                'close': np.random.uniform(100, 110, 100),
                'volume': np.random.uniform(1000, 10000, 100)
            }, index=dates)
        
        results = {}
        
        # Test calculate_indicators
        try:
            df_with_indicators = adapter.calculate_indicators(test_data)
            results['calculate_indicators'] = isinstance(df_with_indicators, pd.DataFrame)
        except Exception as e:
            results['calculate_indicators'] = False
            results['calculate_indicators_error'] = str(e)
        
        # Test check_entry_conditions
        try:
            if 'calculate_indicators' in results and results['calculate_indicators']:
                entry_result = adapter.check_entry_conditions(df_with_indicators, 50)
                results['check_entry_conditions'] = isinstance(entry_result, bool)
            else:
                results['check_entry_conditions'] = False
        except Exception as e:
            results['check_entry_conditions'] = False
            results['check_entry_conditions_error'] = str(e)
        
        # Test check_exit_conditions
        try:
            if 'calculate_indicators' in results and results['calculate_indicators']:
                exit_result = adapter.check_exit_conditions(df_with_indicators, 50, 105.0)
                results['check_exit_conditions'] = isinstance(exit_result, bool)
            else:
                results['check_exit_conditions'] = False
        except Exception as e:
            results['check_exit_conditions'] = False
            results['check_exit_conditions_error'] = str(e)
        
        # Test calculate_position_size
        try:
            if 'calculate_indicators' in results and results['calculate_indicators']:
                position_size = adapter.calculate_position_size(df_with_indicators, 50, 10000.0)
                results['calculate_position_size'] = isinstance(position_size, (int, float)) and position_size >= 0
            else:
                results['calculate_position_size'] = False
        except Exception as e:
            results['calculate_position_size'] = False
            results['calculate_position_size_error'] = str(e)
        
        # Test calculate_stop_loss
        try:
            if 'calculate_indicators' in results and results['calculate_indicators']:
                stop_loss = adapter.calculate_stop_loss(df_with_indicators, 50, 105.0)
                results['calculate_stop_loss'] = isinstance(stop_loss, (int, float)) and stop_loss > 0
            else:
                results['calculate_stop_loss'] = False
        except Exception as e:
            results['calculate_stop_loss'] = False
            results['calculate_stop_loss_error'] = str(e)
        
        # Test get_parameters
        try:
            params = adapter.get_parameters()
            results['get_parameters'] = isinstance(params, dict)
        except Exception as e:
            results['get_parameters'] = False
            results['get_parameters_error'] = str(e)
        
        return results


class MigrationHelper:
    """
    Helper utilities for migrating from legacy strategies to component-based adapters
    """
    
    @staticmethod
    def analyze_legacy_strategy(strategy: BaseStrategy) -> Dict[str, Any]:
        """
        Analyze legacy strategy to suggest component configuration
        
        Args:
            strategy: Legacy BaseStrategy to analyze
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        analysis = {
            'strategy_name': strategy.name,
            'strategy_class': strategy.__class__.__name__,
            'trading_pair': strategy.get_trading_pair(),
            'has_risk_overrides': strategy.get_risk_overrides() is not None,
            'suggested_components': {}
        }
        
        # Analyze risk overrides for component suggestions
        risk_overrides = strategy.get_risk_overrides()
        if risk_overrides:
            analysis['risk_overrides'] = risk_overrides
            
            # Suggest position sizer based on risk overrides
            if 'position_sizer' in risk_overrides:
                sizer_type = risk_overrides['position_sizer']
                if sizer_type == 'fixed_fraction':
                    analysis['suggested_components']['position_sizer'] = {
                        'type': 'fixed_fraction',
                        'params': {
                            'fraction': risk_overrides.get('base_fraction', 0.02)
                        }
                    }
                elif sizer_type == 'confidence_weighted':
                    analysis['suggested_components']['position_sizer'] = {
                        'type': 'confidence_weighted',
                        'params': {
                            'base_fraction': risk_overrides.get('base_fraction', 0.05)
                        }
                    }
            
            # Suggest risk manager based on risk overrides
            analysis['suggested_components']['risk_manager'] = {
                'type': 'fixed',
                'params': {
                    'risk_per_trade': risk_overrides.get('base_fraction', 0.02),
                    'stop_loss_pct': risk_overrides.get('stop_loss_pct', 0.05)
                }
            }
        
        # Default suggestions if no risk overrides
        if not analysis['suggested_components']:
            analysis['suggested_components'] = {
                'signal_generator': {'type': 'random', 'params': {'buy_prob': 0.3, 'sell_prob': 0.3}},
                'risk_manager': {'type': 'fixed', 'params': {'risk_per_trade': 0.02, 'stop_loss_pct': 0.05}},
                'position_sizer': {'type': 'fixed_fraction', 'params': {'fraction': 0.02}}
            }
        
        return analysis
    
    @staticmethod
    def create_migration_plan(strategies: List[BaseStrategy]) -> Dict[str, Any]:
        """
        Create migration plan for multiple legacy strategies
        
        Args:
            strategies: List of legacy strategies to migrate
            
        Returns:
            Dictionary with migration plan
        """
        plan = {
            'total_strategies': len(strategies),
            'strategy_analyses': [],
            'migration_steps': [],
            'estimated_effort': 'low'  # low, medium, high
        }
        
        # Analyze each strategy
        for strategy in strategies:
            analysis = MigrationHelper.analyze_legacy_strategy(strategy)
            plan['strategy_analyses'].append(analysis)
        
        # Create migration steps
        plan['migration_steps'] = [
            "1. Create adapter factory instance",
            "2. For each legacy strategy:",
            "   a. Analyze strategy using MigrationHelper.analyze_legacy_strategy()",
            "   b. Create adapter using AdapterFactory.convert_legacy_strategy()",
            "   c. Validate adapter using AdapterValidationUtils.validate_adapter_compatibility()",
            "   d. Test adapter using AdapterValidationUtils.test_adapter_methods()",
            "3. Update strategy registry to use adapters",
            "4. Run parallel testing with legacy and adapter versions",
            "5. Gradually replace legacy strategies with adapters"
        ]
        
        # Estimate effort based on complexity
        complex_strategies = sum(1 for analysis in plan['strategy_analyses'] 
                               if analysis['has_risk_overrides'])
        
        if complex_strategies > len(strategies) * 0.5:
            plan['estimated_effort'] = 'high'
        elif complex_strategies > len(strategies) * 0.2:
            plan['estimated_effort'] = 'medium'
        
        return plan


# Global factory instance for convenience
adapter_factory = AdapterFactory()