"""
Unit tests for Strategy Converter

Tests the strategy conversion utilities including parameter mapping,
component creation, and validation.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.strategies.base import BaseStrategy
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.migration.strategy_converter import (
    StrategyConverter,
    ConversionReport,
    ComponentMapping
)
from src.strategies.components.signal_generator import MLBasicSignalGenerator, SignalGenerator
from src.strategies.components.risk_manager import FixedRiskManager, RiskManager
from src.strategies.components.position_sizer import ConfidenceWeightedSizer, PositionSizer


class MockStrategy(BaseStrategy):
    """Mock strategy for testing"""
    
    def __init__(self, name="MockStrategy"):
        super().__init__(name)
        self.model_path = "test_model.onnx"
        self.sequence_length = 120
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        
    def calculate_indicators(self, df):
        return df.copy()
    
    def check_entry_conditions(self, df, index):
        return True
    
    def check_exit_conditions(self, df, index, entry_price):
        return False
    
    def calculate_position_size(self, df, index, balance):
        return balance * 0.1
    
    def calculate_stop_loss(self, df, index, price, side):
        return price * 0.95
    
    def get_parameters(self):
        return {
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }


class TestStrategyConverter:
    """Test cases for StrategyConverter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.converter = StrategyConverter()
        self.mock_strategy = MockStrategy()
    
    def test_converter_initialization(self):
        """Test converter initialization"""
        assert self.converter is not None
        assert hasattr(self.converter, '_conversion_mappings')
        assert len(self.converter._conversion_mappings) > 0
        assert 'MlBasic' in self.converter._conversion_mappings
        assert 'MlAdaptive' in self.converter._conversion_mappings
    
    def test_get_supported_strategy_types(self):
        """Test getting supported strategy types"""
        supported_types = self.converter.get_supported_strategy_types()
        
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert 'MlBasic' in supported_types
        assert 'MlAdaptive' in supported_types
    
    def test_extract_legacy_parameters(self):
        """Test parameter extraction from legacy strategy"""
        params = self.converter._extract_legacy_parameters(self.mock_strategy)
        
        assert isinstance(params, dict)
        assert 'model_path' in params
        assert 'sequence_length' in params
        assert 'stop_loss_pct' in params
        assert 'take_profit_pct' in params
        assert params['model_path'] == "test_model.onnx"
        assert params['sequence_length'] == 120
    
    def test_get_default_mapping(self):
        """Test getting default mapping for strategy"""
        # Test with known strategy type
        ml_basic = MlBasic()
        mapping = self.converter._get_default_mapping(ml_basic)
        
        assert mapping is not None
        assert mapping.signal_generator_type == MLBasicSignalGenerator
        assert mapping.risk_manager_type == FixedRiskManager
        assert mapping.position_sizer_type == ConfidenceWeightedSizer
    
    def test_get_default_mapping_fallback(self):
        """Test fallback mapping for unknown strategy type"""
        mapping = self.converter._get_default_mapping(self.mock_strategy)
        
        assert mapping is not None
        # Should fall back to BaseStrategy mapping
        assert 'BaseStrategy' in self.converter._conversion_mappings
    
    @patch('src.strategies.migration.strategy_converter.MLBasicSignalGenerator')
    @patch('src.strategies.migration.strategy_converter.FixedRiskManager')
    @patch('src.strategies.migration.strategy_converter.ConfidenceWeightedSizer')
    def test_create_components(self, mock_sizer, mock_risk, mock_signal):
        """Test component creation"""
        # Set up mocks
        mock_signal.return_value = Mock(spec=SignalGenerator)
        mock_risk.return_value = Mock(spec=RiskManager)
        mock_sizer.return_value = Mock(spec=PositionSizer)
        
        # Get mapping for MlBasic
        mapping = self.converter._conversion_mappings['MlBasic']
        
        # Create test parameters
        component_params = {
            'signal_generator': {'model_path': 'test.onnx', 'name': 'test_signal'},
            'risk_manager': {'stop_loss_percentage': 0.02, 'name': 'test_risk'},
            'position_sizer': {'base_fraction': 0.2, 'name': 'test_sizer'}
        }
        
        # Create mock report
        report = ConversionReport(
            strategy_name="test",
            conversion_timestamp=datetime.now(),
            source_strategy_type="MlBasic",
            target_components={},
            parameter_mappings={},
            validation_results={},
            warnings=[],
            errors=[],
            success=False,
            audit_trail=[]
        )
        
        # Test component creation
        components = self.converter._create_components(mapping, component_params, report)
        
        assert 'signal_generator' in components
        assert 'risk_manager' in components
        assert 'position_sizer' in components
        assert components['signal_generator'] is not None
        assert components['risk_manager'] is not None
        assert components['position_sizer'] is not None
    
    def test_map_parameters(self):
        """Test parameter mapping"""
        mapping = self.converter._conversion_mappings['MlBasic']
        legacy_params = {
            'model_path': 'test.onnx',
            'sequence_length': 120,
            'stop_loss_pct': 0.02,
            'BASE_POSITION_SIZE': 0.2
        }
        
        component_params = self.converter._map_parameters(legacy_params, mapping)
        
        assert isinstance(component_params, dict)
        assert 'signal_generator' in component_params
        assert 'risk_manager' in component_params
        assert 'position_sizer' in component_params
        
        # Check signal generator parameters
        signal_params = component_params['signal_generator']
        assert 'model_path' in signal_params
        assert signal_params['model_path'] == 'test.onnx'
        assert signal_params['name'] == 'ml_basic_signals'
        
        # Check risk manager parameters
        risk_params = component_params['risk_manager']
        assert 'stop_loss_percentage' in risk_params
        assert risk_params['stop_loss_percentage'] == 0.02
        
        # Check position sizer parameters
        sizer_params = component_params['position_sizer']
        assert 'base_fraction' in sizer_params
        assert sizer_params['base_fraction'] == 0.2
    
    @patch('src.strategies.migration.strategy_converter.LegacyStrategyAdapter')
    @patch('src.strategies.migration.strategy_converter.MLBasicSignalGenerator')
    @patch('src.strategies.migration.strategy_converter.FixedRiskManager')
    @patch('src.strategies.migration.strategy_converter.ConfidenceWeightedSizer')
    def test_convert_strategy_success(self, mock_sizer, mock_risk, mock_signal, mock_adapter):
        """Test successful strategy conversion"""
        # Set up mocks
        mock_signal.return_value = Mock(spec=SignalGenerator)
        mock_risk.return_value = Mock(spec=RiskManager)
        mock_sizer.return_value = Mock(spec=PositionSizer)
        mock_adapter.return_value = Mock()
        mock_adapter.return_value.name = "converted_test"
        mock_adapter.return_value.get_trading_pair.return_value = "BTCUSDT"
        mock_adapter.return_value.get_parameters.return_value = {}
        mock_adapter.return_value.get_component_status.return_value = {
            'signal_generator': 'MLBasicSignalGenerator',
            'risk_manager': 'FixedRiskManager',
            'position_sizer': 'ConfidenceWeightedSizer'
        }
        
        # Convert strategy
        adapter, report = self.converter.convert_strategy(self.mock_strategy)
        
        assert adapter is not None
        assert report is not None
        assert report.success is True
        assert len(report.errors) == 0
        assert report.strategy_name == "converted_MockStrategy"
        assert report.source_strategy_type == "MockStrategy"
    
    def test_convert_strategy_with_custom_mapping(self):
        """Test strategy conversion with custom mapping"""
        # Create custom mapping
        custom_mapping = ComponentMapping(
            signal_generator_type=MLBasicSignalGenerator,
            risk_manager_type=FixedRiskManager,
            position_sizer_type=ConfidenceWeightedSizer,
            parameter_mappings={
                'signal_generator': {'model_path': 'model_path'},
                'risk_manager': {'stop_loss_pct': 'stop_loss_percentage'},
                'position_sizer': {}
            },
            component_configs={
                'signal_generator': {'name': 'custom_signal'},
                'risk_manager': {'name': 'custom_risk'},
                'position_sizer': {'name': 'custom_sizer'}
            }
        )
        
        with patch('src.strategies.migration.strategy_converter.MLBasicSignalGenerator') as mock_signal, \
             patch('src.strategies.migration.strategy_converter.FixedRiskManager') as mock_risk, \
             patch('src.strategies.migration.strategy_converter.ConfidenceWeightedSizer') as mock_sizer, \
             patch('src.strategies.migration.strategy_converter.LegacyStrategyAdapter') as mock_adapter:
            
            # Set up mocks
            mock_signal.return_value = Mock(spec=SignalGenerator)
            mock_risk.return_value = Mock(spec=RiskManager)
            mock_sizer.return_value = Mock(spec=PositionSizer)
            mock_adapter.return_value = Mock()
            mock_adapter.return_value.name = "custom_converted"
            mock_adapter.return_value.get_trading_pair.return_value = "BTCUSDT"
            mock_adapter.return_value.get_parameters.return_value = {}
            mock_adapter.return_value.get_component_status.return_value = {
                'signal_generator': 'MLBasicSignalGenerator',
                'risk_manager': 'FixedRiskManager',
                'position_sizer': 'ConfidenceWeightedSizer'
            }
            
            # Convert with custom mapping
            adapter, report = self.converter.convert_strategy(
                self.mock_strategy,
                target_name="custom_converted",
                custom_mapping=custom_mapping
            )
            
            assert adapter is not None
            assert report is not None
            assert report.strategy_name == "custom_converted"
    
    def test_convert_strategy_validation_disabled(self):
        """Test strategy conversion with validation disabled"""
        with patch('src.strategies.migration.strategy_converter.MLBasicSignalGenerator') as mock_signal, \
             patch('src.strategies.migration.strategy_converter.FixedRiskManager') as mock_risk, \
             patch('src.strategies.migration.strategy_converter.ConfidenceWeightedSizer') as mock_sizer, \
             patch('src.strategies.migration.strategy_converter.LegacyStrategyAdapter') as mock_adapter:
            
            # Set up mocks
            mock_signal.return_value = Mock(spec=SignalGenerator)
            mock_risk.return_value = Mock(spec=RiskManager)
            mock_sizer.return_value = Mock(spec=PositionSizer)
            mock_adapter.return_value = Mock()
            mock_adapter.return_value.name = "test_converted"
            mock_adapter.return_value.get_trading_pair.return_value = "BTCUSDT"
            mock_adapter.return_value.get_parameters.return_value = {}
            
            # Convert without validation
            adapter, report = self.converter.convert_strategy(
                self.mock_strategy,
                validate_conversion=False
            )
            
            assert adapter is not None
            assert report is not None
            assert len(report.validation_results) == 0
    
    def test_convert_unsupported_strategy(self):
        """Test conversion of unsupported strategy type"""
        # Create a strategy type that's not in mappings
        class UnsupportedStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("UnsupportedStrategy")
            
            def calculate_indicators(self, df):
                return df
            
            def check_entry_conditions(self, df, index):
                return False
            
            def check_exit_conditions(self, df, index, entry_price):
                return False
            
            def calculate_position_size(self, df, index, balance):
                return 0
            
            def calculate_stop_loss(self, df, index, price, side):
                return price
            
            def get_parameters(self):
                return {}
        
        # Remove all mappings to force failure
        original_mappings = self.converter._conversion_mappings.copy()
        self.converter._conversion_mappings.clear()
        
        try:
            unsupported = UnsupportedStrategy()
            adapter, report = self.converter.convert_strategy(unsupported)
            
            assert adapter is None
            assert report is not None
            assert report.success is False
            assert len(report.errors) > 0
            assert "No conversion mapping available" in report.errors[0]
        finally:
            # Restore mappings
            self.converter._conversion_mappings = original_mappings
    
    def test_batch_convert_strategies(self):
        """Test batch conversion of multiple strategies"""
        strategies = [MockStrategy(f"Strategy{i}") for i in range(3)]
        
        with patch('src.strategies.migration.strategy_converter.MLBasicSignalGenerator') as mock_signal, \
             patch('src.strategies.migration.strategy_converter.FixedRiskManager') as mock_risk, \
             patch('src.strategies.migration.strategy_converter.ConfidenceWeightedSizer') as mock_sizer, \
             patch('src.strategies.migration.strategy_converter.LegacyStrategyAdapter') as mock_adapter:
            
            # Set up mocks
            mock_signal.return_value = Mock(spec=SignalGenerator)
            mock_risk.return_value = Mock(spec=RiskManager)
            mock_sizer.return_value = Mock(spec=PositionSizer)
            mock_adapter.return_value = Mock()
            mock_adapter.return_value.get_trading_pair.return_value = "BTCUSDT"
            mock_adapter.return_value.get_parameters.return_value = {}
            mock_adapter.return_value.get_component_status.return_value = {
                'signal_generator': 'MLBasicSignalGenerator',
                'risk_manager': 'FixedRiskManager',
                'position_sizer': 'ConfidenceWeightedSizer'
            }
            
            # Batch convert
            results = self.converter.batch_convert_strategies(strategies)
            
            assert len(results) == 3
            for adapter, report in results:
                assert adapter is not None
                assert report is not None
                assert report.success is True
    
    def test_add_custom_mapping(self):
        """Test adding custom conversion mapping"""
        custom_mapping = ComponentMapping(
            signal_generator_type=MLBasicSignalGenerator,
            risk_manager_type=FixedRiskManager,
            position_sizer_type=ConfidenceWeightedSizer,
            parameter_mappings={},
            component_configs={}
        )
        
        # Add custom mapping
        self.converter.add_custom_mapping("CustomStrategy", custom_mapping)
        
        # Verify it was added
        assert "CustomStrategy" in self.converter._conversion_mappings
        assert self.converter._conversion_mappings["CustomStrategy"] == custom_mapping
    
    def test_get_mapping_for_strategy(self):
        """Test getting mapping for specific strategy type"""
        mapping = self.converter.get_mapping_for_strategy("MlBasic")
        
        assert mapping is not None
        assert mapping.signal_generator_type == MLBasicSignalGenerator
        
        # Test non-existent mapping
        missing_mapping = self.converter.get_mapping_for_strategy("NonExistentStrategy")
        assert missing_mapping is None
    
    def test_conversion_history(self):
        """Test conversion history tracking"""
        # Initially empty
        assert len(self.converter.get_conversion_history()) == 0
        
        with patch('src.strategies.migration.strategy_converter.MLBasicSignalGenerator') as mock_signal, \
             patch('src.strategies.migration.strategy_converter.FixedRiskManager') as mock_risk, \
             patch('src.strategies.migration.strategy_converter.ConfidenceWeightedSizer') as mock_sizer, \
             patch('src.strategies.migration.strategy_converter.LegacyStrategyAdapter') as mock_adapter:
            
            # Set up mocks
            mock_signal.return_value = Mock(spec=SignalGenerator)
            mock_risk.return_value = Mock(spec=RiskManager)
            mock_sizer.return_value = Mock(spec=PositionSizer)
            mock_adapter.return_value = Mock()
            mock_adapter.return_value.name = "test_converted"
            mock_adapter.return_value.get_trading_pair.return_value = "BTCUSDT"
            mock_adapter.return_value.get_parameters.return_value = {}
            mock_adapter.return_value.get_component_status.return_value = {
                'signal_generator': 'MLBasicSignalGenerator',
                'risk_manager': 'FixedRiskManager',
                'position_sizer': 'ConfidenceWeightedSizer'
            }
            
            # Convert strategy
            adapter, report = self.converter.convert_strategy(self.mock_strategy)
            
            # Check history
            history = self.converter.get_conversion_history()
            assert len(history) == 1
            assert history[0] == report
        
        # Clear history
        self.converter.clear_conversion_history()
        assert len(self.converter.get_conversion_history()) == 0
    
    def test_generate_conversion_summary(self):
        """Test conversion summary generation"""
        # Test with empty history
        summary = self.converter.generate_conversion_summary()
        assert summary['total_conversions'] == 0
        assert summary['successful_conversions'] == 0
        assert summary['success_rate'] == 0.0
        
        # Add some mock conversion reports
        successful_report = ConversionReport(
            strategy_name="test1",
            conversion_timestamp=datetime.now(),
            source_strategy_type="MlBasic",
            target_components={},
            parameter_mappings={},
            validation_results={},
            warnings=[],
            errors=[],
            success=True,
            audit_trail=[]
        )
        
        failed_report = ConversionReport(
            strategy_name="test2",
            conversion_timestamp=datetime.now(),
            source_strategy_type="MlAdaptive",
            target_components={},
            parameter_mappings={},
            validation_results={},
            warnings=[],
            errors=["Test error"],
            success=False,
            audit_trail=[]
        )
        
        self.converter.conversion_history = [successful_report, failed_report]
        
        # Test summary with data
        summary = self.converter.generate_conversion_summary()
        assert summary['total_conversions'] == 2
        assert summary['successful_conversions'] == 1
        assert summary['failed_conversions'] == 1
        assert summary['success_rate'] == 50.0
        assert len(summary['strategy_types_converted']) == 2
        assert 'MlBasic' in summary['strategy_types_converted']
        assert 'MlAdaptive' in summary['strategy_types_converted']


class TestConversionReport:
    """Test cases for ConversionReport"""
    
    def test_conversion_report_creation(self):
        """Test conversion report creation"""
        report = ConversionReport(
            strategy_name="test_strategy",
            conversion_timestamp=datetime.now(),
            source_strategy_type="MlBasic",
            target_components={'signal_generator': 'MLBasicSignalGenerator'},
            parameter_mappings={'test': 'value'},
            validation_results={'test': True},
            warnings=["Warning 1"],
            errors=["Error 1"],
            success=True,
            audit_trail=["Step 1", "Step 2"]
        )
        
        assert report.strategy_name == "test_strategy"
        assert report.source_strategy_type == "MlBasic"
        assert report.success is True
        assert len(report.warnings) == 1
        assert len(report.errors) == 1
        assert len(report.audit_trail) == 2
    
    def test_conversion_report_to_dict(self):
        """Test conversion report serialization"""
        timestamp = datetime.now()
        report = ConversionReport(
            strategy_name="test_strategy",
            conversion_timestamp=timestamp,
            source_strategy_type="MlBasic",
            target_components={'signal_generator': 'MLBasicSignalGenerator'},
            parameter_mappings={'test': 'value'},
            validation_results={'test': True},
            warnings=["Warning 1"],
            errors=["Error 1"],
            success=True,
            audit_trail=["Step 1", "Step 2"]
        )
        
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert report_dict['strategy_name'] == "test_strategy"
        assert report_dict['conversion_timestamp'] == timestamp.isoformat()
        assert report_dict['source_strategy_type'] == "MlBasic"
        assert report_dict['success'] is True
        assert report_dict['warnings'] == ["Warning 1"]
        assert report_dict['errors'] == ["Error 1"]


class TestComponentMapping:
    """Test cases for ComponentMapping"""
    
    def test_component_mapping_creation(self):
        """Test component mapping creation"""
        mapping = ComponentMapping(
            signal_generator_type=MLBasicSignalGenerator,
            risk_manager_type=FixedRiskManager,
            position_sizer_type=ConfidenceWeightedSizer,
            parameter_mappings={
                'signal_generator': {'model_path': 'model_path'},
                'risk_manager': {'stop_loss_pct': 'stop_loss_percentage'},
                'position_sizer': {'base_size': 'base_fraction'}
            },
            component_configs={
                'signal_generator': {'name': 'test_signal'},
                'risk_manager': {'name': 'test_risk'},
                'position_sizer': {'name': 'test_sizer'}
            }
        )
        
        assert mapping.signal_generator_type == MLBasicSignalGenerator
        assert mapping.risk_manager_type == FixedRiskManager
        assert mapping.position_sizer_type == ConfidenceWeightedSizer
        assert 'signal_generator' in mapping.parameter_mappings
        assert 'risk_manager' in mapping.parameter_mappings
        assert 'position_sizer' in mapping.parameter_mappings
        assert mapping.component_configs['signal_generator']['name'] == 'test_signal'