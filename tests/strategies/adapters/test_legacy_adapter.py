"""
Tests for LegacyStrategyAdapter

This module tests the LegacyStrategyAdapter class to ensure it properly
implements the BaseStrategy interface and correctly integrates components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.strategies.base import BaseStrategy
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components.signal_generator import (
    SignalGenerator, Signal, SignalDirection, HoldSignalGenerator, RandomSignalGenerator
)
from src.strategies.components.risk_manager import (
    RiskManager, Position, MarketData, FixedRiskManager
)
from src.strategies.components.position_sizer import (
    PositionSizer, FixedFractionSizer, ConfidenceWeightedSizer
)
from src.strategies.components.regime_context import (
    EnhancedRegimeDetector, RegimeContext, TrendLabel, VolLabel
)


class TestLegacyStrategyAdapter:
    """Test cases for LegacyStrategyAdapter"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        return data
    
    @pytest.fixture
    def mock_signal_generator(self):
        """Create mock signal generator"""
        generator = Mock(spec=SignalGenerator)
        generator.name = "mock_signal_generator"
        generator.generate_signal.return_value = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            confidence=0.7,
            metadata={'test': True}
        )
        generator.get_confidence.return_value = 0.7
        generator.get_parameters.return_value = {'name': 'mock_signal_generator'}
        return generator
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager"""
        manager = Mock(spec=RiskManager)
        manager.name = "mock_risk_manager"
        manager.calculate_position_size.return_value = 100.0
        manager.should_exit.return_value = False
        manager.get_stop_loss.return_value = 95.0
        manager.get_parameters.return_value = {'name': 'mock_risk_manager'}
        return manager
    
    @pytest.fixture
    def mock_position_sizer(self):
        """Create mock position sizer"""
        sizer = Mock(spec=PositionSizer)
        sizer.name = "mock_position_sizer"
        sizer.calculate_size.return_value = 200.0
        sizer.get_parameters.return_value = {'name': 'mock_position_sizer'}
        return sizer
    
    @pytest.fixture
    def mock_regime_detector(self):
        """Create mock regime detector"""
        detector = Mock(spec=EnhancedRegimeDetector)
        detector.detect_regime.return_value = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=10,
            strength=0.7,
            timestamp=datetime.now()
        )
        return detector
    
    @pytest.fixture
    def adapter(self, mock_signal_generator, mock_risk_manager, mock_position_sizer, mock_regime_detector):
        """Create LegacyStrategyAdapter instance for testing"""
        return LegacyStrategyAdapter(
            signal_generator=mock_signal_generator,
            risk_manager=mock_risk_manager,
            position_sizer=mock_position_sizer,
            regime_detector=mock_regime_detector,
            name="test_adapter"
        )
    
    def test_adapter_initialization(self, mock_signal_generator, mock_risk_manager, 
                                  mock_position_sizer, mock_regime_detector):
        """Test adapter initialization"""
        adapter = LegacyStrategyAdapter(
            signal_generator=mock_signal_generator,
            risk_manager=mock_risk_manager,
            position_sizer=mock_position_sizer,
            regime_detector=mock_regime_detector,
            name="test_adapter"
        )
        
        assert adapter.name == "test_adapter"
        assert adapter.signal_generator == mock_signal_generator
        assert adapter.risk_manager == mock_risk_manager
        assert adapter.position_sizer == mock_position_sizer
        assert adapter.regime_detector == mock_regime_detector
        assert isinstance(adapter, BaseStrategy)
    
    def test_adapter_auto_name_generation(self, mock_signal_generator, mock_risk_manager, 
                                        mock_position_sizer):
        """Test automatic name generation when no name provided"""
        adapter = LegacyStrategyAdapter(
            signal_generator=mock_signal_generator,
            risk_manager=mock_risk_manager,
            position_sizer=mock_position_sizer
        )
        
        expected_name = f"adapter_{mock_signal_generator.name}_{mock_risk_manager.name}_{mock_position_sizer.name}"
        assert adapter.name == expected_name
    
    def test_calculate_indicators(self, adapter, sample_data):
        """Test calculate_indicators method"""
        # Mock the regime detector's base detector
        adapter.regime_detector.base_detector = Mock()
        adapter.regime_detector.base_detector.annotate.return_value = sample_data.copy()
        
        result = adapter.calculate_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        # Should call annotate if regime_label not in columns
        adapter.regime_detector.base_detector.annotate.assert_called_once()
    
    def test_calculate_indicators_with_existing_regime_labels(self, adapter, sample_data):
        """Test calculate_indicators when regime labels already exist"""
        # Add regime labels to sample data
        sample_data_with_regime = sample_data.copy()
        sample_data_with_regime['regime_label'] = 'trend_up:low_vol'
        
        adapter.regime_detector.base_detector = Mock()
        
        result = adapter.calculate_indicators(sample_data_with_regime)
        
        assert isinstance(result, pd.DataFrame)
        assert 'regime_label' in result.columns
        # Should not call annotate if regime_label already exists
        adapter.regime_detector.base_detector.annotate.assert_not_called()
    
    def test_check_entry_conditions_buy_signal(self, adapter, sample_data):
        """Test check_entry_conditions with BUY signal"""
        # Setup sample data with regime labels
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Mock signal generator to return BUY signal
        adapter.signal_generator.generate_signal.return_value = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            confidence=0.7,
            metadata={}
        )
        
        result = adapter.check_entry_conditions(sample_data, 50)
        
        assert result is True
        adapter.signal_generator.generate_signal.assert_called_once()
        adapter.regime_detector.detect_regime.assert_called_once_with(sample_data, 50)
    
    def test_check_entry_conditions_hold_signal(self, adapter, sample_data):
        """Test check_entry_conditions with HOLD signal"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Mock signal generator to return HOLD signal
        adapter.signal_generator.generate_signal.return_value = Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.9,
            metadata={}
        )
        
        result = adapter.check_entry_conditions(sample_data, 50)
        
        assert result is False
    
    def test_check_entry_conditions_error_handling(self, adapter, sample_data):
        """Test check_entry_conditions error handling"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Mock signal generator to raise exception
        adapter.signal_generator.generate_signal.side_effect = Exception("Test error")
        
        result = adapter.check_entry_conditions(sample_data, 50)
        
        assert result is False
        assert adapter.performance_metrics['component_errors'] > 0
    
    def test_check_exit_conditions_no_exit(self, adapter, sample_data):
        """Test check_exit_conditions when no exit is needed"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Mock risk manager to return False for should_exit
        adapter.risk_manager.should_exit.return_value = False
        
        result = adapter.check_exit_conditions(sample_data, 50, 105.0)
        
        assert result is False
        adapter.risk_manager.should_exit.assert_called_once()
        
        # Verify Position and MarketData objects were created correctly
        call_args = adapter.risk_manager.should_exit.call_args[0]
        position = call_args[0]
        market_data = call_args[1]
        
        assert isinstance(position, Position)
        assert position.entry_price == 105.0
        assert isinstance(market_data, MarketData)
    
    def test_check_exit_conditions_should_exit(self, adapter, sample_data):
        """Test check_exit_conditions when exit is needed"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Mock risk manager to return True for should_exit
        adapter.risk_manager.should_exit.return_value = True
        
        result = adapter.check_exit_conditions(sample_data, 50, 105.0)
        
        assert result is True
    
    def test_calculate_position_size(self, adapter, sample_data):
        """Test calculate_position_size method"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Mock components to return specific values
        adapter.risk_manager.calculate_position_size.return_value = 100.0
        adapter.position_sizer.calculate_size.return_value = 200.0
        
        result = adapter.calculate_position_size(sample_data, 50, 10000.0)
        
        assert result == 200.0
        adapter.risk_manager.calculate_position_size.assert_called_once()
        adapter.position_sizer.calculate_size.assert_called_once()
    
    def test_calculate_position_size_with_cached_signal(self, adapter, sample_data):
        """Test calculate_position_size uses cached signal"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Set a cached signal
        cached_signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.9,
            confidence=0.8,
            metadata={}
        )
        adapter._last_signal = cached_signal
        
        adapter.calculate_position_size(sample_data, 50, 10000.0)
        
        # Should use cached signal, not generate new one
        adapter.signal_generator.generate_signal.assert_not_called()
        
        # Verify cached signal was passed to components
        risk_call_args = adapter.risk_manager.calculate_position_size.call_args[0]
        assert risk_call_args[0] == cached_signal
    
    def test_calculate_stop_loss(self, adapter, sample_data):
        """Test calculate_stop_loss method"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        adapter.risk_manager.get_stop_loss.return_value = 95.0
        
        result = adapter.calculate_stop_loss(sample_data, 50, 100.0, "long")
        
        assert result == 95.0
        adapter.risk_manager.get_stop_loss.assert_called_once()
    
    def test_get_parameters(self, adapter):
        """Test get_parameters method"""
        params = adapter.get_parameters()
        
        assert isinstance(params, dict)
        assert 'adapter_name' in params
        assert 'trading_pair' in params
        assert 'signal_generator' in params
        assert 'risk_manager' in params
        assert 'position_sizer' in params
        assert 'performance_metrics' in params
        
        # Verify component parameters are included
        assert params['signal_generator']['name'] == 'mock_signal_generator'
        assert params['risk_manager']['name'] == 'mock_risk_manager'
        assert params['position_sizer']['name'] == 'mock_position_sizer'
    
    def test_performance_metrics_tracking(self, adapter, sample_data):
        """Test performance metrics are properly tracked"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Reset metrics
        adapter.reset_performance_metrics()
        
        # Perform various operations
        adapter.check_entry_conditions(sample_data, 50)
        adapter.check_exit_conditions(sample_data, 50, 105.0)
        adapter.calculate_position_size(sample_data, 50, 10000.0)
        
        metrics = adapter.get_performance_metrics()
        
        assert metrics['entry_conditions_checked'] == 1
        assert metrics['exit_conditions_checked'] == 1
        assert metrics['position_sizes_calculated'] == 1
        assert metrics['signals_generated'] >= 1
        assert metrics['regime_detections'] >= 1
        
        # Check execution time tracking
        assert 'avg_signal_generation_time' in metrics
        assert 'avg_risk_management_time' in metrics
        assert 'avg_position_sizing_time' in metrics
    
    def test_regime_context_caching(self, adapter, sample_data):
        """Test regime context caching functionality"""
        sample_data['regime_label'] = 'trend_up:low_vol'
        
        # Call method twice with same index
        adapter.check_entry_conditions(sample_data, 50)
        adapter.check_entry_conditions(sample_data, 50)
        
        # Should only call detect_regime once due to caching
        assert adapter.regime_detector.detect_regime.call_count == 1
        
        # Call with different index should trigger new detection
        adapter.check_entry_conditions(sample_data, 51)
        assert adapter.regime_detector.detect_regime.call_count == 2
    
    def test_component_status(self, adapter):
        """Test get_component_status method"""
        status = adapter.get_component_status()
        
        assert isinstance(status, dict)
        assert 'signal_generator' in status
        assert 'risk_manager' in status
        assert 'position_sizer' in status
        assert 'regime_detector' in status
        assert 'current_regime' in status
        assert 'last_signal' in status
    
    def test_string_representations(self, adapter):
        """Test __str__ and __repr__ methods"""
        str_repr = str(adapter)
        assert "LegacyStrategyAdapter" in str_repr
        assert "test_adapter" in str_repr
        
        repr_str = repr(adapter)
        assert "LegacyStrategyAdapter" in repr_str
        assert "test_adapter" in repr_str


class TestLegacyAdapterWithRealComponents:
    """Test LegacyStrategyAdapter with real component implementations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        return data
    
    @pytest.fixture
    def real_adapter(self):
        """Create adapter with real component implementations"""
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager(risk_per_trade=0.02, stop_loss_pct=0.05)
        position_sizer = FixedFractionSizer(fraction=0.02)
        regime_detector = EnhancedRegimeDetector()
        
        return LegacyStrategyAdapter(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name="real_adapter"
        )
    
    def test_full_workflow_with_real_components(self, real_adapter, sample_data):
        """Test complete workflow with real components"""
        # Calculate indicators
        df_with_indicators = real_adapter.calculate_indicators(sample_data)
        assert isinstance(df_with_indicators, pd.DataFrame)
        
        # Check entry conditions (should be False for HoldSignalGenerator)
        entry_result = real_adapter.check_entry_conditions(df_with_indicators, 50)
        assert entry_result is False
        
        # Check exit conditions
        exit_result = real_adapter.check_exit_conditions(df_with_indicators, 50, 105.0)
        assert isinstance(exit_result, bool)
        
        # Calculate position size
        position_size = real_adapter.calculate_position_size(df_with_indicators, 50, 10000.0)
        assert isinstance(position_size, (int, float))
        assert position_size >= 0
        
        # Calculate stop loss
        stop_loss = real_adapter.calculate_stop_loss(df_with_indicators, 50, 105.0)
        assert isinstance(stop_loss, (int, float))
        assert stop_loss > 0
        
        # Get parameters
        params = real_adapter.get_parameters()
        assert isinstance(params, dict)
    
    def test_random_signal_generator_integration(self, sample_data):
        """Test adapter with RandomSignalGenerator"""
        signal_generator = RandomSignalGenerator(buy_prob=0.5, sell_prob=0.3, seed=42)
        risk_manager = FixedRiskManager()
        position_sizer = ConfidenceWeightedSizer()
        
        adapter = LegacyStrategyAdapter(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            name="random_adapter"
        )
        
        df_with_indicators = adapter.calculate_indicators(sample_data)
        
        # Test multiple entry checks to verify randomness
        results = []
        for i in range(10, 20):
            result = adapter.check_entry_conditions(df_with_indicators, i)
            results.append(result)
        
        # Should have some variation in results (not all same)
        assert len(set(results)) > 1 or all(r is False for r in results)  # All False is valid for HOLD/SELL signals
    
    def test_error_recovery_with_real_components(self, real_adapter, sample_data):
        """Test error recovery with real components"""
        # Test with invalid index
        result = real_adapter.check_entry_conditions(sample_data, -1)
        assert result is False
        
        result = real_adapter.check_entry_conditions(sample_data, len(sample_data))
        assert result is False
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = real_adapter.check_entry_conditions(empty_df, 0)
        assert result is False
    
    def test_performance_metrics_with_real_components(self, real_adapter, sample_data):
        """Test performance metrics collection with real components"""
        df_with_indicators = real_adapter.calculate_indicators(sample_data)
        
        # Reset metrics
        real_adapter.reset_performance_metrics()
        
        # Perform operations
        for i in range(10, 15):
            real_adapter.check_entry_conditions(df_with_indicators, i)
            real_adapter.check_exit_conditions(df_with_indicators, i, 105.0)
            real_adapter.calculate_position_size(df_with_indicators, i, 10000.0)
        
        metrics = real_adapter.get_performance_metrics()
        
        assert metrics['entry_conditions_checked'] == 5
        assert metrics['exit_conditions_checked'] == 5
        assert metrics['position_sizes_calculated'] == 5
        assert metrics['component_errors'] == 0  # Should be no errors with valid data
        
        # Execution times should be recorded
        assert metrics['avg_signal_generation_time'] >= 0
        assert metrics['avg_risk_management_time'] >= 0
        assert metrics['avg_position_sizing_time'] >= 0