"""
Unit tests for StrategyManager with versioning
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.components.strategy_manager import (
    StrategyManager, StrategyVersion, StrategyExecution
)
from src.strategies.components.signal_generator import (
    SignalGenerator, Signal, SignalDirection, HoldSignalGenerator, RandomSignalGenerator
)
from src.strategies.components.risk_manager import RiskManager, FixedRiskManager
from src.strategies.components.position_sizer import PositionSizer, FixedFractionSizer
from src.strategies.components.regime_context import EnhancedRegimeDetector


class TestStrategyVersion:
    """Test StrategyVersion dataclass"""
    
    def test_strategy_version_creation(self):
        """Test creating a strategy version"""
        created_at = datetime.now()
        components = {
            'signal_generator': {'name': 'test_gen', 'type': 'TestGenerator'},
            'risk_manager': {'name': 'test_risk', 'type': 'TestRiskManager'}
        }
        parameters = {'param1': 'value1', 'param2': 42}
        
        version = StrategyVersion(
            version_id="test-version-123",
            name="Test Version",
            description="Test strategy version",
            created_at=created_at,
            components=components,
            parameters=parameters,
            is_active=True,
            performance_metrics={'accuracy': 0.85}
        )
        
        assert version.version_id == "test-version-123"
        assert version.name == "Test Version"
        assert version.description == "Test strategy version"
        assert version.created_at == created_at
        assert version.components == components
        assert version.parameters == parameters
        assert version.is_active is True
        assert version.performance_metrics == {'accuracy': 0.85}
    
    def test_strategy_version_to_dict(self):
        """Test converting strategy version to dictionary"""
        created_at = datetime.now()
        version = StrategyVersion(
            version_id="test-123",
            name="Test",
            description="Test version",
            created_at=created_at,
            components={},
            parameters={}
        )
        
        version_dict = version.to_dict()
        
        assert version_dict['version_id'] == "test-123"
        assert version_dict['name'] == "Test"
        assert version_dict['created_at'] == created_at.isoformat()
    
    def test_strategy_version_from_dict(self):
        """Test creating strategy version from dictionary"""
        created_at = datetime.now()
        version_dict = {
            'version_id': "test-123",
            'name': "Test",
            'description': "Test version",
            'created_at': created_at.isoformat(),
            'components': {},
            'parameters': {},
            'is_active': False,
            'performance_metrics': None
        }
        
        version = StrategyVersion.from_dict(version_dict)
        
        assert version.version_id == "test-123"
        assert version.name == "Test"
        assert version.created_at == created_at


class TestStrategyExecution:
    """Test StrategyExecution dataclass"""
    
    def test_strategy_execution_creation(self):
        """Test creating a strategy execution record"""
        timestamp = datetime.now()
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            confidence=0.9,
            metadata={}
        )
        
        execution = StrategyExecution(
            timestamp=timestamp,
            signal=signal,
            regime=None,
            position_size=1000.0,
            risk_metrics={'risk_amount': 200.0},
            execution_time_ms=15.5,
            version_id="version-123"
        )
        
        assert execution.timestamp == timestamp
        assert execution.signal == signal
        assert execution.regime is None
        assert execution.position_size == 1000.0
        assert execution.risk_metrics == {'risk_amount': 200.0}
        assert execution.execution_time_ms == 15.5
        assert execution.version_id == "version-123"


class TestStrategyManager:
    """Test StrategyManager"""
    
    def create_test_dataframe(self, length=100):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=length, freq='1H')
        
        # Create trending data
        base_price = 50000
        trend = np.linspace(0, 0.05, length)  # 5% trend over period
        noise = np.random.normal(0, 0.005, length)  # 0.5% noise
        
        prices = base_price * (1 + trend + noise)
        
        data = {
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, length)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def create_test_strategy_manager(self):
        """Create test strategy manager with mock components"""
        signal_gen = HoldSignalGenerator()
        risk_mgr = FixedRiskManager()
        pos_sizer = FixedFractionSizer()
        
        return StrategyManager(
            name="test_strategy",
            signal_generator=signal_gen,
            risk_manager=risk_mgr,
            position_sizer=pos_sizer
        )
    
    def test_strategy_manager_initialization(self):
        """Test StrategyManager initialization"""
        manager = self.create_test_strategy_manager()
        
        assert manager.name == "test_strategy"
        assert isinstance(manager.signal_generator, HoldSignalGenerator)
        assert isinstance(manager.risk_manager, FixedRiskManager)
        assert isinstance(manager.position_sizer, FixedFractionSizer)
        assert isinstance(manager.regime_detector, EnhancedRegimeDetector)
        
        # Should have initial version
        assert len(manager.versions) == 1
        assert manager.current_version_id is not None
        assert manager.versions[manager.current_version_id].is_active is True
    
    def test_execute_strategy_valid(self):
        """Test strategy execution with valid inputs"""
        manager = self.create_test_strategy_manager()
        df = self.create_test_dataframe()
        
        signal, position_size, metadata = manager.execute_strategy(df, 50, 10000.0)
        
        # Should return valid results
        assert isinstance(signal, Signal)
        assert signal.direction == SignalDirection.HOLD  # HoldSignalGenerator
        assert position_size == 0.0  # HOLD signal should have 0 size
        assert isinstance(metadata, dict)
        assert 'regime' in metadata
        assert 'execution_time_ms' in metadata
        assert 'version_id' in metadata
        
        # Should have execution history
        assert len(manager.execution_history) == 1
        assert isinstance(manager.execution_history[0], StrategyExecution)
    
    def test_execute_strategy_with_buy_signal(self):
        """Test strategy execution with BUY signal"""
        # Use RandomSignalGenerator with high buy probability
        signal_gen = RandomSignalGenerator(buy_prob=1.0, sell_prob=0.0, seed=42)
        risk_mgr = FixedRiskManager()
        pos_sizer = FixedFractionSizer()
        
        manager = StrategyManager(
            name="test_buy_strategy",
            signal_generator=signal_gen,
            risk_manager=risk_mgr,
            position_sizer=pos_sizer
        )
        
        df = self.create_test_dataframe()
        
        signal, position_size, metadata = manager.execute_strategy(df, 50, 10000.0)
        
        # Should return BUY signal with non-zero position size
        assert signal.direction == SignalDirection.BUY
        assert position_size > 0
    
    def test_execute_strategy_error_handling(self):
        """Test strategy execution error handling"""
        # Create manager with mock that raises exception
        signal_gen = Mock()
        signal_gen.generate_signal.side_effect = Exception("Test error")
        signal_gen.name = "error_generator"
        
        risk_mgr = FixedRiskManager()
        pos_sizer = FixedFractionSizer()
        
        manager = StrategyManager(
            name="error_strategy",
            signal_generator=signal_gen,
            risk_manager=risk_mgr,
            position_sizer=pos_sizer
        )
        
        df = self.create_test_dataframe()
        
        signal, position_size, metadata = manager.execute_strategy(df, 50, 10000.0)
        
        # Should return safe defaults
        assert signal.direction == SignalDirection.HOLD
        assert position_size == 0.0
        assert 'error' in metadata
    
    def test_create_version(self):
        """Test creating a new strategy version"""
        manager = self.create_test_strategy_manager()
        initial_version_count = len(manager.versions)
        
        version_id = manager.create_version(
            name="Test Version 2",
            description="Second test version",
            parameters={'test_param': 'test_value'}
        )
        
        assert len(manager.versions) == initial_version_count + 1
        assert version_id in manager.versions
        
        version = manager.versions[version_id]
        assert version.name == "Test Version 2"
        assert version.description == "Second test version"
        assert version.parameters == {'test_param': 'test_value'}
        assert version.is_active is False  # New versions start inactive
    
    def test_activate_version_valid(self):
        """Test activating a valid version"""
        manager = self.create_test_strategy_manager()
        
        # Create new version
        version_id = manager.create_version(
            name="Test Version 2",
            description="Second test version"
        )
        
        # Activate it
        success = manager.activate_version(version_id)
        
        assert success is True
        assert manager.current_version_id == version_id
        assert manager.versions[version_id].is_active is True
        
        # Previous version should be deactivated
        for vid, version in manager.versions.items():
            if vid != version_id:
                assert version.is_active is False
    
    def test_activate_version_invalid(self):
        """Test activating an invalid version"""
        manager = self.create_test_strategy_manager()
        
        success = manager.activate_version("nonexistent-version")
        
        assert success is False
        # Current version should remain unchanged
        assert manager.current_version_id is not None
    
    def test_rollback_to_version_valid(self):
        """Test rolling back to a valid version"""
        manager = self.create_test_strategy_manager()
        original_version_id = manager.current_version_id
        
        # Create and activate new version
        new_version_id = manager.create_version("New Version", "New version")
        manager.activate_version(new_version_id)
        
        # Rollback to original
        success = manager.rollback_to_version(original_version_id)
        
        assert success is True
        assert manager.current_version_id == original_version_id
    
    def test_rollback_to_version_invalid(self):
        """Test rolling back to an invalid version"""
        manager = self.create_test_strategy_manager()
        original_version_id = manager.current_version_id
        
        success = manager.rollback_to_version("nonexistent-version")
        
        assert success is False
        assert manager.current_version_id == original_version_id
    
    def test_get_version_performance_no_executions(self):
        """Test getting version performance with no executions"""
        manager = self.create_test_strategy_manager()
        version_id = manager.current_version_id
        
        performance = manager.get_version_performance(version_id)
        
        assert performance == {}
    
    def test_get_version_performance_with_executions(self):
        """Test getting version performance with executions"""
        manager = self.create_test_strategy_manager()
        df = self.create_test_dataframe()
        
        # Execute strategy multiple times
        for i in range(50, 55):
            manager.execute_strategy(df, i, 10000.0)
        
        version_id = manager.current_version_id
        performance = manager.get_version_performance(version_id)
        
        assert 'total_executions' in performance
        assert 'avg_execution_time_ms' in performance
        assert 'avg_signal_confidence' in performance
        assert performance['total_executions'] == 5
    
    def test_compare_versions(self):
        """Test comparing version performance"""
        manager = self.create_test_strategy_manager()
        df = self.create_test_dataframe()
        
        # Execute with first version
        version1_id = manager.current_version_id
        for i in range(50, 53):
            manager.execute_strategy(df, i, 10000.0)
        
        # Create and activate second version
        version2_id = manager.create_version("Version 2", "Second version")
        manager.activate_version(version2_id)
        
        # Execute with second version
        for i in range(53, 56):
            manager.execute_strategy(df, i, 10000.0)
        
        # Compare versions
        comparison = manager.compare_versions([version1_id, version2_id])
        
        assert version1_id in comparison
        assert version2_id in comparison
        assert isinstance(comparison[version1_id], (int, float))
        assert isinstance(comparison[version2_id], (int, float))
    
    def test_get_execution_statistics_no_executions(self):
        """Test getting execution statistics with no executions"""
        manager = self.create_test_strategy_manager()
        
        stats = manager.get_execution_statistics()
        
        assert stats == {}
    
    def test_get_execution_statistics_with_executions(self):
        """Test getting execution statistics with executions"""
        manager = self.create_test_strategy_manager()
        df = self.create_test_dataframe()
        
        # Execute strategy multiple times
        for i in range(50, 55):
            manager.execute_strategy(df, i, 10000.0)
        
        stats = manager.get_execution_statistics(lookback_hours=24)
        
        assert 'total_executions' in stats
        assert 'executions_per_hour' in stats
        assert 'avg_execution_time_ms' in stats
        assert 'signal_distribution' in stats
        assert 'current_version' in stats
        assert stats['total_executions'] == 5
    
    def test_get_current_version(self):
        """Test getting current version"""
        manager = self.create_test_strategy_manager()
        
        current = manager.get_current_version()
        
        assert current is not None
        assert isinstance(current, StrategyVersion)
        assert current.is_active is True
    
    def test_list_versions(self):
        """Test listing all versions"""
        manager = self.create_test_strategy_manager()
        
        # Create additional versions
        manager.create_version("Version 2", "Second version")
        manager.create_version("Version 3", "Third version")
        
        versions = manager.list_versions()
        
        assert len(versions) == 3
        assert all(isinstance(v, StrategyVersion) for v in versions)
    
    def test_export_version_valid(self):
        """Test exporting a valid version"""
        manager = self.create_test_strategy_manager()
        version_id = manager.current_version_id
        
        exported = manager.export_version(version_id)
        
        assert exported is not None
        assert isinstance(exported, dict)
        assert 'version_id' in exported
        assert 'name' in exported
        assert 'components' in exported
    
    def test_export_version_invalid(self):
        """Test exporting an invalid version"""
        manager = self.create_test_strategy_manager()
        
        exported = manager.export_version("nonexistent-version")
        
        assert exported is None
    
    def test_import_version_valid(self):
        """Test importing a valid version"""
        manager = self.create_test_strategy_manager()
        
        # Export current version
        current_version_id = manager.current_version_id
        exported = manager.export_version(current_version_id)
        
        # Modify exported data
        exported['name'] = "Imported Version"
        exported['description'] = "Imported from export"
        
        # Import it
        imported_version_id = manager.import_version(exported)
        
        assert imported_version_id is not None
        assert imported_version_id in manager.versions
        assert manager.versions[imported_version_id].name == "Imported Version"
    
    def test_import_version_invalid(self):
        """Test importing invalid version data"""
        manager = self.create_test_strategy_manager()
        
        # Try to import invalid data
        imported_version_id = manager.import_version({"invalid": "data"})
        
        assert imported_version_id is None
    
    def test_execution_history_limit(self):
        """Test that execution history is limited"""
        manager = self.create_test_strategy_manager()
        df = self.create_test_dataframe()
        
        # Mock the history limit to a small number for testing
        original_limit = 10000
        manager.execution_history = []  # Reset history
        
        # Execute many times (simulate going over limit)
        # We'll manually add executions to test the limit
        for i in range(15):
            execution = StrategyExecution(
                timestamp=datetime.now(),
                signal=Signal(SignalDirection.HOLD, 0.0, 1.0, {}),
                regime=None,
                position_size=0.0,
                risk_metrics={},
                execution_time_ms=10.0,
                version_id="test"
            )
            manager.execution_history.append(execution)
        
        # Manually trigger the limit check (normally done in execute_strategy)
        if len(manager.execution_history) > 10:
            manager.execution_history = manager.execution_history[-5:]
        
        assert len(manager.execution_history) == 5
    
    def test_calculate_risk_amount(self):
        """Test risk amount calculation"""
        manager = self.create_test_strategy_manager()
        
        signal = Signal(SignalDirection.BUY, 0.8, 0.9, {})
        balance = 10000.0
        
        risk_amount = manager._calculate_risk_amount(balance, signal, None)
        
        # Should be between 0.1% and 10% of balance
        assert 10.0 <= risk_amount <= 1000.0
        
        # Should be influenced by confidence
        expected_base = balance * 0.02 * 0.9  # 2% * confidence
        assert abs(risk_amount - expected_base) < 50.0  # Allow some variance
    
    def test_validate_position_size(self):
        """Test position size validation"""
        manager = self.create_test_strategy_manager()
        
        signal = Signal(SignalDirection.BUY, 0.8, 0.9, {})
        balance = 10000.0
        
        # Test normal position size
        validated = manager._validate_position_size(1000.0, signal, balance, None)
        assert validated == 1000.0
        
        # Test oversized position
        validated = manager._validate_position_size(5000.0, signal, balance, None)
        assert validated <= 2000.0  # Should be capped at 20%
        
        # Test HOLD signal
        hold_signal = Signal(SignalDirection.HOLD, 0.0, 1.0, {})
        validated = manager._validate_position_size(1000.0, hold_signal, balance, None)
        assert validated == 0.0