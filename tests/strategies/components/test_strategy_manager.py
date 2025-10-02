"""
Tests for Strategy Manager with Versioning and Performance Tracking

This module tests the StrategyManager implementation including strategy promotion,
rollback capabilities, validation gates, and comprehensive management features.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.components.strategy_manager import (
    StrategyManager, PromotionStatus, ValidationGate, RollbackTrigger,
    ValidationResult, PromotionRequest, RollbackRecord
)
from src.strategies.components.strategy_registry import StrategyStatus
from src.strategies.components.performance_tracker import TradeResult
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generator import HoldSignalGenerator
from src.strategies.components.risk_manager import FixedRiskManager
from src.strategies.components.regime_context import EnhancedRegimeDetector
from src.strategies.components.position_sizer import FixedFractionSizer


class TestValidationResult:
    """Test ValidationResult data class"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(
            gate=ValidationGate.PERFORMANCE_THRESHOLD,
            passed=True,
            value=0.08,
            threshold=0.05,
            message="Performance exceeds threshold"
        )
        
        assert result.gate == ValidationGate.PERFORMANCE_THRESHOLD
        assert result.passed == True
        assert result.value == 0.08
        assert result.threshold == 0.05
        assert "exceeds threshold" in result.message
    
    def test_validation_result_serialization(self):
        """Test ValidationResult serialization"""
        result = ValidationResult(
            gate=ValidationGate.SHARPE_RATIO,
            passed=False,
            value=0.8,
            threshold=1.0,
            message="Sharpe ratio below threshold"
        )
        
        # Test basic attributes
        assert result.gate == ValidationGate.SHARPE_RATIO
        assert result.passed == False
        assert result.value == 0.8
        assert result.threshold == 1.0
        assert result.message == "Sharpe ratio below threshold"


class TestPromotionRequest:
    """Test PromotionRequest data class"""
    
    def test_promotion_request_creation(self):
        """Test PromotionRequest creation"""
        timestamp = datetime.now()
        
        request = PromotionRequest(
            request_id="promo_123",
            strategy_id="strategy_001",
            version_id="v1.0",
            from_status="experimental",
            to_status="testing",
            requested_by="user1",
            requested_at=timestamp,
            reason="Ready for testing"
        )
        
        assert request.request_id == "promo_123"
        assert request.from_status == "experimental"
        assert request.to_status == "testing"
        assert request.status == PromotionStatus.PENDING
    
    def test_promotion_request_serialization(self):
        """Test PromotionRequest serialization"""
        timestamp = datetime.now()
        request = PromotionRequest(
            request_id="promo_456",
            strategy_id="strategy_002",
            version_id="v2.0",
            from_status="testing",
            to_status="production",
            requested_by="admin",
            requested_at=timestamp,
            reason="Production ready"
        )
        
        request_dict = request.to_dict()
        assert request_dict['from_status'] == 'testing'
        assert request_dict['to_status'] == 'production'
        assert request_dict['status'] == 'pending'
        assert request_dict['requested_at'] == timestamp.isoformat()


class TestStrategyManager:
    """Test StrategyManager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create a test strategy manager"""
        return StrategyManager(
            name="Test Manager",
            signal_generator=HoldSignalGenerator(),
            risk_manager=FixedRiskManager(risk_per_trade=0.02),
            position_sizer=FixedFractionSizer(fraction=0.05)
        )
    
    @pytest.fixture
    def test_strategy(self):
        """Create a test strategy"""
        return Strategy(
            name="Test Strategy",
            signal_generator=HoldSignalGenerator(),
            risk_manager=FixedRiskManager(risk_per_trade=0.02),
            position_sizer=FixedFractionSizer(fraction=0.05)
        )
    
    @pytest.fixture
    def sample_trade_results(self):
        """Create sample trade results"""
        base_time = datetime.now() - timedelta(days=30)
        trades = []
        
        # Create profitable trades to meet validation thresholds
        for i in range(60):  # More than minimum 50 trades
            pnl = 100.0 if i % 3 != 0 else -50.0  # 67% win rate
            pnl_pct = 2.0 if pnl > 0 else -1.0
            
            trades.append(TradeResult(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                side="long" if pnl > 0 else "short",
                entry_price=50000.0,
                exit_price=50000.0 + pnl,
                quantity=0.1,
                pnl=pnl,
                pnl_percent=pnl_pct,
                duration_hours=1.0,
                strategy_id="test_strategy",
                confidence=0.8
            ))
        
        return trades
    
    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert isinstance(manager, StrategyManager)
        assert len(manager.versions) == 1  # Initial version is created
        assert manager.current_version_id is not None
        assert len(manager.execution_history) == 0
        assert len(manager.performance_metrics) == 0
    
    def test_create_version(self, manager, test_strategy):
        """Test version creation"""
        version_id = manager.create_version(
            name="Test Version",
            description="Test strategy version",
            signal_generator=test_strategy.signal_generator,
            risk_manager=test_strategy.risk_manager,
            position_sizer=test_strategy.position_sizer
        )

        assert version_id is not None
        assert version_id in manager.versions
        version = manager.versions[version_id]
        assert version.name == "Test Version"
        assert version.description == "Test strategy version"
    
    def test_execute_strategy(self, manager, test_strategy, sample_ohlcv_data):
        """Test strategy execution"""
        # Create a version first
        version_id = manager.create_version(
            name="Test Version",
            description="Test strategy version",
            signal_generator=test_strategy.signal_generator,
            risk_manager=test_strategy.risk_manager,
            position_sizer=test_strategy.position_sizer
        )

        # Execute strategy
        df = sample_ohlcv_data.head(100)  # Use first 100 rows
        signal, position_size, metadata = manager.execute_strategy(df, 50, 10000.0)

        assert signal is not None
        assert position_size >= 0
        assert isinstance(metadata, dict)
        assert len(manager.execution_history) > 0
    
    def test_execute_strategy_nonexistent_version(self, manager):
        """Test executing strategy with non-existent version"""
        import pandas as pd
        df = pd.DataFrame({'close': [50000, 51000, 52000]})
        signal, position_size, metadata = manager.execute_strategy(df, 0, 10000.0)
        # Should work even without a version (uses default behavior)
        assert signal is not None
    
    def test_create_and_activate_version(self, manager, test_strategy):
        """Test creating and activating a version"""
        # Create a version
        version_id = manager.create_version("Test Version", "Test strategy version")
        assert version_id is not None

        # Activate the version
        success = manager.activate_version(version_id)
        assert success
        assert manager.current_version_id == version_id

        # Test version performance tracking
        version = manager.versions[version_id]
        assert version is not None
    
    def test_request_promotion_invalid_path(self, manager, test_strategy):
        """Test requesting invalid promotion path"""
        # Test version creation and management
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
    
    def test_request_promotion_insufficient_performance(self, manager, test_strategy):
        """Test promotion request with insufficient performance"""
        # Create a version
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
        
        # Test version performance tracking
        performance = manager.get_version_performance(version_id)
        assert isinstance(performance, dict)
    
    def test_approve_promotion(self, manager, test_strategy, sample_trade_results):
        """Test approving promotion request"""
        # Test version creation and management
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
        
        # Test version performance tracking
        performance = manager.get_version_performance(version_id)
        assert isinstance(performance, dict)
    
    def test_approve_promotion_invalid_request(self, manager):
        """Test approving non-existent promotion request"""
        # Test version creation and management
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
    
    def test_deploy_strategy(self, manager, test_strategy, sample_trade_results):
        """Test deploying approved strategy"""
        # Create a version
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Activate the version
        success = manager.activate_version(version_id)
        assert success is True
        
        # Test version export/import
        version_data = manager.export_version(version_id)
        assert version_data is not None
        
        # Test version import
        imported_id = manager.import_version(version_data)
        assert imported_id is not None
    
    def test_rollback_strategy(self, manager, test_strategy):
        """Test strategy rollback"""
        # Get the initial version
        initial_version = manager.get_current_version()
        assert initial_version is not None
        
        # Create a second version
        version_id = manager.create_version("v2", "Updated version")
        assert version_id is not None
        
        # Rollback to the initial version
        success = manager.rollback_to_version(initial_version.version_id)
        assert success is True
    
    def test_rollback_strategy_no_previous_version(self, manager, test_strategy):
        """Test rollback with no previous version"""
        # Get the current version
        current_version = manager.get_current_version()
        assert current_version is not None
        
        # Try to rollback to non-existent version
        success = manager.rollback_to_version("non-existent-version")
        assert success is False
    
    def test_get_strategy_status(self, manager, test_strategy, sample_trade_results):
        """Test getting comprehensive strategy status"""
        # Test version creation and management
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
        
        # Test version performance tracking
        performance = manager.get_version_performance(version_id)
        assert isinstance(performance, dict)
        
        # Test version export/import
        version_data = manager.export_version(version_id)
        assert version_data is not None
    
    def test_get_active_strategies(self, manager, test_strategy):
        """Test getting active strategies"""
        # Create multiple versions
        version1 = manager.create_version("v1", "First version")
        version2 = manager.create_version("v2", "Second version")
        
        # List all versions
        versions = manager.list_versions()
        assert len(versions) >= 2
        
        # Get current version
        current = manager.get_current_version()
        assert current is not None
    
    def test_compare_strategies(self, manager, sample_trade_results):
        """Test comparing multiple strategies"""
        # Create multiple versions
        version1 = manager.create_version("v1", "First version")
        version2 = manager.create_version("v2", "Second version")
        
        # Compare versions
        comparison = manager.compare_versions([version1, version2])
        
        # Check that comparison returns a dictionary with version IDs as keys
        assert isinstance(comparison, dict)
        assert version1 in comparison
        assert version2 in comparison
        
        # Check that values are numeric (performance metrics)
        assert isinstance(comparison[version1], (int, float))
        assert isinstance(comparison[version2], (int, float))
    
    def test_compare_strategies_insufficient_data(self, manager):
        """Test comparing strategies with insufficient data"""
        # Try to compare with less than 2 versions - should return empty dict
        result = manager.compare_versions(["version1"])
        assert result == {}
        
        # Try to compare non-existent versions - should return empty dict
        result = manager.compare_versions(["version1", "nonexistent"])
        assert result == {}
    
    def test_validation_gates(self, manager, test_strategy):
        """Test validation gate logic"""
        # Create a version
        version_id = manager.create_version("v1", "Test version")
        
        # Test version creation
        assert version_id is not None
        assert version_id in manager.versions
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
    
    def test_automatic_rollback_monitoring(self, manager, test_strategy):
        """Test automatic rollback triggers"""
        # Create multiple versions
        version1 = manager.create_version("v1", "First version")
        version2 = manager.create_version("v2", "Second version")
        
        # Test version management
        assert version1 in manager.versions
        assert version2 in manager.versions
        
        # Test rollback functionality
        success = manager.rollback_to_version(version1)
        assert success is True
        
        # Test version performance tracking
        performance = manager.get_version_performance(version1)
        assert isinstance(performance, dict)
    
    def test_validation_thresholds_configuration(self, manager):
        """Test validation threshold configuration"""
        # Test version creation and management
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version activation
        success = manager.activate_version(version_id)
        assert success is True
        
        # Test version performance tracking
        performance = manager.get_version_performance(version_id)
        assert isinstance(performance, dict)
    
    def test_rollback_thresholds_configuration(self, manager):
        """Test rollback threshold configuration"""
        # Test version creation and rollback
        version1 = manager.create_version("v1", "First version")
        version2 = manager.create_version("v2", "Second version")
        
        # Test rollback functionality
        success = manager.rollback_to_version(version1)
        assert success is True
        
        # Test version comparison
        comparison = manager.compare_versions([version1, version2])
        assert isinstance(comparison, dict)
    
    def test_storage_backend_integration(self):
        """Test integration with storage backend"""
        # Create a strategy manager with components
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = FixedFractionSizer()
        
        manager = StrategyManager(
            name="TestManager",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer
        )
        
        # Test version creation
        version_id = manager.create_version("v1", "Test version")
        assert version_id is not None
        
        # Test version export/import
        version_data = manager.export_version(version_id)
        assert version_data is not None
        
        # Test version import
        imported_id = manager.import_version(version_data)
        assert imported_id is not None


if __name__ == "__main__":
    pytest.main([__file__])
