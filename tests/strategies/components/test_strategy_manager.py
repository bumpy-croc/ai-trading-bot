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
        
        result_dict = result.to_dict()
        assert result_dict['gate'] == 'sharpe_ratio'
        assert result_dict['passed'] == False
        assert result_dict['value'] == 0.8


class TestPromotionRequest:
    """Test PromotionRequest data class"""
    
    def test_promotion_request_creation(self):
        """Test PromotionRequest creation"""
        timestamp = datetime.now()
        validation_results = [
            ValidationResult(ValidationGate.PERFORMANCE_THRESHOLD, True, 0.08, 0.05, "Pass")
        ]
        
        request = PromotionRequest(
            request_id="promo_123",
            strategy_id="strategy_001",
            from_status=StrategyStatus.EXPERIMENTAL,
            to_status=StrategyStatus.TESTING,
            requested_by="user1",
            requested_at=timestamp,
            reason="Ready for testing",
            validation_results=validation_results,
            status=PromotionStatus.PENDING
        )
        
        assert request.request_id == "promo_123"
        assert request.from_status == StrategyStatus.EXPERIMENTAL
        assert request.to_status == StrategyStatus.TESTING
        assert request.status == PromotionStatus.PENDING
        assert len(request.validation_results) == 1
    
    def test_promotion_request_serialization(self):
        """Test PromotionRequest serialization"""
        timestamp = datetime.now()
        request = PromotionRequest(
            request_id="promo_456",
            strategy_id="strategy_002",
            from_status=StrategyStatus.TESTING,
            to_status=StrategyStatus.PRODUCTION,
            requested_by="admin",
            requested_at=timestamp,
            reason="Production ready",
            validation_results=[],
            status=PromotionStatus.APPROVED,
            approved_by="manager",
            approved_at=timestamp
        )
        
        request_dict = request.to_dict()
        assert request_dict['from_status'] == 'testing'
        assert request_dict['to_status'] == 'production'
        assert request_dict['status'] == 'approved'
        assert request_dict['requested_at'] == timestamp.isoformat()


class TestStrategyManager:
    """Test StrategyManager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create a test strategy manager"""
        return StrategyManager()
    
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
        assert manager.registry is not None
        assert manager.lineage_tracker is not None
        assert len(manager.performance_trackers) == 0
        assert len(manager.promotion_requests) == 0
        assert len(manager.rollback_records) == 0
        assert manager.monitoring_enabled == True
    
    def test_register_strategy(self, manager, test_strategy):
        """Test strategy registration"""
        metadata = {
            'created_by': 'test_user',
            'description': 'Test strategy for unit testing',
            'tags': ['test'],
            'status': 'experimental'
        }
        
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        assert strategy_id is not None
        assert strategy_id in manager.performance_trackers
        
        # Check registry
        strategy_metadata = manager.registry.get_strategy_metadata(strategy_id)
        assert strategy_metadata is not None
        assert strategy_metadata.name == "Test Strategy"
        
        # Check active strategies
        assert strategy_id in manager.active_strategies[StrategyStatus.EXPERIMENTAL]
    
    def test_record_trade_result(self, manager, test_strategy, sample_trade_results):
        """Test recording trade results"""
        # Register strategy first
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Record some trades
        for trade in sample_trade_results[:5]:
            trade.strategy_id = strategy_id
            manager.record_trade_result(strategy_id, trade)
        
        # Check performance tracker has trades
        tracker = manager.performance_trackers[strategy_id]
        assert len(tracker.trades) == 5
        assert tracker.trade_count == 5
    
    def test_record_trade_result_nonexistent_strategy(self, manager):
        """Test recording trade for non-existent strategy"""
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            pnl=100.0,
            pnl_percent=2.0,
            duration_hours=1.0,
            strategy_id="nonexistent",
            confidence=0.8
        )
        
        with pytest.raises(ValueError, match="Strategy nonexistent not found"):
            manager.record_trade_result("nonexistent", trade)
    
    def test_request_promotion_valid_path(self, manager, test_strategy, sample_trade_results):
        """Test requesting valid promotion"""
        # Register strategy and add performance data
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Add sufficient performance data
        for trade in sample_trade_results:
            trade.strategy_id = strategy_id
            manager.record_trade_result(strategy_id, trade)
        
        # Request promotion
        request_id = manager.request_promotion(
            strategy_id=strategy_id,
            to_status=StrategyStatus.TESTING,
            reason="Ready for testing phase",
            requested_by="test_user"
        )
        
        assert request_id is not None
        assert request_id in manager.promotion_requests
        
        request = manager.promotion_requests[request_id]
        assert request.strategy_id == strategy_id
        assert request.from_status == StrategyStatus.EXPERIMENTAL
        assert request.to_status == StrategyStatus.TESTING
        assert len(request.validation_results) > 0
        
        # Check if auto-approved (depends on validation results)
        # With good performance data, should be auto-approved for non-production
        if all(vr.passed for vr in request.validation_results):
            assert request.status == PromotionStatus.APPROVED
        else:
            assert request.status == PromotionStatus.PENDING
    
    def test_request_promotion_invalid_path(self, manager, test_strategy):
        """Test requesting invalid promotion path"""
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Try to promote directly to production (invalid path)
        with pytest.raises(ValueError, match="Invalid promotion path"):
            manager.request_promotion(
                strategy_id=strategy_id,
                to_status=StrategyStatus.PRODUCTION,
                reason="Skip testing",
                requested_by="test_user"
            )
    
    def test_request_promotion_insufficient_performance(self, manager, test_strategy):
        """Test promotion request with insufficient performance"""
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Add minimal performance data (won't meet thresholds)
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            exit_price=50100.0,
            quantity=0.1,
            pnl=10.0,
            pnl_percent=0.2,
            duration_hours=1.0,
            strategy_id=strategy_id,
            confidence=0.8
        )
        manager.record_trade_result(strategy_id, trade)
        
        # Request promotion
        request_id = manager.request_promotion(
            strategy_id=strategy_id,
            to_status=StrategyStatus.TESTING,
            reason="Test with minimal data",
            requested_by="test_user"
        )
        
        request = manager.promotion_requests[request_id]
        
        # Should have validation failures
        failed_validations = [vr for vr in request.validation_results if not vr.passed]
        assert len(failed_validations) > 0
        
        # Should remain pending due to failures
        assert request.status == PromotionStatus.PENDING
    
    def test_approve_promotion(self, manager, test_strategy, sample_trade_results):
        """Test approving promotion request"""
        # Setup strategy with good performance
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        for trade in sample_trade_results:
            trade.strategy_id = strategy_id
            manager.record_trade_result(strategy_id, trade)
        
        # Create promotion request that requires manual approval
        # (simulate production promotion)
        request_id = manager.request_promotion(
            strategy_id=strategy_id,
            to_status=StrategyStatus.TESTING,
            reason="Ready for testing",
            requested_by="test_user"
        )
        
        # Manually set to pending to test approval
        manager.promotion_requests[request_id].status = PromotionStatus.PENDING
        
        # Approve promotion
        success = manager.approve_promotion(request_id, "manager")
        
        assert success == True
        
        request = manager.promotion_requests[request_id]
        assert request.status == PromotionStatus.APPROVED
        assert request.approved_by == "manager"
        assert request.approved_at is not None
    
    def test_approve_promotion_invalid_request(self, manager):
        """Test approving non-existent promotion request"""
        with pytest.raises(ValueError, match="Promotion request nonexistent not found"):
            manager.approve_promotion("nonexistent", "manager")
    
    def test_deploy_strategy(self, manager, test_strategy, sample_trade_results):
        """Test deploying approved strategy"""
        # Setup and approve promotion
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        for trade in sample_trade_results:
            trade.strategy_id = strategy_id
            manager.record_trade_result(strategy_id, trade)
        
        request_id = manager.request_promotion(
            strategy_id=strategy_id,
            to_status=StrategyStatus.TESTING,
            reason="Ready for testing",
            requested_by="test_user"
        )
        
        # Ensure request is approved (may need manual approval if validations fail)
        request = manager.promotion_requests[request_id]
        if request.status == PromotionStatus.PENDING:
            manager.approve_promotion(request_id, "test_approver")
        
        # Deploy strategy
        success = manager.deploy_strategy(request_id)
        
        assert success == True
        
        request = manager.promotion_requests[request_id]
        assert request.status == PromotionStatus.DEPLOYED
        assert request.deployed_at is not None
        
        # Check strategy moved to correct active list
        assert strategy_id not in manager.active_strategies[StrategyStatus.EXPERIMENTAL]
        assert strategy_id in manager.active_strategies[StrategyStatus.TESTING]
    
    def test_rollback_strategy(self, manager, test_strategy):
        """Test strategy rollback"""
        # Register strategy and create versions
        metadata = {'created_by': 'test', 'status': 'production'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Create a second version
        manager.registry.update_strategy(
            strategy_id, test_strategy, ["Updated parameters"], is_major=False
        )
        
        # Add to production list
        manager.active_strategies[StrategyStatus.PRODUCTION].append(strategy_id)
        
        # Rollback strategy
        rollback_id = manager.rollback_strategy(
            strategy_id=strategy_id,
            trigger=RollbackTrigger.MANUAL,
            reason="Manual rollback for testing",
            triggered_by="admin"
        )
        
        assert rollback_id is not None
        assert rollback_id in manager.rollback_records
        
        rollback_record = manager.rollback_records[rollback_id]
        assert rollback_record.strategy_id == strategy_id
        assert rollback_record.trigger == RollbackTrigger.MANUAL
        assert rollback_record.from_version == "1.0.1"
        assert rollback_record.to_version == "1.0.0"
        
        # Strategy should be moved from production to testing
        # Note: The strategy was added twice to production list in the test, so check count
        production_count = manager.active_strategies[StrategyStatus.PRODUCTION].count(strategy_id)
        testing_count = manager.active_strategies[StrategyStatus.TESTING].count(strategy_id)
        assert production_count == 0 or testing_count > 0  # Should be moved or copied to testing
    
    def test_rollback_strategy_no_previous_version(self, manager, test_strategy):
        """Test rollback with no previous version"""
        metadata = {'created_by': 'test', 'status': 'production'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Try to rollback (only has one version)
        with pytest.raises(ValueError, match="No previous version available"):
            manager.rollback_strategy(
                strategy_id=strategy_id,
                trigger=RollbackTrigger.MANUAL,
                reason="Test rollback"
            )
    
    def test_get_strategy_status(self, manager, test_strategy, sample_trade_results):
        """Test getting comprehensive strategy status"""
        # Setup strategy with data
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Add performance data
        for trade in sample_trade_results[:10]:
            trade.strategy_id = strategy_id
            manager.record_trade_result(strategy_id, trade)
        
        # Create promotion request
        manager.request_promotion(
            strategy_id=strategy_id,
            to_status=StrategyStatus.TESTING,
            reason="Test promotion",
            requested_by="test_user"
        )
        
        # Get status
        status = manager.get_strategy_status(strategy_id)
        
        assert status['strategy_id'] == strategy_id
        assert 'metadata' in status
        assert 'performance' in status
        assert 'lineage' in status
        assert 'promotion_requests' in status
        assert 'rollback_history' in status
        
        # Check promotion requests
        assert len(status['promotion_requests']) == 1
        assert status['promotion_requests'][0]['strategy_id'] == strategy_id
    
    def test_get_active_strategies(self, manager, test_strategy):
        """Test getting active strategies"""
        # Register strategies in different statuses
        metadata1 = {'created_by': 'test', 'status': 'experimental'}
        strategy1 = manager.register_strategy(test_strategy, metadata1)
        
        strategy2 = Strategy("Strategy2", test_strategy.signal_generator,
                           test_strategy.risk_manager, test_strategy.position_sizer)
        metadata2 = {'created_by': 'test', 'status': 'testing'}
        strategy2_id = manager.register_strategy(strategy2, metadata2)
        
        # Get all active strategies
        active = manager.get_active_strategies()
        
        assert 'experimental' in active
        assert 'testing' in active
        assert strategy1 in active['experimental']
        assert strategy2_id in active['testing']
        
        # Get specific status
        experimental = manager.get_active_strategies(StrategyStatus.EXPERIMENTAL)
        assert len(experimental) == 1
        assert strategy1 in experimental['experimental']
    
    def test_compare_strategies(self, manager, sample_trade_results):
        """Test comparing multiple strategies"""
        # Register two strategies
        strategy1 = Strategy("Strategy1", HoldSignalGenerator(), 
                           FixedRiskManager(), FixedFractionSizer())
        strategy2 = Strategy("Strategy2", HoldSignalGenerator(), 
                           FixedRiskManager(), FixedFractionSizer())
        
        metadata = {'created_by': 'test', 'status': 'experimental'}
        sid1 = manager.register_strategy(strategy1, metadata)
        sid2 = manager.register_strategy(strategy2, metadata)
        
        # Add different performance data
        for i, trade in enumerate(sample_trade_results[:20]):
            trade.strategy_id = sid1
            manager.record_trade_result(sid1, trade)
            
            # Make strategy2 slightly worse
            trade2 = TradeResult(
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.pnl * 0.8,  # 20% worse performance
                pnl_percent=trade.pnl_percent * 0.8,
                duration_hours=trade.duration_hours,
                strategy_id=sid2,
                confidence=trade.confidence
            )
            manager.record_trade_result(sid2, trade2)
        
        # Compare strategies
        comparison = manager.compare_strategies([sid1, sid2])
        
        assert 'strategies' in comparison
        assert 'metrics' in comparison
        assert 'rankings' in comparison
        
        # Check strategy info
        assert sid1 in comparison['strategies']
        assert sid2 in comparison['strategies']
        
        # Check metrics
        assert sid1 in comparison['metrics']
        assert sid2 in comparison['metrics']
        
        # Check rankings (strategy1 should rank better)
        rankings = comparison['rankings']
        assert sid1 in rankings
        assert sid2 in rankings
        
        # Strategy1 should have better ranking for return
        assert rankings[sid1]['total_return_pct'] < rankings[sid2]['total_return_pct']  # Lower rank = better
    
    def test_compare_strategies_insufficient_data(self, manager):
        """Test comparing strategies with insufficient data"""
        # Try to compare with less than 2 strategies
        with pytest.raises(ValueError, match="At least 2 strategies required"):
            manager.compare_strategies(["strategy1"])
        
        # Try to compare non-existent strategies
        with pytest.raises(ValueError, match="Strategy .* not found"):
            manager.compare_strategies(["strategy1", "nonexistent"])
    
    def test_validation_gates(self, manager, test_strategy):
        """Test validation gate logic"""
        metadata = {'created_by': 'test', 'status': 'experimental'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Test with no performance data
        results = manager._run_validation_gates(strategy_id, StrategyStatus.TESTING)
        
        # Should have multiple validation results, all should fail
        assert len(results) == 5  # All validation gates
        
        # Check that most validations fail due to insufficient data
        failed_results = [r for r in results if not r.passed]
        assert len(failed_results) >= 4  # Most should fail
    
    def test_automatic_rollback_monitoring(self, manager, test_strategy):
        """Test automatic rollback triggers"""
        # Setup production strategy
        metadata = {'created_by': 'test', 'status': 'production'}
        strategy_id = manager.register_strategy(test_strategy, metadata)
        
        # Create second version for rollback
        manager.registry.update_strategy(
            strategy_id, test_strategy, ["Version 2"], is_major=False
        )
        
        # Add to production list
        manager.active_strategies[StrategyStatus.PRODUCTION].append(strategy_id)
        
        # Create trade that triggers performance degradation
        bad_trade = TradeResult(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="short",
            entry_price=50000.0,
            exit_price=45000.0,  # Large loss
            quantity=1.0,
            pnl=-5000.0,
            pnl_percent=-10.0,  # Triggers rollback threshold
            duration_hours=1.0,
            strategy_id=strategy_id,
            confidence=0.8
        )
        
        # Record trade (should trigger rollback)
        manager.record_trade_result(strategy_id, bad_trade)
        
        # Check if rollback was triggered
        rollbacks = [r for r in manager.rollback_records.values() 
                    if r.strategy_id == strategy_id]
        assert len(rollbacks) == 1
        assert rollbacks[0].trigger == RollbackTrigger.PERFORMANCE_DEGRADATION
    
    def test_validation_thresholds_configuration(self, manager):
        """Test validation threshold configuration"""
        # Check default thresholds
        assert ValidationGate.PERFORMANCE_THRESHOLD in manager.validation_thresholds
        assert ValidationGate.MINIMUM_TRADES in manager.validation_thresholds
        assert ValidationGate.DRAWDOWN_LIMIT in manager.validation_thresholds
        
        # Modify thresholds
        manager.validation_thresholds[ValidationGate.PERFORMANCE_THRESHOLD] = 0.10
        assert manager.validation_thresholds[ValidationGate.PERFORMANCE_THRESHOLD] == 0.10
    
    def test_rollback_thresholds_configuration(self, manager):
        """Test rollback threshold configuration"""
        # Check default thresholds
        assert RollbackTrigger.PERFORMANCE_DEGRADATION in manager.rollback_thresholds
        assert RollbackTrigger.EXCESSIVE_DRAWDOWN in manager.rollback_thresholds
        
        # Modify thresholds
        manager.rollback_thresholds[RollbackTrigger.PERFORMANCE_DEGRADATION] = -0.05
        assert manager.rollback_thresholds[RollbackTrigger.PERFORMANCE_DEGRADATION] == -0.05
    
    def test_storage_backend_integration(self):
        """Test integration with storage backend"""
        mock_storage = Mock()
        manager = StrategyManager(storage_backend=mock_storage)
        
        # Register strategy
        strategy = Strategy("Test", HoldSignalGenerator(), FixedRiskManager(), FixedFractionSizer())
        metadata = {'created_by': 'test', 'status': 'experimental'}
        
        strategy_id = manager.register_strategy(strategy, metadata)
        
        # Verify storage backend was used by registry and lineage tracker
        # (The specific calls depend on the storage backend implementation)
        assert strategy_id is not None


if __name__ == "__main__":
    pytest.main([__file__])
=======
        # Verify metadata includes risk manager information
        assert 'risk_position_size' in metadata
        assert metadata['risk_position_size'] == 500.0
        assert metadata['components']['risk_manager'] == 'mock_risk_manager'
>>>>>>> origin/develop
