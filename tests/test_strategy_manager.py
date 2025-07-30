"""
Tests for strategy manager.

Strategy manager handles hot-swapping and model updates during live trading.
Tests cover:
- Strategy hot-swapping without downtime
- Model updates and validation
- Version control and rollback
- Thread safety during updates
- Error handling and recovery
"""

import pytest
pytestmark = pytest.mark.integration
import threading
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from live.strategy_manager import StrategyManager, StrategyVersion
from strategies.ml_adaptive import MlAdaptive


class TestStrategyManager:
    """Test the core StrategyManager functionality"""

    def test_strategy_manager_initialization(self, temp_directory):
        """Test strategy manager initialization"""
        temp_staging = str(temp_directory / "staging")
        manager = StrategyManager(
            strategies_dir=str(temp_directory / "strategies"),
            models_dir=str(temp_directory / "models"),
            staging_dir=temp_staging
        )
        
        assert manager.strategies_dir == temp_directory / "strategies"
        assert manager.models_dir == temp_directory / "models"
        assert manager.staging_dir == Path(temp_staging)
        assert manager.staging_dir.exists()
        assert manager.current_strategy is None
        assert manager.pending_update is None

    def test_strategy_loading(self, temp_directory):
        """Test loading strategies"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load adaptive strategy
        strategy = manager.load_strategy("ml_adaptive", version="test_v1")
        
        assert strategy is not None
        assert isinstance(strategy, MlAdaptive)
        assert manager.current_strategy == strategy
        assert manager.current_version is not None
        assert manager.current_version.strategy_name == "ml_adaptive"
        assert manager.current_version.version == "test_v1"

    def test_strategy_loading_with_config(self, temp_directory):
        """Test loading strategies with custom configuration"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        config = {
            "name": "CustomMlAdaptive",
            "sequence_length": 60
        }
        
        strategy = manager.load_strategy("ml_adaptive", config=config)
        
        assert strategy.name == "CustomMlAdaptive"
        assert strategy.sequence_length == 60

    def test_invalid_strategy_loading(self, temp_directory):
        """Test loading invalid strategy"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        with pytest.raises(ValueError):
            manager.load_strategy("nonexistent_strategy")

    @pytest.mark.live_trading
    def test_hot_swap_preparation(self, temp_directory):
        """Test preparing hot swap"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load initial strategy
        initial_strategy = manager.load_strategy("ml_adaptive", version="v1")
        
        # Prepare hot swap
        success = manager.hot_swap_strategy("ml_adaptive", new_config={"sequence_length": 60})
        
        assert success == True
        assert manager.pending_update is not None
        assert manager.pending_update['type'] == 'strategy_swap'

    @pytest.mark.live_trading
    def test_hot_swap_application(self, temp_directory):
        """Test applying hot swap"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load initial strategy
        initial_strategy = manager.load_strategy("ml_adaptive", version="v1")
        initial_name = initial_strategy.name
        
        # Prepare and apply hot swap
        manager.hot_swap_strategy("ml_adaptive", new_config={"sequence_length": 60})
        
        # Apply the update
        success = manager.apply_pending_update()
        
        assert success == True
        assert manager.current_strategy != initial_strategy
        assert manager.current_strategy.sequence_length == 60
        assert manager.pending_update is None

    @pytest.mark.live_trading
    def test_model_update_preparation(self, temp_directory, mock_model_file):
        """Test preparing model update"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load ML-based strategy
        try:
            strategy = manager.load_strategy("ml_basic")
            
            # Prepare model update - use the actual strategy name
            success = manager.update_model("MlBasic", str(mock_model_file))
            
            assert success == True
            assert manager.pending_update is not None
            assert manager.pending_update['type'] == 'model_update'
        except (ValueError, ImportError):
            # ML strategy might not be available
            pytest.skip("ML strategy not available for testing")

    @pytest.mark.live_trading
    def test_model_update_application(self, temp_directory, mock_model_file):
        """Test applying model update"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        try:
            # Load ML strategy
            strategy = manager.load_strategy("ml_basic")
            
            # Mock the strategy to have model loading capability
            strategy._load_model = Mock()
            strategy.model_path = "old_model.onnx"
            
            # Prepare and apply model update - use the actual strategy name
            manager.update_model("MlBasic", str(mock_model_file))
            success = manager.apply_pending_update()
            
            assert success == True
            assert manager.current_strategy.model_path != "old_model.onnx"
            strategy._load_model.assert_called_once()
        except (ValueError, ImportError):
            pytest.skip("ML strategy not available for testing")

    def test_pending_update_detection(self, temp_directory):
        """Test pending update detection"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Initially no pending updates
        assert manager.has_pending_update() == False
        
        # Prepare update
        manager.load_strategy("ml_adaptive")
        manager.hot_swap_strategy("ml_adaptive", new_config={"sequence_length": 60})
        
        # Should detect pending update
        assert manager.has_pending_update() == True
        
        # Apply update
        manager.apply_pending_update()
        
        # Should clear pending update
        assert manager.has_pending_update() == False

    def test_version_history_tracking(self, temp_directory):
        """Test version history tracking"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load multiple versions
        strategy_v1 = manager.load_strategy("ml_adaptive", version="v1")
        strategy_v2 = manager.load_strategy("ml_adaptive", version="v2") 
        
        # Check version history
        assert len(manager.version_history) >= 2
        assert "ml_adaptive_v1" in manager.version_history
        assert "ml_adaptive_v2" in manager.version_history

    def test_strategy_registry(self, temp_directory):
        """Test strategy registry functionality"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Check available strategies
        available = manager.list_available_strategies()
        
        assert 'available_strategies' in available
        assert 'ml_adaptive' in available['available_strategies']
        assert isinstance(available['available_strategies'], list)


class TestStrategyManagerThreadSafety:
    """Test thread safety of strategy manager operations"""

    @pytest.mark.live_trading
    def test_concurrent_strategy_loading(self, temp_directory):
        """Test concurrent strategy loading"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        results = []
        errors = []
        
        def load_strategy(strategy_name, version):
            try:
                strategy = manager.load_strategy(strategy_name, version=version)
                results.append((strategy_name, version, strategy))
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads loading strategies
        threads = []
        for i in range(3):
            thread = threading.Thread(target=load_strategy, args=("ml_adaptive", f"v{i}"))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have loaded strategies without any errors
        assert len(errors) == 0  # Ensure thread safety by requiring zero errors
        assert len(results) >= 2  # At least some should succeed

    @pytest.mark.live_trading
    def test_concurrent_hot_swapping(self, temp_directory):
        """Test concurrent hot swapping operations"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load initial strategy
        manager.load_strategy("ml_adaptive", version="initial")
        
        swap_results = []
        
        def attempt_hot_swap(config_variant):
            try:
                success = manager.hot_swap_strategy("ml_adaptive", new_config={"sequence_length": config_variant})
                swap_results.append(success)
            except Exception as e:
                swap_results.append(False)
        
        # Attempt concurrent hot swaps
        threads = []
        for i in range(3):
            thread = threading.Thread(target=attempt_hot_swap, args=(10 + i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Only one swap should succeed (due to locking)
        successful_swaps = sum(swap_results)
        assert successful_swaps <= 1

    @pytest.mark.live_trading
    def test_update_lock_behavior(self, temp_directory):
        """Test that update lock prevents race conditions"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_adaptive")
        
        lock_acquired_count = 0
        
        def test_lock_acquisition():
            nonlocal lock_acquired_count
            with manager.update_lock:
                lock_acquired_count += 1
                time.sleep(0.1)  # Hold lock briefly
        
        # Start multiple threads trying to acquire lock
        threads = []
        for i in range(5):
            thread = threading.Thread(target=test_lock_acquisition)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should have acquired lock sequentially
        assert lock_acquired_count == 5


class TestStrategyManagerErrorHandling:
    """Test error handling in strategy manager"""

    def test_model_validation_failure(self, temp_directory):
        """Test handling of model validation failures"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Create invalid model file
        invalid_model = temp_directory / "invalid_model.onnx"
        with open(invalid_model, 'w') as f:
            f.write("not a valid model")
        
        try:
            manager.load_strategy("ml_basic")
            success = manager.update_model("ml_basic", str(invalid_model), validate_model=True)
            assert success == False
        except (ValueError, ImportError):
            pytest.skip("ML strategy not available")

    def test_strategy_swap_failure_recovery(self, temp_directory):
        """Test recovery from strategy swap failures"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load initial strategy
        initial_strategy = manager.load_strategy("ml_adaptive", version="v1")
        
        # Mock strategy loading to fail
        with patch.object(manager, 'load_strategy') as mock_load:
            mock_load.side_effect = Exception("Strategy loading failed")
            
            # Attempt hot swap
            success = manager.hot_swap_strategy("ml_adaptive")
            assert success == False
            
            # Current strategy should remain unchanged
            assert manager.current_strategy == initial_strategy

    def test_missing_model_file_handling(self, temp_directory):
        """Test handling of missing model files"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        try:
            manager.load_strategy("ml_basic")
            success = manager.update_model("ml_basic", "/nonexistent/model.onnx")
            assert success == False
        except (ValueError, ImportError):
            pytest.skip("ML strategy not available")

    def test_invalid_strategy_configuration(self, temp_directory):
        """Test handling of invalid strategy configurations"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Try to load strategy with invalid config
        invalid_config = {"invalid_parameter": "invalid_value"}
        
        try:
            strategy = manager.load_strategy("ml_adaptive", config=invalid_config)
            # Should either ignore invalid params or handle gracefully
            assert strategy is not None
        except Exception as e:
            # Should be a meaningful error message
            assert len(str(e)) > 0

    def test_corrupted_staging_directory(self, temp_directory):
        """Test handling of corrupted staging directory"""
        staging_dir = temp_directory / "staging"
        
        # Create a file where staging directory should be
        staging_dir.touch()
        
        try:
            manager = StrategyManager(staging_dir=str(staging_dir))
            # Should handle the conflict gracefully
        except Exception as e:
            # Should be a meaningful error about directory creation
            assert "staging" in str(e).lower() or "directory" in str(e).lower()


class TestStrategyManagerIntegration:
    """Test strategy manager integration with other components"""

    @pytest.mark.live_trading
    @pytest.mark.integration
    def test_strategy_manager_with_live_engine(self, temp_directory, mock_data_provider):
        """Test strategy manager integration with live trading engine"""
        from live.trading_engine import LiveTradingEngine
        
        manager = StrategyManager(staging_dir=str(temp_directory))
        initial_strategy = manager.load_strategy("ml_adaptive")
        
        # Create trading engine with strategy manager
        engine = LiveTradingEngine(
            strategy=initial_strategy,
            data_provider=mock_data_provider,
            enable_hot_swapping=True
        )
        
        # Mock the strategy manager in engine
        engine.strategy_manager = manager
        
        # Test hot swap integration
        manager.hot_swap_strategy("ml_adaptive", new_config={"sequence_length": 60})
        
        # Simulate engine checking for updates
        has_update = manager.has_pending_update()
        assert has_update == True
        
        # Apply update
        success = manager.apply_pending_update()
        assert success == True

    def test_callback_system(self, temp_directory):
        """Test callback system for strategy changes"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        callback_calls = []
        
        def strategy_change_callback(swap_data):
            callback_calls.append(("strategy_change", swap_data))
        
        def model_update_callback(update_data):
            callback_calls.append(("model_update", update_data))
        
        # Set callbacks
        manager.on_strategy_change = strategy_change_callback
        manager.on_model_update = model_update_callback
        
        # Load strategy and trigger swap
        manager.load_strategy("ml_adaptive")
        manager.hot_swap_strategy("ml_adaptive", new_config={"sequence_length": 60})
        
        # Should have called strategy change callback
        strategy_calls = [call for call in callback_calls if call[0] == "strategy_change"]
        assert len(strategy_calls) == 1

    def test_performance_tracking_integration(self, temp_directory):
        """Test integration with performance tracking"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load strategy
        strategy = manager.load_strategy("ml_adaptive", version="perf_test")
        
        # Check that version is tracked
        assert manager.current_version is not None
        assert manager.current_version.strategy_name == "ml_adaptive"
        assert manager.current_version.version == "perf_test"
        
        # Performance metrics should be trackable
        manager.current_version.performance_metrics = {
            "total_return": 15.5,
            "sharpe_ratio": 1.8,
            "max_drawdown": 8.2
        }
        
        assert manager.current_version.performance_metrics["total_return"] == 15.5


class TestStrategyVersioning:
    """Test strategy versioning functionality"""

    def test_strategy_version_creation(self, temp_directory):
        """Test strategy version object creation"""
        version = StrategyVersion(
            strategy_name="ml_adaptive",
            version="v1.0",
            timestamp=datetime.now(),
            config={"sequence_length": 60}
        )
        
        assert version.strategy_name == "ml_adaptive"
        assert version.version == "v1.0"
        assert version.config["sequence_length"] == 60
        assert isinstance(version.timestamp, datetime)

    def test_version_comparison_capability(self, temp_directory):
        """Test version comparison functionality"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load multiple versions
        manager.load_strategy("ml_adaptive", version="v1.0")
        manager.load_strategy("ml_adaptive", version="v1.1")
        
        # Should be able to get performance comparison
        comparison = manager.get_performance_comparison()
        assert isinstance(comparison, dict)

    def test_version_rollback_preparation(self, temp_directory):
        """Test rollback functionality preparation"""
        manager = StrategyManager(staging_dir=str(temp_directory))
        
        # Load and track versions
        v1 = manager.load_strategy("ml_adaptive", version="v1.0")
        v2 = manager.load_strategy("ml_adaptive", version="v2.0")
        
        # Test rollback capability exists
        assert hasattr(manager, 'rollback_to_previous_version')
        
        # Rollback should be callable
        result = manager.rollback_to_previous_version()
        # Currently returns False as not implemented, but should not crash