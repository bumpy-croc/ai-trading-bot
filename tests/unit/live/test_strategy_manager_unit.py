import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from live.strategy_manager import StrategyManager, StrategyVersion

pytestmark = pytest.mark.unit


class TestStrategyManager:
    def test_strategy_manager_initialization(self, temp_directory):
        temp_staging = str(temp_directory / "staging")
        manager = StrategyManager(
            strategies_dir=str(temp_directory / "strategies"),
            models_dir=str(temp_directory / "models"),
            staging_dir=temp_staging,
        )
        assert manager.strategies_dir == temp_directory / "strategies"
        assert manager.models_dir == temp_directory / "models"
        assert manager.staging_dir == Path(temp_staging)
        assert manager.staging_dir.exists()
        assert manager.current_strategy is None
        assert manager.pending_update is None

    def test_strategy_loading(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        strategy = manager.load_strategy("ml_basic", version="test_v1")
        from strategies.ml_basic import MlBasic
        assert isinstance(strategy, MlBasic)
        assert manager.current_strategy == strategy
        assert manager.current_version.strategy_name == "ml_basic"
        assert manager.current_version.version == "test_v1"

    def test_strategy_loading_with_config(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        config = {"name": "CustomMlBasic", "sequence_length": 120}
        strategy = manager.load_strategy("ml_basic", config=config)
        assert strategy.name == "CustomMlBasic"
        assert strategy.sequence_length == 120

    def test_invalid_strategy_loading(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        with pytest.raises(ValueError):
            manager.load_strategy("nonexistent_strategy")

    def test_pending_update_detection(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        assert manager.has_pending_update() is False
        manager.load_strategy("ml_basic")
        manager.hot_swap_strategy("ml_basic", new_config={"sequence_length": 120})
        assert manager.has_pending_update() is True
        manager.apply_pending_update()
        assert manager.has_pending_update() is False

    def test_version_history_tracking(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_basic", version="v1")
        manager.load_strategy("ml_basic", version="v2")
        assert len(manager.version_history) >= 2
        assert "ml_basic_v1" in manager.version_history
        assert "ml_basic_v2" in manager.version_history

    def test_strategy_registry(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        available = manager.list_available_strategies()
        assert 'available_strategies' in available
        assert 'ml_basic' in available['available_strategies']


class TestStrategyManagerThreadSafety:
    def test_concurrent_strategy_loading(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        results, errors = [], []

        def load_strategy(strategy_name, version):
            try:
                s = manager.load_strategy(strategy_name, version=version)
                results.append((strategy_name, version, s))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=load_strategy, args=("ml_basic", f"v{i}"))
            threads.append(t); t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) >= 2

    def test_concurrent_hot_swapping(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_basic", version="initial")
        swap_results = []

        def attempt_hot_swap(variant):
            try:
                swap_results.append(manager.hot_swap_strategy("ml_basic", new_config={"sequence_length": variant}))
            except Exception:
                swap_results.append(False)

        threads = []
        for i in range(3):
            t = threading.Thread(target=attempt_hot_swap, args=(120 + i,))
            threads.append(t); t.start()
        for t in threads:
            t.join()
        assert sum(swap_results) <= 1

    def test_update_lock_behavior(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_basic")
        lock_acquired_count = 0

        def acquire():
            nonlocal lock_acquired_count
            with manager.update_lock:
                lock_acquired_count += 1
                time.sleep(0.05)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=acquire)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert lock_acquired_count == 5


class TestStrategyVersioning:
    def test_strategy_version_creation(self, temp_directory):
        version = StrategyVersion(strategy_name="ml_basic", version="v1.0", timestamp=datetime.now(), config={"sequence_length": 120})
        assert version.strategy_name == "ml_basic"
        assert version.version == "v1.0"
        assert version.config["sequence_length"] == 120
        assert isinstance(version.timestamp, datetime)

    def test_version_comparison_capability(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_basic", version="v1.0")
        manager.load_strategy("ml_basic", version="v1.1")
        comparison = manager.get_performance_comparison()
        assert isinstance(comparison, dict)
