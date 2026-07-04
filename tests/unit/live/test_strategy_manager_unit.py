import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.engines.live.strategy_manager import StrategyManager, StrategyVersion

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
        from src.strategies.components import Strategy as ComponentStrategy

        assert isinstance(strategy, ComponentStrategy)
        assert manager.current_strategy == strategy
        assert manager.current_version.strategy_name == "ml_basic"
        assert manager.current_version.version == "test_v1"

    def test_strategy_loading_with_config(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        config = {"name": "CustomMlBasic", "sequence_length": 120}
        strategy = manager.load_strategy("ml_basic", config=config)
        assert strategy.name == "CustomMlBasic"

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
        assert "available_strategies" in available
        assert "ml_basic" in available["available_strategies"]


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
            threads.append(t)
            t.start()
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
                swap_results.append(
                    manager.hot_swap_strategy("ml_basic", new_config={"sequence_length": variant})
                )
            except Exception:
                swap_results.append(False)

        threads = []
        for i in range(3):
            t = threading.Thread(target=attempt_hot_swap, args=(120 + i,))
            threads.append(t)
            t.start()
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
        version = StrategyVersion(
            version_id="ml_basic_v1.0",
            strategy_name="ml_basic",
            version="v1.0",
            created_at=datetime.now(UTC),
            config={"sequence_length": 120},
        )
        assert version.strategy_name == "ml_basic"
        assert version.version == "v1.0"
        assert version.config["sequence_length"] == 120
        assert isinstance(version.created_at, datetime)

    def test_version_comparison_capability(self, temp_directory):
        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_basic", version="v1.0")
        manager.load_strategy("ml_basic", version="v1.1")
        comparison = manager.get_performance_comparison()
        assert isinstance(comparison, dict)

    def test_update_model_without_current_strategy_reports_descriptive_error(
        self, temp_directory, caplog
    ):
        """Regression for #765: with no strategy loaded, update_model must fail
        with the descriptive mismatch ValueError, not an AttributeError from
        dereferencing ``self.current_strategy.name`` inside the f-string."""
        import logging

        manager = StrategyManager(staging_dir=str(temp_directory))
        assert manager.current_strategy is None

        model_path = temp_directory / "model.onnx"
        model_path.write_bytes(b"dummy")

        with caplog.at_level(logging.ERROR):
            result = manager.update_model("ml_basic", str(model_path), validate_model=False)

        assert result is False
        assert "doesn't match" in caplog.text
        assert "<none>" in caplog.text
        assert "AttributeError" not in caplog.text
        assert "NoneType" not in caplog.text

    def test_update_model_mismatched_strategy_name_reports_current_name(
        self, temp_directory, caplog
    ):
        """The mismatch error still names the loaded strategy when one exists."""
        import logging

        manager = StrategyManager(staging_dir=str(temp_directory))
        manager.load_strategy("ml_basic")

        model_path = temp_directory / "model.onnx"
        model_path.write_bytes(b"dummy")

        with caplog.at_level(logging.ERROR):
            result = manager.update_model("other_strategy", str(model_path), validate_model=False)

        assert result is False
        assert "doesn't match other_strategy" in caplog.text


class TestStrategyManagerSymbolThreading:
    """Hot-swap/load must thread the engine's trading symbol (#867)."""

    ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
    CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"

    def _mock_engine(self):
        from unittest.mock import MagicMock

        engine = MagicMock()
        engine.health_check.return_value = {"status": "healthy"}
        return engine

    def test_load_strategy_threads_symbol(self, temp_directory):
        from unittest.mock import patch

        with patch(self.ENGINE_PATH) as engine_cls, patch(self.CONFIG_PATH):
            engine_cls.return_value = self._mock_engine()
            manager = StrategyManager(staging_dir=str(temp_directory), symbol="ETHUSDT")

            strategy = manager.load_strategy("ml_basic")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    def test_hot_swap_threads_symbol(self, temp_directory):
        from unittest.mock import MagicMock, patch

        with patch(self.ENGINE_PATH) as engine_cls, patch(self.CONFIG_PATH):
            engine_cls.return_value = self._mock_engine()
            manager = StrategyManager(staging_dir=str(temp_directory), symbol="ETHUSDT")
            manager.current_strategy = MagicMock()

            assert manager.hot_swap_strategy("ml_basic", new_config={"name": "SwapTest"}) is True

        swapped = manager.pending_update["data"]["new_strategy"]
        assert swapped.signal_generator.symbol == "ETHUSDT"

    def test_explicit_config_symbol_wins_over_manager_symbol(self, temp_directory):
        from unittest.mock import patch

        with patch(self.ENGINE_PATH) as engine_cls, patch(self.CONFIG_PATH):
            engine_cls.return_value = self._mock_engine()
            manager = StrategyManager(staging_dir=str(temp_directory), symbol="ETHUSDT")

            strategy = manager.load_strategy("ml_basic", config={"symbol": "BTCUSDT"})

        assert strategy.signal_generator.symbol == "BTCUSDT"

    def test_no_symbol_keeps_legacy_default(self, temp_directory):
        from unittest.mock import patch

        with patch(self.ENGINE_PATH) as engine_cls, patch(self.CONFIG_PATH):
            engine_cls.return_value = self._mock_engine()
            manager = StrategyManager(staging_dir=str(temp_directory))

            strategy = manager.load_strategy("ml_basic")

        assert strategy.signal_generator.symbol == "BTCUSDT"

    def test_swap_time_model_unavailable_rejects_swap_and_keeps_current(
        self, temp_directory, caplog, monkeypatch
    ):
        """A missing-model failure during hot-swap must not kill the loop.

        The swap is rejected, the current strategy stays active, and the
        failure is logged at ERROR.
        """
        import logging
        from unittest.mock import MagicMock, patch

        from src.prediction.models.exceptions import ModelNotAvailableError

        monkeypatch.delenv("FEATURE_ALLOW_CROSS_SYMBOL_MODEL", raising=False)

        registry = MagicMock()
        registry.select_bundle.side_effect = ModelNotAvailableError("no ETHUSDT basic model")
        registry.list_bundles.return_value = []
        engine = self._mock_engine()
        engine.model_registry = registry

        with patch(self.ENGINE_PATH) as engine_cls, patch(self.CONFIG_PATH):
            engine_cls.return_value = engine
            manager = StrategyManager(staging_dir=str(temp_directory), symbol="ETHUSDT")
            current = MagicMock()
            manager.current_strategy = current

            with caplog.at_level(logging.ERROR):
                result = manager.hot_swap_strategy("ml_basic")

        assert result is False
        assert manager.current_strategy is current
        assert manager.pending_update is None
        assert "Hot-swap failed" in caplog.text
