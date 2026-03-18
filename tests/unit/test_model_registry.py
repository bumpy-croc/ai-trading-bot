from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prediction.config import PredictionConfig
from src.prediction.models.exceptions import ModelLoadError
from src.prediction.models.registry import PredictionModelRegistry, StrategyModel


def _make_bundle(tmpdir: Path, symbol: str, model_type: str, timeframe: str, version: str) -> Path:
    base = tmpdir / symbol / model_type / version
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.onnx").write_bytes(b"dummy")
    (base / "metadata.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "model_type": model_type,
                "timeframe": timeframe,
                "version_id": version,
            }
        )
    )
    return base


def test_select_bundle_and_many(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    _make_bundle(reg_root, "ETHUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    reg = PredictionModelRegistry(cfg)
    b = reg.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")
    assert isinstance(b, StrategyModel)
    sel = reg.select_many(
        [
            ("BTCUSDT", "basic", "1h"),
            ("ETHUSDT", "basic", "1h"),
        ]
    )
    assert len(sel) == 2


def test_select_many_fail_fast(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))
    reg = PredictionModelRegistry(cfg)

    with pytest.raises(ModelLoadError):
        reg.select_many(
            [
                ("BTCUSDT", "basic", "1h"),
                ("ETHUSDT", "basic", "1h"),  # missing
            ]
        )


# ---- Cache invalidation tests ----


from unittest.mock import MagicMock, call, patch  # noqa: E402, isort:skip


def test_invalidate_cache_maps_structured_name_to_runner(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    # First attempt returns 0 (not found), second returns 2 for runner name
    mock_cache_manager.invalidate_model.side_effect = [0, 2]

    invalidated = reg.invalidate_cache("BTCUSDT:1h:basic")

    assert invalidated == 2
    assert mock_cache_manager.invalidate_model.call_args_list == [
        call("BTCUSDT:1h:basic"),
        call("model.onnx"),
    ]


def test_invalidate_cache_clear_all(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    mock_cache_manager.clear.return_value = 5
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    invalidated = reg.invalidate_cache()

    assert invalidated == 5
    mock_cache_manager.clear.assert_called_once_with()


# ---- Thread safety tests ----


def test_registry_concurrent_reload_thread_safety(tmp_path: Path, monkeypatch):
    """Test thread safety during concurrent reload_models calls"""
    import threading
    import time

    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    _make_bundle(reg_root, "ETHUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    reg = PredictionModelRegistry(cfg)

    num_threads = 5
    iterations = 10
    errors = []
    reload_count = [0]  # Use list to share across threads

    def worker_reload():
        """Worker that reloads the model registry"""
        try:
            for _ in range(iterations):
                reg.reload_models()
                reload_count[0] += 1
                time.sleep(0.001)
        except Exception as e:
            errors.append({"operation": "reload", "error": str(e)})

    def worker_read():
        """Worker that reads bundles"""
        try:
            for _ in range(iterations):
                bundles = reg.list_bundles()
                assert isinstance(bundles, list)
                for b in bundles:
                    assert hasattr(b, "symbol")
                    assert hasattr(b, "model_type")
                time.sleep(0.001)
        except Exception as e:
            errors.append({"operation": "read", "error": str(e)})

    # Create threads
    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=worker_reload))
    for _ in range(2):
        threads.append(threading.Thread(target=worker_read))

    # Start all threads
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Thread safety errors: {errors}"

    # Verify registry is still functional
    bundles = reg.list_bundles()
    assert len(bundles) == 2  # Should have both BTCUSDT and ETHUSDT


def test_registry_concurrent_select_bundle_thread_safety(tmp_path: Path, monkeypatch):
    """Test thread safety during concurrent select_bundle calls"""
    import threading
    import time

    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    _make_bundle(reg_root, "ETHUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    reg = PredictionModelRegistry(cfg)

    num_threads = 10
    iterations = 20
    errors = []
    results = []

    def worker_select(thread_id):
        """Worker that selects bundles"""
        try:
            for i in range(iterations):
                # Alternate between BTCUSDT and ETHUSDT
                symbol = "BTCUSDT" if i % 2 == 0 else "ETHUSDT"
                bundle = reg.select_bundle(symbol=symbol, model_type="basic", timeframe="1h")

                assert bundle is not None
                assert bundle.symbol == symbol
                assert bundle.model_type == "basic"

                results.append(
                    {
                        "thread_id": thread_id,
                        "iteration": i,
                        "symbol": symbol,
                    }
                )
                time.sleep(0.0001)

        except Exception as e:
            errors.append({"thread_id": thread_id, "error": str(e)})

    # Create and start threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker_select, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Thread safety errors: {errors}"

    # Verify all operations completed
    assert len(results) == num_threads * iterations

    # Verify both symbols were selected
    symbols_selected = set(r["symbol"] for r in results)
    assert symbols_selected == {"BTCUSDT", "ETHUSDT"}


# ---- reload_models cache-clearing tests ----


def _make_fake_bundle(symbol: str, model_type: str, timeframe: str, version: str):
    """Create a mock StrategyModel bundle for reload_models tests."""
    mock_runner = MagicMock()
    mock_runner.close = MagicMock()
    return StrategyModel(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        version_id=version,
        directory=Path("/fake"),
        metadata={"symbol": symbol},
        feature_schema=None,
        metrics=None,
        runner=mock_runner,
    )


def test_reload_models_clears_cache_on_success(tmp_path: Path, monkeypatch):
    """Verify cache_manager.clear() is called after a successful model swap."""
    # Arrange
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    mock_cache_manager.clear.return_value = 3
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    # Reset mock after initial _load to isolate reload_models behavior
    mock_cache_manager.reset_mock()

    # Patch _scan_registry to return a valid bundle so reload doesn't early-return
    fake_bundle = _make_fake_bundle("BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    fake_bundles = {("BTCUSDT", "1h", "basic"): fake_bundle}
    with patch.object(reg, "_scan_registry", return_value=(fake_bundles, {})):
        # Act
        reg.reload_models()

    # Assert: cache_manager.clear() is called once after reload
    mock_cache_manager.clear.assert_called_once()


@pytest.mark.fast
def test_reload_models_warns_when_no_cache_manager(tmp_path: Path, monkeypatch, caplog):
    """Verify a warning is logged when cache_manager is None during reload."""
    # Arrange
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    # No cache_manager provided
    reg = PredictionModelRegistry(cfg, cache_manager=None)

    # Patch _scan_registry to return a valid bundle so reload doesn't early-return
    fake_bundle = _make_fake_bundle("BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    fake_bundles = {("BTCUSDT", "1h", "basic"): fake_bundle}

    # Act
    import logging

    with (
        caplog.at_level(logging.WARNING),
        patch.object(reg, "_scan_registry", return_value=(fake_bundles, {})),
    ):
        reg.reload_models()

    # Assert: warning about missing cache_manager is logged
    assert any("No cache_manager available" in msg for msg in caplog.messages)


@pytest.mark.fast
def test_reload_models_handles_cache_clear_exception(tmp_path: Path, monkeypatch, caplog):
    """Verify reload_models handles exceptions from cache_manager.clear() gracefully."""
    # Arrange
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    mock_cache_manager.clear.side_effect = RuntimeError("cache backend unavailable")
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    # Reset mock after initial _load
    mock_cache_manager.reset_mock()
    mock_cache_manager.clear.side_effect = RuntimeError("cache backend unavailable")

    # Patch _scan_registry to return a valid bundle
    fake_bundle = _make_fake_bundle("BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    fake_bundles = {("BTCUSDT", "1h", "basic"): fake_bundle}

    # Act
    import logging

    with (
        caplog.at_level(logging.WARNING),
        patch.object(reg, "_scan_registry", return_value=(fake_bundles, {})),
    ):
        reg.reload_models()

    # Assert: exception is caught and logged, not raised
    assert any("Failed to clear prediction cache" in msg for msg in caplog.messages)


@pytest.mark.fast
def test_reload_models_cache_clear_returns_none(tmp_path: Path, monkeypatch):
    """Verify reload_models handles cache_manager.clear() returning None."""
    # Arrange
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    mock_cache_manager.clear.return_value = None
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    # Reset mock after initial _load
    mock_cache_manager.reset_mock()
    mock_cache_manager.clear.return_value = None

    # Patch _scan_registry to return a valid bundle
    fake_bundle = _make_fake_bundle("BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    fake_bundles = {("BTCUSDT", "1h", "basic"): fake_bundle}

    with patch.object(reg, "_scan_registry", return_value=(fake_bundles, {})):
        # Act -- should not raise despite None return value
        reg.reload_models()

    # Assert
    mock_cache_manager.clear.assert_called_once()
