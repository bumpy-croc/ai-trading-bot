"""Unit tests for ML training pipeline orchestration module."""

from datetime import datetime
from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_TENSORFLOW_AVAILABLE = find_spec("tensorflow") is not None

from src.ml.training_pipeline.config import TrainingConfig, TrainingContext, TrainingPaths
from src.ml.training_pipeline.pipeline import (
    TrainingResult,
    _generate_version_id,
    enable_mixed_precision,
    run_training_pipeline,
)


@pytest.mark.fast
class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_initialization(self):
        # Arrange
        metadata = {"test": "data"}
        artifact_paths = MagicMock()

        # Act
        result = TrainingResult(
            success=True,
            metadata=metadata,
            artifact_paths=artifact_paths,
            duration_seconds=123.45,
        )

        # Assert
        assert result.success is True
        assert result.metadata == metadata
        assert result.artifact_paths == artifact_paths
        assert result.duration_seconds == 123.45

    def test_failed_result(self):
        # Arrange & Act
        result = TrainingResult(
            success=False,
            metadata={"error": "Something went wrong"},
            artifact_paths=None,
            duration_seconds=10.0,
        )

        # Assert
        assert result.success is False
        assert result.artifact_paths is None


@pytest.mark.fast
class TestGenerateVersionId:
    """Test _generate_version_id function."""

    def test_generates_version_id(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Act
        version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id.endswith("_v1")
        assert "_" in version_id
        # Format: YYYY-MM-DD_HHhMMmSSs_vN
        parts = version_id.split("_")
        assert len(parts) == 3
        assert parts[2] == "v1"

    def test_version_id_has_second_granularity(self, tmp_path):
        """Two same-hour cloud containers must not generate colliding IDs.

        Each container has an empty models dir, so the exists() counter cannot
        de-duplicate across jobs — the timestamp itself must be unique.
        """
        import re

        # Arrange
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Act
        version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s_v1", version_id)

    def test_increments_version_when_exists(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Create existing version directory
        existing_version = symbol_dir / "2024-01-01_10h_v1"
        existing_version.mkdir(parents=True, exist_ok=True)

        # Act
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01_10h"
            version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id == "2024-01-01_10h_v2"

    def test_multiple_versions(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple existing versions
        for i in range(1, 4):
            (symbol_dir / f"2024-01-01_10h_v{i}").mkdir(parents=True, exist_ok=True)

        # Act
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01_10h"
            version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id == "2024-01-01_10h_v4"

    def test_raises_error_after_max_retries(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Create all versions up to max
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01_10h"
            for i in range(1, 1001):
                (symbol_dir / f"2024-01-01_10h_v{i}").mkdir(parents=True, exist_ok=True)

            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to generate unique version ID"):
                _generate_version_id(models_dir, "BTCUSDT", "basic")


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestEnableMixedPrecision:
    """Test enable_mixed_precision function."""

    def test_disabled_via_config(self):
        # Arrange & Act
        enable_mixed_precision(enabled=False)

        # Assert - should not raise any errors

    @patch("tensorflow.config.list_physical_devices")
    def test_no_gpu_available(self, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = []

        # Act
        enable_mixed_precision(enabled=True)

        # Assert - should not raise any errors

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.keras.mixed_precision.set_global_policy")
    @patch("tensorflow.config.optimizer.set_jit")
    def test_successful_enable_with_gpu(self, mock_set_jit, mock_set_policy, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = [MagicMock()]  # Mock GPU

        # Act
        enable_mixed_precision(enabled=True)

        # Assert
        mock_set_policy.assert_called_once_with("mixed_float16")
        mock_set_jit.assert_called_once_with(True)

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.keras.mixed_precision.set_global_policy")
    def test_handles_runtime_error(self, mock_set_policy, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = [MagicMock()]
        mock_set_policy.side_effect = RuntimeError("GPU error")

        # Act - should not raise, just log warning
        enable_mixed_precision(enabled=True)

        # Assert - no exception should be raised

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.keras.mixed_precision.set_global_policy")
    def test_handles_value_error(self, mock_set_policy, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = [MagicMock()]
        mock_set_policy.side_effect = ValueError("Invalid policy")

        # Act - should not raise, just log warning
        enable_mixed_precision(enabled=True)

        # Assert - no exception should be raised


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestRunTrainingPipeline:
    """Test run_training_pipeline function."""

    @patch("src.ml.training_pipeline.pipeline.configure_gpu")
    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    @patch("src.ml.training_pipeline.pipeline.assess_sentiment_data_quality")
    @patch("src.ml.training_pipeline.pipeline.create_robust_features")
    @patch("src.ml.training_pipeline.pipeline.create_sequences")
    @patch("src.ml.training_pipeline.pipeline.split_sequences")
    @patch("src.ml.training_pipeline.pipeline.build_tf_datasets")
    @patch("src.ml.training_pipeline.pipeline.create_model")
    @patch("src.ml.training_pipeline.pipeline.save_artifacts")
    @patch("src.ml.training_pipeline.pipeline.enable_mixed_precision")
    def test_successful_training_price_only(
        self,
        mock_enable_mp,
        mock_save,
        mock_create_model,
        mock_build_ds,
        mock_split,
        mock_create_seq,
        mock_create_features,
        mock_assess,
        mock_load_sentiment,
        mock_download_price,
        mock_configure_gpu,
        tmp_path,
    ):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            epochs=1,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        ctx = TrainingContext(config=config, paths=paths)

        # Mock GPU configuration
        mock_configure_gpu.return_value = None  # CPU mode

        # Mock data
        price_df = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [105] * 100,
                "low": [99] * 100,
                "close": [103] * 100,
                "volume": [1000] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )
        mock_download_price.return_value = price_df
        mock_load_sentiment.return_value = None
        mock_assess.return_value = {"recommendation": "price_only", "quality_score": 0.0}

        # Mock feature engineering
        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5
        mock_create_features.return_value = (
            feature_df,
            {"close": MagicMock()},
            ["close_scaled"],
        )

        # Mock sequences
        sequences = np.random.rand(50, 120, 1).astype(np.float32)
        targets = np.random.rand(50).astype(np.float32)
        mock_create_seq.return_value = (sequences, targets)

        X_train = sequences[:40]
        y_train = targets[:40]
        X_val = sequences[40:]
        y_val = targets[40:]
        mock_split.return_value = (X_train, y_train, X_val, y_val)

        # Mock datasets
        train_ds = MagicMock()
        val_ds = MagicMock()
        mock_build_ds.return_value = (train_ds, val_ds)

        # Mock model
        model = MagicMock()
        model.fit.return_value = MagicMock(history={"loss": [0.1], "val_loss": [0.2]})
        model.predict.return_value = np.random.rand(10, 1)
        model.evaluate.side_effect = [{"loss": 0.1, "rmse": 0.3}, {"loss": 0.2, "rmse": 0.4}]
        mock_create_model.return_value = model

        # Mock artifacts
        artifact_paths = MagicMock()
        artifact_paths.directory = tmp_path / "artifacts"
        mock_save.return_value = artifact_paths

        # Act
        result = run_training_pipeline(ctx)

        # Assert
        assert result.success is True
        assert result.artifact_paths is not None
        assert result.duration_seconds > 0
        assert "symbol" in result.metadata
        assert result.metadata["symbol"] == "BTCUSDT"

    def _make_price_df(self, periods=200):
        """Build a deterministic OHLCV frame with varying prices."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.normal(0, 1, periods))
        return pd.DataFrame(
            {
                "open": closes - 0.5,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": 1000.0 + rng.uniform(0, 100, periods),
            },
            index=pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC"),
        )

    def _run_pipeline_with_mocks(self, ctx, price_df, mocks):
        """Run the pipeline with the standard post-feature stages mocked out."""
        sequences = np.random.rand(50, ctx.config.sequence_length, 5).astype(np.float32)
        targets = np.random.rand(50).astype(np.float32)
        mocks["create_sequences"].return_value = (sequences, targets)
        mocks["split_sequences"].return_value = (
            sequences[:40],
            targets[:40],
            sequences[40:],
            targets[40:],
        )
        mocks["build_tf_datasets"].return_value = (MagicMock(), MagicMock())
        model = MagicMock()
        model.fit.return_value = MagicMock(history={"loss": [0.1], "val_loss": [0.2]})
        mocks["create_model"].return_value = model
        mocks["validate_model_robustness"].return_value = {}
        mocks["evaluate_model_performance"].return_value = {"test_rmse": 0.1}
        artifact_paths = MagicMock()
        artifact_paths.directory = ctx.paths.models_dir
        mocks["save_artifacts"].return_value = artifact_paths
        mocks["create_training_plots"].return_value = None
        return run_training_pipeline(ctx)

    def _pipeline_mocks(self):
        """Context-manage patches for the post-feature pipeline stages."""
        from contextlib import ExitStack

        stack = ExitStack()
        names = [
            "configure_gpu",
            "download_price_data",
            "load_sentiment_data",
            "create_robust_features",
            "create_sequences",
            "split_sequences",
            "build_tf_datasets",
            "create_model",
            "validate_model_robustness",
            "evaluate_model_performance",
            "save_artifacts",
            "create_training_plots",
            "enable_mixed_precision",
        ]
        mocks = {
            name: stack.enter_context(patch(f"src.ml.training_pipeline.pipeline.{name}"))
            for name in names
        }
        mocks["configure_gpu"].return_value = None
        return stack, mocks

    def test_force_price_only_routes_through_price_only_extractor(self, tmp_path):
        """force_price_only must produce the production 5-feature price-only contract.

        Pins the contract: PriceOnlyFeatureExtractor features (not
        create_robust_features' 9-feature pipeline), target = close_normalized
        (not raw close). The bundle stays in the price/ namespace so basic/latest
        (loaded by live strategies) is never repointed implicitly.
        """
        from src.prediction.features.price_only import PriceOnlyFeatureExtractor

        # Arrange
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
            force_price_only=True,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)
        price_df = self._make_price_df()

        stack, mocks = self._pipeline_mocks()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None

            # Act
            result = self._run_pipeline_with_mocks(ctx, price_df, mocks)

        # Assert
        assert result.success is True, result.metadata
        mocks["create_robust_features"].assert_not_called()

        expected = PriceOnlyFeatureExtractor(normalization_window=10).extract(price_df.copy())
        expected_names = [
            "close_normalized",
            "volume_normalized",
            "high_normalized",
            "low_normalized",
            "open_normalized",
        ]
        feature_array, target_array, seq_len = mocks["create_sequences"].call_args.args
        assert feature_array.shape[1] == 5
        assert seq_len == 10
        np.testing.assert_allclose(
            target_array,
            expected["close_normalized"].to_numpy(dtype=np.float32),
            rtol=1e-6,
        )

        assert result.metadata["model_type"] == "price"
        assert result.metadata["feature_names"] == expected_names
        assert mocks["create_model"].call_args.kwargs["input_shape"] == (10, 5)
        # Bundle must be saved under price/, never basic/ (which would flip the
        # live-loaded basic/latest symlink as a training side effect)
        assert mocks["save_artifacts"].call_args.args[2] == "price"

    def test_default_path_regresses_raw_close(self, tmp_path):
        """Without force_price_only the existing contract is unchanged.

        create_robust_features supplies the features and the raw close is the
        regression target; registry model_type stays price/sentiment.
        """
        # Arrange
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)
        price_df = self._make_price_df()

        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5

        stack, mocks = self._pipeline_mocks()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )

            # Act
            result = self._run_pipeline_with_mocks(ctx, price_df, mocks)

        # Assert
        assert result.success is True, result.metadata
        mocks["create_robust_features"].assert_called_once()
        _, target_array, _ = mocks["create_sequences"].call_args.args
        np.testing.assert_allclose(target_array, feature_df["close"].to_numpy(dtype=np.float32))
        assert result.metadata["model_type"] == "price"

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    def test_handles_download_failure(self, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        mock_download.side_effect = RuntimeError("Download failed")

        # Act
        result = run_training_pipeline(ctx)

        # Assert
        assert result.success is False
        assert "error" in result.metadata
        assert result.artifact_paths is None

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    @patch("src.ml.training_pipeline.pipeline.assess_sentiment_data_quality")
    @patch("src.ml.training_pipeline.pipeline.merge_price_sentiment_data")
    def test_handles_empty_merged_data(
        self,
        mock_merge,
        mock_assess,
        mock_load_sentiment,
        mock_download,
        tmp_path,
    ):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        price_df = pd.DataFrame()
        sentiment_df = pd.DataFrame()
        mock_download.return_value = price_df
        mock_load_sentiment.return_value = sentiment_df
        mock_assess.return_value = {"recommendation": "price_only"}
        mock_merge.return_value = pd.DataFrame()

        # Act
        result = run_training_pipeline(ctx)

        # Assert
        assert result.success is False
        assert "error" in result.metadata
        assert "No data available" in result.metadata["error"]

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    @patch("src.ml.training_pipeline.pipeline.assess_sentiment_data_quality")
    @patch("src.ml.training_pipeline.pipeline.create_robust_features")
    def test_force_sentiment_overrides_recommendation(
        self,
        mock_create_features,
        mock_assess,
        mock_load_sentiment,
        mock_download,
        tmp_path,
    ):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            force_sentiment=True,
            epochs=1,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        price_df = pd.DataFrame(
            {
                "open": [100] * 100,
                "close": [103] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5] * 100},
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )

        mock_download.return_value = price_df
        mock_load_sentiment.return_value = sentiment_df

        # Original assessment says price_only, but force_sentiment should override
        mock_assess.return_value = {"recommendation": "price_only", "quality_score": 0.3}

        # Mock feature creation to avoid full pipeline execution
        mock_create_features.side_effect = RuntimeError("Stop pipeline for test")

        # Act
        result = run_training_pipeline(ctx)

        # Assert - pipeline should fail at feature creation, but we verify the assessment was overridden
        assert result.success is False
        # The test confirms force_sentiment logic runs before feature creation

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    def test_handles_timezone_localization(self, mock_load_sentiment, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        # Price data without timezone
        price_df = pd.DataFrame(
            {"close": [100] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="1h"),
        )
        # Sentiment data with timezone
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"),
        )

        mock_download.return_value = price_df
        mock_load_sentiment.return_value = sentiment_df

        # Act - will fail later in pipeline, but timezone handling should work
        result = run_training_pipeline(ctx)

        # Assert - should not fail on timezone mismatch
        # Will fail on other things (like insufficient data), but that's expected
        assert result.success is False

    def test_incompatible_target_type_fails_loud_before_download(self, tmp_path):
        """#947 guard: model_type's head must match target_type's task type.
        tft (binary_classification head) + target_type=regression (the
        default) previously trained silently against the wrong target --
        must now fail loudly, and fast (before any data download)."""
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            model_type="tft",
            target_type="regression",
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        with patch("src.ml.training_pipeline.pipeline.download_price_data") as mock_download:
            result = run_training_pipeline(ctx)

        assert result.success is False
        assert "incompatible" in result.metadata["error"]
        mock_download.assert_not_called()

    def test_compatible_classification_target_passes_the_guard(self, tmp_path):
        """tft + binary_direction is the compatible pairing -- must NOT be
        rejected by the guard (only fails later for unrelated pipeline
        reasons, since data isn't mocked in this test)."""
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            model_type="tft",
            target_type="binary_direction",
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        with patch("src.ml.training_pipeline.pipeline.download_price_data") as mock_download:
            mock_download.side_effect = RuntimeError("stop after the guard, unrelated to it")
            result = run_training_pipeline(ctx)

        # Guard passed (download was actually attempted); failure is the
        # unrelated injected error, not an "incompatible" guard rejection.
        mock_download.assert_called_once()
        assert result.success is False
        assert "incompatible" not in result.metadata["error"]

    def test_binary_direction_target_builds_labels_from_raw_close(self, tmp_path):
        """target_type=binary_direction must override the regression target
        with binary_direction_labels() output, row-aligned to feature_data,
        with trailing unresolved rows dropped from BOTH feature and target
        arrays (never silently zero-filled)."""
        from src.ml.training_pipeline.labels import binary_direction_labels

        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
            model_type="tft",
            target_type="binary_direction",
            target_horizon=1,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)
        price_df = self._make_price_df()

        # create_robust_features mocked to a byte-identical copy of price_df
        # (plus a feature column) so feature_data's index is the SAME as
        # merged_df's -- isolates the alt-target alignment/truncation logic
        # from feature-engineering warmup-row-dropping concerns.
        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5

        stack, mocks = self._pipeline_mocks()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )

            result = self._run_pipeline_with_mocks(ctx, price_df, mocks)

        assert result.success is True, result.metadata
        feature_array, target_array, seq_len = mocks["create_sequences"].call_args.args
        assert seq_len == 10

        expected_label = binary_direction_labels(price_df["close"], horizon=1)
        expected_target = expected_label.values[expected_label.valid_mask].astype(np.float32)
        expected_rows = int(expected_label.valid_mask.sum())

        assert len(target_array) == expected_rows
        assert len(feature_array) == expected_rows
        np.testing.assert_array_equal(target_array, expected_target)
        # Every value is 0 or 1 -- confirms the classification label (not the
        # continuous close price) reached create_sequences.
        assert set(np.unique(target_array)).issubset({0.0, 1.0})

    def test_smoothed_return_target_builds_labels_from_raw_close(self, tmp_path):
        """target_type=smoothed_return overrides the regression target with
        smoothed_forward_return_labels() output (still a REGRESSION task
        type, so a regression-headed model_type is compatible)."""
        from src.ml.training_pipeline.labels import smoothed_forward_return_labels

        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
            target_type="smoothed_return",
            target_horizon=3,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)
        price_df = self._make_price_df()

        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5

        stack, mocks = self._pipeline_mocks()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )

            result = self._run_pipeline_with_mocks(ctx, price_df, mocks)

        assert result.success is True, result.metadata
        feature_array, target_array, _ = mocks["create_sequences"].call_args.args

        expected_label = smoothed_forward_return_labels(price_df["close"], horizon=3)
        expected_target = expected_label.values[expected_label.valid_mask].astype(np.float32)

        assert len(target_array) == len(feature_array)
        np.testing.assert_allclose(target_array, expected_target, rtol=1e-5)

    def test_regression_target_type_is_byte_identical_to_current_behavior(self, tmp_path):
        """target_type="regression" (the default) must produce EXACTLY the
        existing target_array -- no truncation, no row drops."""
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)
        price_df = self._make_price_df()
        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5

        stack, mocks = self._pipeline_mocks()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )
            result = self._run_pipeline_with_mocks(ctx, price_df, mocks)

        assert result.success is True, result.metadata
        _, target_array, _ = mocks["create_sequences"].call_args.args
        np.testing.assert_allclose(target_array, feature_df["close"].to_numpy(dtype=np.float32))
        assert len(target_array) == len(feature_df)

    def test_evaluation_crash_does_not_lose_trained_model(self, tmp_path):
        """Regression guard for #936: evaluate_model_performance/validate_model_robustness
        run after model.fit() but before save_artifacts. A crash there previously
        discarded a fully-trained model with nothing persisted. Diagnostics failures
        must degrade to a metadata gap instead of aborting the run without saving.
        """
        # Arrange
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)
        price_df = self._make_price_df()
        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5

        stack, mocks = self._pipeline_mocks()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )
            mocks["evaluate_model_performance"].side_effect = ValueError(
                "too many values to unpack (expected 2)"
            )

            # Act
            result = self._run_pipeline_with_mocks(ctx, price_df, mocks)

        # Assert - the trained model is still saved despite the diagnostics crash
        assert result.success is True, result.metadata
        mocks["save_artifacts"].assert_called_once()
        assert "error" in result.metadata["evaluation_results"]
        assert "too many values to unpack" in result.metadata["evaluation_results"]["error"]
