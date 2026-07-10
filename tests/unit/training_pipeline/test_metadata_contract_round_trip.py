"""Producer -> consumer round-trip tests for the classification/distribution
metadata contract (TARGET-REDESIGN tournament, #933 Phase 2, PR #948 review
fix round).

The gauntlet's architecture review found the consumer half of this contract
(OnnxRunner, exam_signal_generator) shipped in full but nothing wrote
task_type / class_labels / target_distribution into metadata.json at
training time -- so a classifier bundle was unconsumable and entrant (d)
always degraded to HOLD. These tests exercise BOTH halves for real:

1. run_training_pipeline() (mostly mocked internals, but save_artifacts
   runs for real) actually writes task_type/class_labels/target_distribution
   into a real metadata.json on disk.
2. A REAL PredictionModelRegistry loads that metadata.json into a REAL
   OnnxRunner instance (only the onnxruntime.InferenceSession call itself is
   mocked, matching this repo's existing ONNX unit-test convention --
   tests/conftest.py stubs the `onnxruntime` module suite-wide for speed) and
   a classification prediction round-trips without any loud-failure path
   firing. This is exactly the registry/runner metadata-path fix:
   OnnxRunner._load_metadata previously looked for a "{stem}_metadata.json"
   sidecar that no writer in this codebase ever produced, so it silently
   never saw metadata.json's real content regardless of what the registry
   correctly loaded into StrategyModel.metadata.
3. ClassificationExamSignalGenerator consumes a real registry-loaded
   classifier end-to-end (no HOLD-degradation from missing probabilities).
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.training_pipeline.config import (
    DiagnosticsOptions,
    TrainingConfig,
    TrainingContext,
    TrainingPaths,
)
from src.ml.training_pipeline.pipeline import run_training_pipeline
from src.prediction.config import PredictionConfig
from src.prediction.distribution_stats import FrozenDistribution, percentile_rank_confidence
from src.prediction.engine import PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.models.registry import PredictionModelRegistry
from src.strategies.components.exam_signal_generator import ClassificationExamSignalGenerator


@pytest.fixture(autouse=True)
def _mock_providers():
    """OnnxRunner._load_model calls get_preferred_providers() before
    constructing InferenceSession; the stub `onnxruntime` module
    (tests/conftest.py) has no get_available_providers, so this must be
    mocked too (matches tests/unit/predictions/test_models.py's setup)."""
    with patch(
        "src.prediction.models.onnx_runner.get_preferred_providers",
        return_value=["CPUExecutionProvider"],
    ):
        yield


def _mock_inference_session(output_values: list[float]):
    """A mock onnxruntime.InferenceSession that always returns the given
    output vector, regardless of input -- matches this repo's existing
    ONNX unit-test convention (see tests/unit/predictions/test_models.py),
    since tests/conftest.py stubs the real `onnxruntime` module suite-wide.
    """
    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name="input")]
    session.run.return_value = [np.array([output_values], dtype=np.float32)]
    return session


def _make_ohlcv_df(periods: int = 130) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.3, periods))
    return pd.DataFrame(
        {
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": 1000.0 + rng.uniform(0, 50, periods),
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC"),
    )


class TestPipelineWritesClassificationMetadata:
    """Producer half: run_training_pipeline() writes the metadata contract
    for real (save_artifacts is NOT mocked -- json.dump runs against a real
    tmp_path)."""

    def _pipeline_mocks_except_save_artifacts(self):
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
            "create_training_plots",
            "enable_mixed_precision",
        ]
        mocks = {
            name: stack.enter_context(patch(f"src.ml.training_pipeline.pipeline.{name}"))
            for name in names
        }
        mocks["configure_gpu"].return_value = None
        return stack, mocks

    def test_binary_direction_writes_task_type_and_class_labels(self, tmp_path):
        price_df = _make_ohlcv_df()
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
            diagnostics=DiagnosticsOptions(
                generate_plots=False, evaluate_robustness=False, convert_to_onnx=False
            ),
        )
        paths = TrainingPaths(
            project_root=tmp_path, data_dir=tmp_path / "data", models_dir=tmp_path / "models"
        )
        ctx = TrainingContext(config=config, paths=paths)

        stack, mocks = self._pipeline_mocks_except_save_artifacts()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            feature_df = price_df.copy()
            feature_df["close_scaled"] = 0.5
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )
            sequences = np.random.rand(50, 10, 1).astype(np.float32)
            targets = np.random.randint(0, 2, 50).astype(np.float32)
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
            mocks["evaluate_model_performance"].return_value = {
                "error": "diagnostics skipped in test"
            }
            mocks["create_training_plots"].return_value = None

            result = run_training_pipeline(ctx)

        assert result.success is True, result.metadata
        metadata_path = result.artifact_paths.metadata_path
        assert metadata_path.exists()
        on_disk = json.loads(metadata_path.read_text())

        assert on_disk["task_type"] == "binary_classification"
        assert on_disk["class_labels"] == [-1, 1]
        assert "target_distribution" not in on_disk

    def test_smoothed_return_writes_task_type_and_target_distribution(self, tmp_path):
        price_df = _make_ohlcv_df()

        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
            target_type="smoothed_return",
            target_horizon=3,
            diagnostics=DiagnosticsOptions(
                generate_plots=False, evaluate_robustness=False, convert_to_onnx=False
            ),
        )
        paths = TrainingPaths(
            project_root=tmp_path, data_dir=tmp_path / "data", models_dir=tmp_path / "models"
        )
        ctx = TrainingContext(config=config, paths=paths)

        stack, mocks = self._pipeline_mocks_except_save_artifacts()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            feature_df = price_df.copy()
            feature_df["close_scaled"] = 0.5
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )
            sequences = np.random.rand(50, 10, 1).astype(np.float32)
            targets = np.random.uniform(-0.05, 0.05, 50).astype(np.float32)
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
            mocks["evaluate_model_performance"].return_value = {
                "error": "diagnostics skipped in test"
            }
            mocks["create_training_plots"].return_value = None

            result = run_training_pipeline(ctx)

        assert result.success is True, result.metadata
        on_disk = json.loads(result.artifact_paths.metadata_path.read_text())

        assert on_disk["task_type"] == "regression"
        assert "class_labels" not in on_disk
        assert "target_distribution" in on_disk
        dist = FrozenDistribution.from_metadata(on_disk["target_distribution"])
        # Round-trips into a usable distribution (doesn't raise, produces a
        # bounded confidence for a representative value).
        confidence = percentile_rank_confidence(0.02, dist)
        assert 0.0 <= confidence <= 1.0

    def test_regression_target_writes_task_type_only(self, tmp_path):
        """The incumbent default target_type must still get task_type
        written (harmless, self-describing) but no class_labels/
        target_distribution (out of scope -- MLBasicSignalGenerator's own
        confidence formula is untouched by this contract)."""
        price_df = _make_ohlcv_df()
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            epochs=1,
            sequence_length=10,
        )
        paths = TrainingPaths(
            project_root=tmp_path, data_dir=tmp_path / "data", models_dir=tmp_path / "models"
        )
        ctx = TrainingContext(config=config, paths=paths)

        stack, mocks = self._pipeline_mocks_except_save_artifacts()
        with stack:
            mocks["download_price_data"].return_value = price_df
            mocks["load_sentiment_data"].return_value = None
            feature_df = price_df.copy()
            feature_df["close_scaled"] = 0.5
            mocks["create_robust_features"].return_value = (
                feature_df,
                {"close": MagicMock()},
                ["close_scaled"],
            )
            sequences = np.random.rand(50, 10, 1).astype(np.float32)
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
            mocks["evaluate_model_performance"].return_value = {
                "error": "diagnostics skipped in test"
            }
            mocks["create_training_plots"].return_value = None

            result = run_training_pipeline(ctx)

        assert result.success is True, result.metadata
        on_disk = json.loads(result.artifact_paths.metadata_path.read_text())
        assert on_disk["task_type"] == "regression"
        assert "class_labels" not in on_disk
        assert "target_distribution" not in on_disk


class TestRealRegistryLoadedOnnxRunnerConsumesMetadata:
    """Consumer half: a REAL PredictionModelRegistry + REAL OnnxRunner
    (only onnxruntime.InferenceSession itself mocked, per this repo's
    existing ONNX unit-test convention), loading a metadata.json shaped
    exactly like the producer writes it."""

    @staticmethod
    def _make_bundle_dir(reg_root, symbol, model_type, version, metadata):
        bundle_dir = reg_root / symbol / model_type / version
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "model.onnx").write_bytes(b"placeholder -- InferenceSession is mocked below")
        (bundle_dir / "metadata.json").write_text(json.dumps(metadata))
        return bundle_dir

    def test_binary_classifier_bundle_predicts_without_raising(self, tmp_path):
        reg_root = tmp_path / "models"
        self._make_bundle_dir(
            reg_root,
            "BTCUSDT",
            "basic_binary_direction",
            "2026-01-01_1h_v1",
            {
                "symbol": "BTCUSDT",
                "model_type": "basic_binary_direction",
                "timeframe": "1h",
                "task_type": "binary_classification",
                "class_labels": [-1, 1],
            },
        )

        with patch("onnxruntime.InferenceSession", return_value=_mock_inference_session([0.82])):
            cfg = PredictionConfig(model_registry_path=str(reg_root))
            registry = PredictionModelRegistry(cfg)
            bundle = registry.select_bundle(
                symbol="BTCUSDT", model_type="basic_binary_direction", timeframe="1h"
            )

            # If the registry/runner metadata wiring were still broken, this
            # would raise ValueError("... no 'class_labels' declared") --
            # OnnxRunner would never have seen class_labels at all (proven
            # by this same setup pre-fix: it raised exactly that).
            features = np.random.rand(1, 120, 5).astype(np.float32)
            prediction = bundle.runner.predict(features)

        assert prediction.probabilities == pytest.approx((0.18, 0.82))
        assert prediction.direction == 1
        assert prediction.confidence == pytest.approx(0.82)

    def test_ternary_classifier_bundle_predicts_without_raising(self, tmp_path):
        reg_root = tmp_path / "models"
        self._make_bundle_dir(
            reg_root,
            "BTCUSDT",
            "basic_triple_barrier",
            "2026-01-01_1h_v1",
            {
                "symbol": "BTCUSDT",
                "model_type": "basic_triple_barrier",
                "timeframe": "1h",
                "task_type": "ternary_classification",
                "class_labels": [-1, 0, 1],
            },
        )

        with patch(
            "onnxruntime.InferenceSession", return_value=_mock_inference_session([0.1, 0.2, 0.7])
        ):
            cfg = PredictionConfig(model_registry_path=str(reg_root))
            registry = PredictionModelRegistry(cfg)
            bundle = registry.select_bundle(
                symbol="BTCUSDT", model_type="basic_triple_barrier", timeframe="1h"
            )

            features = np.random.rand(1, 120, 5).astype(np.float32)
            prediction = bundle.runner.predict(features)

        assert prediction.probabilities == pytest.approx((0.1, 0.2, 0.7))
        assert prediction.direction == 1


class TestExamSignalGeneratorConsumesRealRegistryBundle:
    """Full chain: PredictionConfig -> PredictionEngine -> real registry ->
    real OnnxRunner (InferenceSession mocked) -> ClassificationExamSignalGenerator."""

    def test_classification_exam_signal_generator_end_to_end(self, tmp_path):
        reg_root = tmp_path / "models"
        TestRealRegistryLoadedOnnxRunnerConsumesMetadata._make_bundle_dir(
            reg_root,
            "BTCUSDT",
            "basic_binary_direction",
            "2026-01-01_1h_v1",
            {
                "symbol": "BTCUSDT",
                "model_type": "basic_binary_direction",
                "timeframe": "1h",
                "task_type": "binary_classification",
                "class_labels": [-1, 1],
            },
        )

        with patch("onnxruntime.InferenceSession", return_value=_mock_inference_session([0.9])):
            # Build a real, registry-backed PredictionEngine pointed at the
            # tmp registry, with the same price-only feature pipeline the
            # exam signal generators use (mirrors exam_signal_generator.py's
            # _build_price_only_prediction_engine).
            sequence_length = 120
            engine_config = PredictionConfig(model_registry_path=str(reg_root))
            engine = PredictionEngine(engine_config)
            engine.feature_pipeline = FeaturePipeline(
                config={
                    "technical_features": {"enabled": False},
                    "sentiment_features": {"enabled": False},
                    "market_features": {"enabled": False},
                    "price_only_features": {"enabled": False},
                },
                custom_extractors=[PriceOnlyFeatureExtractor(normalization_window=sequence_length)],
            )

            generator = ClassificationExamSignalGenerator(
                sequence_length=sequence_length,
                model_name="BTCUSDT:1h:basic_binary_direction:2026-01-01_1h_v1",
            )
            generator.prediction_engine = engine  # inject the real, tmp-registry-backed engine

            df = _make_ohlcv_df(periods=150)
            signal = generator.generate_signal(df, 130)

        # The loud-failure/HOLD-degradation path fires when probabilities
        # is None or result.error is set -- assert neither happened.
        assert signal.metadata.get("reason") != "prediction_failed_or_not_classification"
        assert "probabilities" in signal.metadata
        assert signal.metadata["probabilities"] == pytest.approx((0.1, 0.9))
        assert signal.confidence == pytest.approx(0.9)
