"""High-level orchestration for training models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore

from src.ml.training_pipeline import DiagnosticsOptions, TrainingConfig, TrainingContext
from src.ml.training_pipeline.artifacts import (
    ArtifactPaths,
    create_training_plots,
    evaluate_model_performance,
    save_artifacts,
    validate_model_robustness,
)
from src.ml.training_pipeline.datasets import build_tf_datasets, create_sequences, split_sequences
from src.ml.training_pipeline.features import (
    assess_sentiment_data_quality,
    create_robust_features,
    merge_price_sentiment_data,
)
from src.ml.training_pipeline.gpu_config import configure_gpu
from src.ml.training_pipeline.ingestion import download_price_data, load_sentiment_data
from src.ml.training_pipeline.models import create_model, get_model_callbacks

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    success: bool
    metadata: dict
    artifact_paths: Optional[ArtifactPaths]
    duration_seconds: float


def _generate_version_id(models_dir: Path, symbol: str, model_type: str) -> str:
    """Generate auto-incrementing version ID to prevent collisions.

    Checks if version directory exists and increments counter until finding
    a unique version ID. Format: YYYY-MM-DD_HHh_vN where N starts at 1.

    Args:
        models_dir: Root models directory
        symbol: Trading symbol (e.g., BTCUSDT)
        model_type: Model type (e.g., basic, sentiment)

    Returns:
        Unique version ID string

    Raises:
        RuntimeError: If unable to generate unique version ID after max retries
    """
    base_timestamp = datetime.utcnow().strftime("%Y-%m-%d_%Hh")
    version_counter = 1
    max_retries = 1000

    while version_counter <= max_retries:
        version_id = f"{base_timestamp}_v{version_counter}"
        target_dir = models_dir / symbol.upper() / model_type / version_id
        if not target_dir.exists():
            return version_id
        version_counter += 1

    raise RuntimeError(
        f"Failed to generate unique version ID after {max_retries} attempts "
        f"for {symbol} {model_type}. Check disk space and permissions."
    )


def enable_mixed_precision(enabled: bool) -> None:
    """Enable mixed precision training for faster GPU performance.

    Args:
        enabled: Whether to enable mixed precision training
    """
    if not _TENSORFLOW_AVAILABLE:
        return
    if not enabled:
        logger.info("Mixed precision explicitly disabled via configuration")
        return
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("Mixed precision disabled: no GPU detected")
        return
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        # XLA JIT compilation may not be fully supported on MPS, so we catch errors
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("Enabled mixed precision and XLA JIT compilation")
        except Exception:  # noqa: BLE001
            # XLA may not be available on all platforms (e.g., MPS)
            logger.info("Enabled mixed precision (XLA not available on this platform)")
    except (RuntimeError, ValueError) as exc:
        # Mixed precision is an optimization - training should continue with regular precision
        # if setup fails (e.g., GPU driver issues, TensorFlow version incompatibility)
        logger.warning("Failed to enable mixed precision: %s", exc)
        logger.debug("Full exception trace:", exc_info=True)


def run_training_pipeline(ctx: TrainingContext) -> TrainingResult:
    """Run the training pipeline."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model training but is not installed. "
            "Install it with: pip install tensorflow"
        )
    start_time = perf_counter()

    # Configure GPU early in the pipeline
    device_name = configure_gpu()
    if device_name:
        logger.info(f"🚀 Training will use device: {device_name}")
    else:
        logger.info("🚀 Training will use CPU")
    try:
        price_df = download_price_data(ctx)
        sentiment_df = load_sentiment_data(ctx)
        if sentiment_df is None or sentiment_df.empty:
            sentiment_assessment = {
                "recommendation": "price_only",
                "quality_score": 0.0,
                "reason": "Sentiment disabled or unavailable",
            }
            merged_df = price_df.copy()
        else:
            if sentiment_df.index.tz is not None and price_df.index.tz is None:
                logger.info("Localizing price data to UTC to match sentiment index")
                try:
                    price_df = price_df.tz_localize("UTC")
                except ValueError as exc:
                    logger.error("Failed to localize price data: %s", exc)
                    raise ValueError("Price data timezone localization failed") from exc
            sentiment_assessment = assess_sentiment_data_quality(sentiment_df, price_df)
            merged_df = merge_price_sentiment_data(price_df, sentiment_df, ctx.config.timeframe)

        if ctx.config.force_sentiment and sentiment_df is not None:
            sentiment_assessment["recommendation"] = "full_sentiment"

        if sentiment_assessment["recommendation"] not in ["full_sentiment", "hybrid_with_fallback"]:
            logger.info("Using price-only dataset based on sentiment assessment")
            merged_df = price_df.copy()

        # Validate that merged data is non-empty before proceeding
        if merged_df.empty:
            raise ValueError(
                f"No data available after merging price and sentiment data for {ctx.config.symbol}. "
                "Check data availability and timeframe alignment."
            )

        feature_data, scalers, feature_names = create_robust_features(
            merged_df.copy(), sentiment_assessment, ctx.config.sequence_length
        )
        feature_array = feature_data[feature_names].to_numpy(dtype=np.float32)
        target_array = feature_data["close"].to_numpy(dtype=np.float32)
        sequences, targets = create_sequences(
            feature_array,
            target_array,
            ctx.config.sequence_length,
        )
        X_train, y_train, X_val, y_val = split_sequences(sequences, targets)
        train_ds, val_ds = build_tf_datasets(
            X_train,
            y_train,
            X_val,
            y_val,
            ctx.config.batch_size,
        )

        enable_mixed_precision(ctx.config.mixed_precision)

        has_sentiment = sentiment_assessment["recommendation"] in [
            "full_sentiment",
            "hybrid_with_fallback",
        ]

        # Create model using factory (supports multiple architectures)
        model = create_model(
            model_type=ctx.config.model_type,
            input_shape=(ctx.config.sequence_length, len(feature_names)),
            variant=ctx.config.model_variant,
            has_sentiment=has_sentiment,  # For CNN-LSTM compatibility
        )

        # Get model-specific callbacks
        callbacks_list = get_model_callbacks(ctx.config.model_type, patience=15)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ctx.config.epochs,
            callbacks=callbacks_list,
            verbose=1,
        )

        robustness_results = (
            validate_model_robustness(model, X_val, y_val, feature_names, has_sentiment)
            if ctx.config.diagnostics.evaluate_robustness
            else {}
        )
        evaluation_results = evaluate_model_performance(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            scalers.get("close"),
        )

        metadata = {
            "symbol": ctx.config.symbol,
            "model_type": "sentiment" if has_sentiment else "price",
            "training_date": pd.Timestamp.now().isoformat(),
            "feature_names": feature_names,
            "sequence_length": ctx.config.sequence_length,
            "has_sentiment": has_sentiment,
            "sentiment_assessment": sentiment_assessment,
            "robustness_results": robustness_results,
            "training_params": {
                "epochs": ctx.config.epochs,
                "batch_size": ctx.config.batch_size,
                "timeframe": ctx.config.timeframe,
                "start_date": ctx.config.start_date.isoformat(),
                "end_date": ctx.config.end_date.isoformat(),
                "architecture": ctx.config.model_type,  # New: model architecture
                "architecture_variant": ctx.config.model_variant,  # New: architecture variant
            },
            "evaluation_results": evaluation_results,
            "diagnostics": {
                "plots": ctx.config.diagnostics.generate_plots,
                "robustness": ctx.config.diagnostics.evaluate_robustness,
                "onnx": ctx.config.diagnostics.convert_to_onnx,
            },
        }

        output_dir = ctx.paths.models_dir
        version_id = _generate_version_id(output_dir, ctx.config.symbol, metadata["model_type"])
        metadata["version_id"] = version_id
        artifact_paths = save_artifacts(
            model,
            ctx.config.symbol,
            metadata["model_type"],
            output_dir,
            metadata,
            version_id,
            ctx.config.diagnostics.convert_to_onnx,
        )

        artifact_paths.plot_path = create_training_plots(
            history,
            model,
            X_val,
            y_val,
            feature_names,
            ctx.config.symbol,
            metadata["model_type"],
            artifact_paths.directory,
            ctx.config.diagnostics.generate_plots,
        )

        duration = perf_counter() - start_time
        return TrainingResult(True, metadata, artifact_paths, duration)
    except Exception as exc:  # noqa: BLE001 - Catch all pipeline errors for graceful degradation
        # Top-level handler ensures pipeline always returns TrainingResult instead of crashing
        # Enables proper cleanup, error reporting, and CLI error handling for any failure
        logger.error("Training pipeline failed: %s", exc)
        return TrainingResult(False, {"error": str(exc)}, None, perf_counter() - start_time)
