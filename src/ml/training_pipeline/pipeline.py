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
import tensorflow as tf

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
from src.ml.training_pipeline.ingestion import download_price_data, load_sentiment_data
from src.ml.training_pipeline.models import create_adaptive_model, default_callbacks

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
    """
    base_timestamp = datetime.utcnow().strftime("%Y-%m-%d_%Hh")
    version_counter = 1

    while True:
        version_id = f"{base_timestamp}_v{version_counter}"
        target_dir = models_dir / symbol.upper() / model_type / version_id
        if not target_dir.exists():
            return version_id
        version_counter += 1


def enable_mixed_precision(enabled: bool) -> None:
    if not enabled:
        return
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("Mixed precision disabled: no GPU detected")
        return
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        tf.config.optimizer.set_jit(True)
        logger.info("Enabled mixed precision and XLA")
    except Exception as exc:  # noqa: BLE001 - Catch all TensorFlow configuration errors
        # Mixed precision is an optimization - training should continue with regular precision
        # if setup fails (e.g., GPU driver issues, TensorFlow version incompatibility)
        logger.warning("Failed to enable mixed precision: %s", exc)


def run_training_pipeline(ctx: TrainingContext) -> TrainingResult:
    start_time = perf_counter()
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
                price_df = price_df.tz_localize("UTC")
            sentiment_assessment = assess_sentiment_data_quality(sentiment_df, price_df)
            merged_df = merge_price_sentiment_data(price_df, sentiment_df, ctx.config.timeframe)

        if ctx.config.force_sentiment and sentiment_df is not None:
            sentiment_assessment["recommendation"] = "full_sentiment"

        if sentiment_assessment["recommendation"] not in ["full_sentiment", "hybrid_with_fallback"]:
            logger.info("Using price-only dataset based on sentiment assessment")
            merged_df = price_df.copy()

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
        model = create_adaptive_model((ctx.config.sequence_length, len(feature_names)), has_sentiment)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ctx.config.epochs,
            callbacks=default_callbacks(),
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
