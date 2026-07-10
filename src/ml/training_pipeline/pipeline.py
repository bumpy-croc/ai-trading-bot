"""High-level orchestration for training models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    if not TYPE_CHECKING:
        tf = None

from src.config.constants import (
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
)
from src.ml.training_pipeline import TrainingContext
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
from src.ml.training_pipeline.labels import (
    LabelResult,
    binary_direction_labels,
    smoothed_forward_return_labels,
    triple_barrier_labels,
)
from src.ml.training_pipeline.meta_labels import (
    META_LABEL_FEATURE_ORDER,
    build_meta_label_features,
    encode_meta_label_features_for_training,
    resolve_fired_trade,
    run_primary_signal_forward,
)
from src.ml.training_pipeline.models import create_model, get_model_callbacks
from src.ml.training_pipeline.sklearn_onnx_export import export_logistic_regression_to_onnx
from src.ml.training_pipeline.task_types import (
    TaskType,
    get_model_task_type,
    get_target_class_labels,
    validate_target_head_compatibility,
)
from src.prediction.distribution_stats import FrozenDistribution
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator

logger = logging.getLogger(__name__)

# Version generation constants
MAX_VERSION_GENERATION_RETRIES = 1000  # Allows ~40 versions/hour for 24 hours before collision


@dataclass
class TrainingResult:
    success: bool
    metadata: dict
    artifact_paths: ArtifactPaths | None
    duration_seconds: float


def _generate_version_id(models_dir: Path, symbol: str, model_type: str) -> str:
    """Generate auto-incrementing version ID to prevent collisions.

    Checks if version directory exists and increments counter until finding
    a unique version ID. Format: YYYY-MM-DD_HHhMMmSSs_vN where N starts at 1.

    Second-level granularity keeps IDs unique across cloud training containers,
    where each job sees an empty models dir and the exists() counter cannot
    de-duplicate against sibling jobs started in the same hour.

    Args:
        models_dir: Root models directory
        symbol: Trading symbol (e.g., BTCUSDT)
        model_type: Model type (e.g., basic, sentiment)

    Returns:
        Unique version ID string

    Raises:
        RuntimeError: If unable to generate unique version ID after max retries
    """
    base_timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%Hh%Mm%Ss")
    version_counter = 1

    while version_counter <= MAX_VERSION_GENERATION_RETRIES:
        version_id = f"{base_timestamp}_v{version_counter}"
        target_dir = models_dir / symbol.upper() / model_type / version_id
        if not target_dir.exists():
            return version_id
        version_counter += 1

    raise RuntimeError(
        f"Failed to generate unique version ID after {MAX_VERSION_GENERATION_RETRIES} attempts "
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
        except (RuntimeError, ValueError, AttributeError) as exc:
            # XLA may not be available on all platforms (e.g., MPS on Apple Silicon)
            logger.info("Enabled mixed precision (XLA not available: %s)", exc)
    except (RuntimeError, ValueError) as exc:
        # Mixed precision is an optimization - training should continue with regular precision
        # if setup fails (e.g., GPU driver issues, TensorFlow version incompatibility)
        logger.warning("Failed to enable mixed precision: %s", exc)
        logger.debug("Full exception trace:", exc_info=True)


def _build_alt_target(
    target_type: str,
    target_horizon: int,
    merged_df: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an alternative-target label array aligned to feature_data's rows.

    Labels are computed on the full ``merged_df`` timeline (the raw OHLCV
    data, before feature-engineering warmup rows are dropped) so forward-
    horizon lookups see genuinely future bars, then reindexed onto
    ``feature_data``'s row index -- feature engineering may have dropped
    leading rows for rolling-window warmup, and the target array must stay
    row-aligned with the feature array `create_sequences` will pair it with.

    Args:
        target_type: One of "binary_direction", "smoothed_return",
            "triple_barrier" (never "regression" -- callers keep the
            existing unconditional close-price target for that case).
        target_horizon: Forward horizon in bars (binary_direction /
            smoothed_return only; triple_barrier uses the ratified
            risk-limits.json stop/take-profit/max-holding-hours defaults
            instead, per the TARGET-REDESIGN tournament preregistration §2c).
        merged_df: Raw OHLCV DataFrame (pre-feature-engineering).
        feature_data: Post-feature-engineering DataFrame to align labels to.

    Returns:
        Tuple of (target_array, valid_mask), both aligned to feature_data's
        row order. valid_mask is False for any row without a fully-realized
        forward label (trailing rows, or rows feature engineering added that
        merged_df never had) -- callers must drop those rows from BOTH the
        feature and target arrays before training, never zero-fill them.
    """
    label: LabelResult
    if target_type == "binary_direction":
        label = binary_direction_labels(merged_df["close"], horizon=target_horizon)
    elif target_type == "smoothed_return":
        label = smoothed_forward_return_labels(merged_df["close"], horizon=target_horizon)
    elif target_type == "triple_barrier":
        label = triple_barrier_labels(
            merged_df,
            take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
            stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
            # max_holding_bars assumes 1 bar == 1 hour (the tournament's
            # fixed 1h timeframe); a genuine bars-per-hour conversion for
            # other timeframes is out of scope for this scaffolding.
            max_holding_bars=DEFAULT_MAX_HOLDING_HOURS,
        )
        # triple_barrier_labels() returns raw direction values {-1, 0, 1}
        # (the same convention get_target_class_labels("triple_barrier")
        # documents for INFERENCE-time decoding: argmax index -> class_labels
        # [index] is the direction). sparse_categorical_crossentropy, which
        # tft_ternary is compiled with, requires integer class INDICES in
        # [0, num_classes) instead -- feeding raw -1 would be an invalid
        # class index. Remap through the same class_labels list used at
        # inference time so training encoding and inference decoding can
        # never drift apart.
        class_labels = get_target_class_labels(target_type)
        assert class_labels is not None  # triple_barrier always has class_labels
        label_to_index = {direction: index for index, direction in enumerate(class_labels)}
        label = LabelResult(
            values=np.array([label_to_index[int(v)] for v in label.values], dtype=np.int64),
            valid_mask=label.valid_mask,
            horizon_bars=label.horizon_bars,
        )
    else:
        raise ValueError(f"No label generator wired for target_type={target_type!r}")

    label_values = pd.Series(label.values, index=merged_df.index)
    label_valid = pd.Series(label.valid_mask, index=merged_df.index)

    aligned_values = label_values.reindex(feature_data.index).to_numpy(dtype=np.float32)
    aligned_valid = label_valid.reindex(feature_data.index).fillna(False).to_numpy(dtype=bool)

    return aligned_values, aligned_valid


def _run_meta_label_pipeline(ctx: TrainingContext) -> TrainingResult:
    """Train entrant (a)'s meta-labeling classifier (TARGET-REDESIGN
    tournament preregistration §2a).

    A separate, sklearn-only data/training flow from the TF architectures
    above -- ``ctx.config.model_type`` is irrelevant here (always trains a
    ``sklearn.linear_model.LogisticRegression`` regardless of --model-type;
    see train.py's --model-type help text). Steps: (1) download price data,
    (2) run the ALREADY-TRAINED primary signal (``primary_model_type``)
    forward to find its fire points, (3) resolve each fire's trade outcome
    through the exam-harness exit geometry, (4) build + encode the
    meta-label feature set, (5) fit LogisticRegression, (6) hand-roll an
    ONNX export (sklearn_onnx_export.py), (7) save artifacts under
    ``{symbol}/meta_label/{version}/`` with an atomic ``latest`` symlink,
    mirroring artifacts.py's save_artifacts pattern.
    """
    start_time = perf_counter()
    try:
        price_df = download_price_data(ctx)

        primary_signal_generator = MLBasicSignalGenerator(
            model_type=ctx.config.primary_model_type,
            timeframe=ctx.config.timeframe,
            symbol=ctx.config.symbol,
        )

        fired = run_primary_signal_forward(primary_signal_generator, price_df)
        if not fired:
            raise ValueError(
                f"Primary signal (model_type={ctx.config.primary_model_type!r}) produced "
                f"zero fires over {ctx.config.start_date} to {ctx.config.end_date} -- "
                f"nothing to meta-label. Try a longer window or a different primary model."
            )

        resolved_fires = []
        resolutions = []
        for fire in fired:
            resolution = resolve_fired_trade(
                price_df,
                fire.index,
                fire.direction,
                take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
                stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
                max_holding_bars=DEFAULT_MAX_HOLDING_HOURS,
            )
            # Unresolved (truncated at series end) fires are dropped, never
            # treated as unprofitable -- see resolve_fired_trade's contract.
            if resolution is not None:
                resolved_fires.append(fire)
                resolutions.append(resolution)

        if not resolved_fires:
            raise ValueError(
                f"{len(fired)} primary signal fire(s) found, but none resolved within "
                f"the training window (all truncated at series end) -- nothing to "
                f"meta-label. Try a longer window."
            )

        features_df = build_meta_label_features(price_df, resolved_fires, resolutions)
        X, y = encode_meta_label_features_for_training(features_df)

        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Meta-label training data has only one class present across "
                f"{len(resolved_fires)} resolved fire(s) (all profitable or all "
                f"unprofitable) -- LogisticRegression cannot fit a decision boundary. "
                f"Try a longer/different training window."
            )

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        train_accuracy = float(model.score(X, y))

        # meta_label_logistic is the model_type task_types.py declares for
        # this classifier (see its docstring: not create_model()-dispatched,
        # but still declared for #947-guard/metadata-convention uniformity).
        registry_model_type = "meta_label"
        metadata = {
            "symbol": ctx.config.symbol,
            "model_type": registry_model_type,
            "training_date": pd.Timestamp.now(tz="UTC").isoformat(),
            "feature_names": list(META_LABEL_FEATURE_ORDER),
            "sequence_length": 1,  # tabular features, not sequence-shaped
            "task_type": TaskType.BINARY_CLASSIFICATION.value,
            # meta_label's classifier output is P(primary signal's fired
            # trade profitable), NOT a direction prediction -- class_labels=
            # [0, 1] here means "not profitable"/"profitable". Consumers
            # must read probabilities[1] (P(profitable)) directly, never
            # `direction`/`confidence` the way direction-value classifiers
            # (binary_direction/triple_barrier) are interpreted -- see
            # task_types.py's TARGET_CLASS_LABELS docstring and
            # MetaLabelExamSignalGenerator.
            "class_labels": [0, 1],
            "primary_model_type": ctx.config.primary_model_type,
            "training_params": {
                "target_type": "meta_label",
                "timeframe": ctx.config.timeframe,
                "start_date": ctx.config.start_date.isoformat(),
                "end_date": ctx.config.end_date.isoformat(),
                "primary_model_type": ctx.config.primary_model_type,
                "n_fires": len(fired),
                "n_resolved": len(resolved_fires),
            },
            "evaluation_results": {"train_accuracy": train_accuracy},
        }

        output_dir = ctx.paths.models_dir
        version_id = _generate_version_id(output_dir, ctx.config.symbol, registry_model_type)
        metadata["version_id"] = version_id

        type_dir = output_dir / ctx.config.symbol.upper() / registry_model_type
        version_dir = type_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.pkl"
        import joblib

        joblib.dump(model, model_path)

        onnx_path = version_dir / "model.onnx"
        export_logistic_regression_to_onnx(
            model, n_features=len(META_LABEL_FEATURE_ORDER), output_path=onnx_path
        )

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        feature_schema = {
            "sequence_length": 1,
            "features": [{"name": name, "required": True} for name in META_LABEL_FEATURE_ORDER],
        }
        feature_schema_path = version_dir / "feature_schema.json"
        with open(feature_schema_path, "w", encoding="utf-8") as f:
            json.dump(feature_schema, f, indent=2)

        # Atomic latest symlink update -- mirrors save_artifacts' TOCTOU-safe
        # temp-symlink-then-replace pattern.
        latest_link = type_dir / "latest"
        temp_link = type_dir / f".latest.{version_dir.name}.tmp"
        try:
            if temp_link.exists() or temp_link.is_symlink():
                temp_link.unlink()
            temp_link.symlink_to(version_dir.name)
            temp_link.replace(latest_link)
        except OSError as e:
            logger.warning("Failed to update latest symlink for meta_label bundle: %s", e)

        artifact_paths = ArtifactPaths(
            directory=version_dir,
            keras_path=model_path,  # native (non-ONNX) artifact; sklearn, not Keras
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            plot_path=None,
        )

        duration = perf_counter() - start_time
        logger.info(
            "Meta-label training complete: %d fires, %d resolved, train_accuracy=%.3f",
            len(fired),
            len(resolved_fires),
            train_accuracy,
        )
        return TrainingResult(
            success=True,
            metadata=metadata,
            artifact_paths=artifact_paths,
            duration_seconds=duration,
        )

    except Exception as exc:
        duration = perf_counter() - start_time
        logger.error("Meta-label training failed: %s", exc, exc_info=True)
        return TrainingResult(
            success=False,
            metadata={"error": str(exc)},
            artifact_paths=None,
            duration_seconds=duration,
        )


def run_training_pipeline(ctx: TrainingContext) -> TrainingResult:
    """Run the training pipeline."""
    if ctx.config.target_type == "meta_label":
        # Sklearn-only path -- dispatched BEFORE the TensorFlow availability
        # check (meta_label needs no TF at all) and BEFORE
        # validate_target_head_compatibility (ctx.config.model_type is
        # irrelevant for meta_label; the #947 guard's model_type/target_type
        # cross-check doesn't apply to this separate training flow).
        return _run_meta_label_pipeline(ctx)
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model training but is not installed. "
            "Install it with: pip install tensorflow"
        )
    start_time = perf_counter()

    # Configure GPU early in the pipeline
    device_name = configure_gpu()
    if device_name:
        logger.info("[GPU] Training will use device: %s", device_name)
    else:
        logger.info("[CPU] Training will use CPU")
    try:
        # #947 guard: refuse loudly, before any download/GPU work is spent,
        # when model_type's compiled head is incompatible with target_type
        # (e.g. tft's binary sigmoid/BCE head trained against the regression
        # close target -- previously silent, garbage-model failure).
        validate_target_head_compatibility(ctx.config.model_type, ctx.config.target_type)

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

        if ctx.config.force_price_only:
            # Route through the same 5-feature, causally-normalized extractor that
            # `atb train price` and production inference use. Historically this flag
            # only skipped the sentiment download while still building
            # create_robust_features' 9-feature globally-scaled pipeline, silently
            # diverging from the price-only contract.
            extractor = PriceOnlyFeatureExtractor(normalization_window=ctx.config.sequence_length)
            feature_data = extractor.extract(merged_df.copy())
            feature_names = extractor.get_feature_names()
            feature_data = feature_data.dropna(subset=feature_names)
            scalers: dict[str, MinMaxScaler] = {}
            # Predict the rolling-normalized close, matching `atb train price`. The raw
            # close would regress dollar magnitudes with no denormalization step
            # (scalers is empty), making RMSE/MAPE incomparable with existing
            # price-only baselines.
            target_array = feature_data["close_normalized"].to_numpy(dtype=np.float32)
        else:
            feature_data, scalers, feature_names = create_robust_features(
                merged_df.copy(), sentiment_assessment, ctx.config.sequence_length
            )
            target_array = feature_data["close"].to_numpy(dtype=np.float32)

        # Alternative training targets (TARGET-REDESIGN tournament, #933
        # Phase 2): derive the label from merged_df's raw OHLCV data instead
        # of the incumbent regression close target, then drop rows without a
        # fully-realized forward label from BOTH arrays before sequencing.
        if ctx.config.target_type != "regression":
            target_array, valid_mask = _build_alt_target(
                ctx.config.target_type,
                ctx.config.target_horizon,
                merged_df,
                feature_data,
            )
            feature_data = feature_data[valid_mask]
            target_array = target_array[valid_mask]

        feature_array = feature_data[feature_names].to_numpy(dtype=np.float32)
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

        # Diagnostics run after model.fit() but before save_artifacts. A crash here
        # (e.g. a per-architecture metric-set mismatch) must not discard a fully-trained
        # model with nothing persisted -- degrade to a metadata gap instead, and let only
        # a genuine training/data failure (the outer except below) abort without saving.
        robustness_results: Any
        evaluation_results: Any
        try:
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
        except (ValueError, RuntimeError, KeyError) as exc:
            logger.error(
                "Post-training evaluation/robustness diagnostics failed: %s. "
                "Saving the trained model anyway with a diagnostics-error placeholder.",
                exc,
                exc_info=True,
            )
            robustness_results = {}
            evaluation_results = {"error": str(exc)}

        # Classification/distribution metadata contract (TARGET-REDESIGN
        # tournament, #933 Phase 2): the producer half of what OnnxRunner
        # (task_type/class_labels) and SmoothedReturnExamSignalGenerator
        # (target_distribution) consume. task_type is always written
        # (get_model_task_type already validated compatible with
        # ctx.config.target_type by the #947 guard at the top of this
        # function, so it never raises here). class_labels only applies to
        # classification target types. target_distribution is computed ONCE
        # from y_train ONLY (never X_val/y_val) -- the harness-wide rule
        # that the confidence mapping must never see eval-window data.
        # Diagnostics-only: a failure here degrades gracefully (same
        # pattern as the evaluation/robustness block above) rather than
        # discarding an already-trained model.
        target_type_metadata: dict[str, Any] = {
            "task_type": get_model_task_type(ctx.config.model_type).value
        }
        class_labels = get_target_class_labels(ctx.config.target_type)
        if class_labels is not None:
            target_type_metadata["class_labels"] = class_labels
        if ctx.config.target_type == "smoothed_return":
            try:
                target_type_metadata["target_distribution"] = FrozenDistribution.from_samples(
                    np.abs(y_train)
                ).to_metadata()
            except ValueError as exc:
                logger.warning(
                    "Failed to compute target_distribution metadata for smoothed_return: "
                    "%s. Entrant (d)'s percentile-rank confidence will be unavailable for "
                    "this model until retrained.",
                    exc,
                )

        # force_price_only bundles stay in price/ even though they share `atb train
        # price`'s basic/ contract: writing to basic/ here would implicitly repoint
        # basic/latest — the symlink live strategies load. Promotion into basic/ is
        # always an explicit step (atb train cloud-promote / atb models promote).
        metadata = {
            "symbol": ctx.config.symbol,
            "model_type": "sentiment" if has_sentiment else "price",
            "training_date": pd.Timestamp.now(tz="UTC").isoformat(),
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
                "target_type": ctx.config.target_type,
            },
            "evaluation_results": evaluation_results,
            "diagnostics": {
                "plots": ctx.config.diagnostics.generate_plots,
                "robustness": ctx.config.diagnostics.evaluate_robustness,
                "onnx": ctx.config.diagnostics.convert_to_onnx,
            },
            **target_type_metadata,
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
    except (ValueError, RuntimeError, ImportError, OSError, KeyError) as exc:
        # Catch specific expected failure modes in the training pipeline
        # ValueError: Data validation failures, insufficient data, invalid configuration
        # RuntimeError: TensorFlow training failures, version ID generation failures
        # ImportError: Missing required packages (tensorflow, tf2onnx)
        # OSError: File system errors (permissions, disk space)
        # KeyError: Missing required configuration or metadata fields
        logger.error("Training pipeline failed: %s", exc, exc_info=True)
        return TrainingResult(False, {"error": str(exc)}, None, perf_counter() - start_time)
    except Exception as exc:
        # Unexpected errors - log with full traceback for debugging
        logger.exception("Training pipeline failed with unexpected error: %s", exc)
        return TrainingResult(
            False, {"error": f"Unexpected error: {exc}"}, None, perf_counter() - start_time
        )
