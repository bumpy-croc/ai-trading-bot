#!/usr/bin/env python3
"""Entry points for training-related CLI commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore
    callbacks = None  # type: ignore

from src.infrastructure.runtime.paths import get_project_root
from src.ml.training_pipeline import DiagnosticsOptions, TrainingConfig, TrainingContext
from src.ml.training_pipeline.gpu_config import configure_gpu
from src.ml.training_pipeline.pipeline import TrainingResult, run_training_pipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.trading.symbols.factory import SymbolFactory

PROJECT_ROOT = get_project_root()
MODEL_REGISTRY = PROJECT_ROOT / "src" / "ml" / "models"


def _parse_dates(start: str, end: str) -> Tuple[datetime, datetime]:
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date format: {exc}") from exc
    if start_dt >= end_dt:
        raise ValueError("start-date must be before end-date")
    return start_dt, end_dt


def _diagnostics_from_args(args) -> DiagnosticsOptions:
    return DiagnosticsOptions(
        generate_plots=not getattr(args, "skip_plots", False),
        evaluate_robustness=not getattr(args, "skip_robustness", False),
        convert_to_onnx=not getattr(args, "skip_onnx", False),
    )


def train_model_main(args) -> int:
    """Train a combined model."""
    if not _TENSORFLOW_AVAILABLE:
        print("❌ tensorflow is required for model training but is not installed.")
        print("Install it with: pip install tensorflow")
        return 1

    # Handle preset if provided
    if hasattr(args, "preset") and args.preset:
        from src.ml.training_pipeline.presets import create_config_from_preset

        # Build kwargs for preset overrides
        overrides = {}
        if hasattr(args, "model_type") and args.model_type:
            overrides["model_type"] = args.model_type

        config = create_config_from_preset(
            args.preset,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=None,  # Will use preset defaults
            end_date=None,
            force_sentiment=args.force_sentiment,
            force_price_only=args.force_price_only,
            **overrides,
        )
        print(f"📋 Using '{args.preset}' preset")
    else:
        # Manual configuration (original behavior)
        try:
            start_date, end_date = _parse_dates(args.start_date, args.end_date)
        except ValueError as exc:
            print(f"❌ {exc}")
            return 1

        model_type = getattr(args, "model_type", "balanced")

        config = TrainingConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            force_sentiment=args.force_sentiment,
            force_price_only=args.force_price_only,
            mixed_precision=not getattr(args, "disable_mixed_precision", False),
            model_type=model_type,
            diagnostics=_diagnostics_from_args(args),
        )

    ctx = TrainingContext(config=config)
    print(
        "🚀 Starting training for",
        f"{config.symbol} ({config.timeframe})",
        f"model={config.model_type}",
        f"epochs={config.epochs}",
        f"seq_len={config.sequence_length}",
        f"batch_size={config.batch_size}",
        sep=" ",
    )

    result: TrainingResult = run_training_pipeline(ctx)
    if not result.success:
        print(f"❌ Training failed: {result.metadata.get('error')}")
        return 1

    print("✅ Training complete in %.1fs" % result.duration_seconds)
    eval_results = result.metadata.get("evaluation_results", {})
    if eval_results:
        print(
            "📊 Test RMSE: %.6f | MAPE: %.2f%%"
            % (eval_results.get("test_rmse", 0.0), eval_results.get("mape", 0.0))
        )
    artifacts = result.artifact_paths
    if artifacts:
        print(f"Keras model: {artifacts.keras_path}")
        if artifacts.onnx_path:
            print(f"ONNX model: {artifacts.onnx_path}")
        print(f"Metadata: {artifacts.metadata_path}")
        if artifacts.plot_path:
            print(f"Training plot: {artifacts.plot_path}")
    return 0


# --- Price-only training (legacy path) ---------------------------------------------------------


def _prepare_price_only_sequences(
    df: pd.DataFrame,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    extractor = PriceOnlyFeatureExtractor(normalization_window=sequence_length)
    enriched = extractor.extract(df.copy())
    feature_cols = [
        "close_normalized",
        "volume_normalized",
        "high_normalized",
        "low_normalized",
        "open_normalized",
    ]
    missing = [col for col in feature_cols if col not in enriched.columns]
    if missing:
        raise ValueError(f"Missing normalized price columns: {missing}")

    enriched = enriched.dropna(subset=feature_cols)
    if len(enriched) <= sequence_length:
        raise ValueError("Insufficient rows after normalization for sequence construction")

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    values = enriched[feature_cols].to_numpy(dtype=np.float32)
    targets = enriched["close_normalized"].to_numpy(dtype=np.float32)
    for idx in range(sequence_length, len(enriched)):
        X_list.append(values[idx - sequence_length : idx])
        y_list.append(targets[idx])

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, feature_cols


def _build_price_only_model(sequence_length: int, num_features: int):
    """Build a price-only model."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model building but is not installed. "
            "Install it with: pip install tensorflow"
        )
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features))
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def _download_price_frame(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    from argparse import Namespace

    from cli.commands import data as data_commands

    symbol_exchange = SymbolFactory.to_exchange_symbol(symbol, "binance")
    ns = Namespace(
        symbol=symbol_exchange,
        timeframe=timeframe,
        start_date=start,
        end_date=end,
        output_dir=str(PROJECT_ROOT / "data"),
        format="csv",
    )
    status = data_commands._download(ns)
    if status != 0:
        raise RuntimeError("Failed to download price data")
    pattern = f"{symbol_exchange}_{timeframe}_{start}_{end}.*"
    files = sorted(
        (PROJECT_ROOT / "data").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not files:
        raise FileNotFoundError("Downloaded file not found")
    latest = files[0]
    df = pd.read_feather(latest) if latest.suffix == ".feather" else pd.read_csv(latest)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def train_price_model_main(args) -> int:
    """Train a price model."""
    if not _TENSORFLOW_AVAILABLE:
        print("❌ tensorflow is required for model training but is not installed.")
        print("Install it with: pip install tensorflow")
        return 1

    # Configure GPU early
    device_name = configure_gpu()
    if device_name:
        print(f"🚀 Using device: {device_name}")
    else:
        print("🚀 Using CPU")

    sequence_length = args.sequence_length
    try:
        start_date, end_date = _parse_dates(args.start_date, args.end_date)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    try:
        price_df = _download_price_frame(args.symbol, args.timeframe, start_str, end_str)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to download price data: {exc}")
        return 1

    try:
        X, y, feature_cols = _prepare_price_only_sequences(price_df.astype(float), sequence_length)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to prepare sequences: {exc}")
        return 1

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = _build_price_only_model(sequence_length, len(feature_cols))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
            ),
        ],
        verbose=1,
    )

    # Evaluate model performance
    train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_rmse = model.evaluate(X_val, y_val, verbose=0)
    test_predictions = model.predict(X_val, verbose=0)
    # MAPE calculation (targets are normalized, so MAPE is relative to normalized range)
    mape = float(np.mean(np.abs((y_val - test_predictions.flatten()) / (y_val + 1e-8))) * 100)

    version_id = datetime.utcnow().strftime("%Y-%m-%d_%Hh_v1")
    metadata = {
        "model_id": f"{args.symbol.lower()}_price_v3",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_type": "basic",
        "version_id": version_id,
        "framework": "onnx",
        "model_file": "model.onnx",
        "created_at": datetime.utcnow().isoformat(),
        "sequence_length": sequence_length,
        "feature_names": feature_cols,
        "feature_strategy": "price_only_rolling_minmax",
        "price_normalization": {
            "method": "rolling_minmax",
            "window": sequence_length,
            "target_feature": "close",
        },
        "training_params": {
            "epochs": len(history.history.get("loss", [])),
            "batch_size": args.batch_size,
            "timeframe": args.timeframe,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "evaluation_results": {
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "mape": mape,
        },
        "dataset": {
            "row_count": int(len(price_df)),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
        },
    }

    registry_root = MODEL_REGISTRY
    bundle_dir = registry_root / args.symbol / "basic" / version_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save(bundle_dir / "model.keras")

    from src.ml.training_pipeline.artifacts import convert_to_onnx

    convert_to_onnx(model, bundle_dir / "model.onnx")

    with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    schema = {
        "sequence_length": sequence_length,
        "features": [{"name": name, "required": True} for name in feature_cols],
    }
    with open(bundle_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    latest_link = bundle_dir.parent / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        try:
            latest_link.unlink()
        except OSError:
            pass
    latest_link.symlink_to(version_id)

    print(f"✅ Saved bundle to {bundle_dir}")
    return 0


def train_price_only_model_main(args) -> int:
    print(f"🚀 Price-Only Model Training for {args.symbol}")
    print("Price-only training not yet implemented")
    return 1


def simple_model_validator_main(args) -> int:
    print(f"🔍 Simple Model Validation for {args.symbol}")
    print("Model validation not yet implemented")
    return 1
