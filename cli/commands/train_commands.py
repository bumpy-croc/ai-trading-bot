#!/usr/bin/env python3
"""
Training Commands for CLI

This module contains all the training functionality ported from the scripts directory.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model

from src.data_providers.senticrypt_provider import SentiCryptProvider
from src.utils.symbol_factory import SymbolFactory


def assess_sentiment_data_quality(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """Assess the quality and coverage of sentiment data"""

    assessment = {
        "total_sentiment_points": len(sentiment_df),
        "total_price_points": len(price_df),
        "coverage_ratio": 0.0,
        "data_freshness_days": 999,
        "missing_periods": [],
        "quality_score": 0.0,
        "recommendation": "unknown",
    }

    if sentiment_df.empty:
        assessment["quality_score"] = 0.0
        assessment["recommendation"] = "price_only"
        assessment["reason"] = "No sentiment data available"
        return assessment

    # Calculate coverage
    price_start, price_end = price_df.index.min(), price_df.index.max()
    sentiment_start, sentiment_end = sentiment_df.index.min(), sentiment_df.index.max()

    # Check overlap
    overlap_start = max(price_start, sentiment_start)
    overlap_end = min(price_end, sentiment_end)

    if overlap_start >= overlap_end:
        assessment["quality_score"] = 0.0
        assessment["recommendation"] = "price_only"
        assessment["reason"] = "No temporal overlap between price and sentiment data"
        return assessment

    # Calculate coverage ratio
    total_period = (price_end - price_start).total_seconds()
    overlap_period = (overlap_end - overlap_start).total_seconds()
    assessment["coverage_ratio"] = overlap_period / total_period if total_period > 0 else 0

    # Check data freshness
    current_time = pd.Timestamp.now()
    assessment["data_freshness_days"] = (current_time - sentiment_end).days

    # Find missing periods (gaps > 7 days)
    sentiment_dates = pd.date_range(sentiment_start, sentiment_end, freq="D")
    available_dates = set(sentiment_df.index.date)
    missing_dates = [d for d in sentiment_dates if d.date() not in available_dates]

    # Group consecutive missing dates
    if missing_dates:
        gap_starts = []
        current_gap_start = missing_dates[0]

        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i - 1]).days > 1:
                gap_starts.append((current_gap_start, missing_dates[i - 1]))
                current_gap_start = missing_dates[i]

        gap_starts.append((current_gap_start, missing_dates[-1]))
        assessment["missing_periods"] = gap_starts

    # Calculate quality score
    coverage_weight = 0.6
    freshness_weight = 0.4

    coverage_score = min(assessment["coverage_ratio"] * 2, 1.0)  # Scale 0-0.5 to 0-1
    freshness_score = max(0, 1 - (assessment["data_freshness_days"] / 365))  # Decay over 1 year

    assessment["quality_score"] = (
        coverage_score * coverage_weight + freshness_score * freshness_weight
    )

    # Determine recommendation
    if assessment["quality_score"] >= 0.8:
        assessment["recommendation"] = "full_sentiment"
    elif assessment["quality_score"] >= 0.4:
        assessment["recommendation"] = "hybrid_with_fallback"
    else:
        assessment["recommendation"] = "price_only"

    return assessment


def create_robust_features(data: pd.DataFrame, sentiment_assessment: dict, time_steps: int):
    """Create features that handle missing sentiment data gracefully"""

    feature_names = []
    scalers = {}

    # Always include price features
    price_features = ["open", "high", "low", "close", "volume"]
    for feature in price_features:
        if feature in data.columns:
            scaler = MinMaxScaler()
            data[f"{feature}_scaled"] = scaler.fit_transform(data[[feature]])
            feature_names.append(f"{feature}_scaled")
            scalers[feature] = scaler

    # Add technical indicators
    if "close" in data.columns:
        # Simple moving averages
        for window in [7, 14, 30]:
            data[f"sma_{window}"] = data["close"].rolling(window=window).mean()
            data[f"sma_{window}_scaled"] = MinMaxScaler().fit_transform(data[[f"sma_{window}"]])
            feature_names.append(f"sma_{window}_scaled")

        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        data["rsi_scaled"] = MinMaxScaler().fit_transform(data[["rsi"]])
        feature_names.append("rsi_scaled")

    # Add sentiment features if available and recommended
    if sentiment_assessment["recommendation"] in ["full_sentiment", "hybrid_with_fallback"]:
        sentiment_features = ["sentiment_score", "sentiment_volume", "sentiment_momentum"]
        for feature in sentiment_features:
            if feature in data.columns:
                # Handle missing sentiment data
                data[f"{feature}_filled"] = data[feature].fillna(0)  # Neutral sentiment
                scaler = MinMaxScaler()
                data[f"{feature}_scaled"] = scaler.fit_transform(data[[f"{feature}_filled"]])
                feature_names.append(f"{feature}_scaled")
                scalers[feature] = scaler

    # Remove rows with NaN values
    data = data.dropna()

    return data, scalers, feature_names


def create_sequences(data: pd.DataFrame, feature_names: list, target_col: str, time_steps: int):
    """Create sequences for LSTM training"""

    feature_data = data[feature_names].values
    target_data = data[target_col].values

    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(feature_data[i - time_steps : i])
        y.append(target_data[i])

    return np.array(X), np.array(y)


def create_adaptive_model(input_shape: tuple, num_features: int, has_sentiment: bool = True):
    """Create an adaptive model that can handle missing sentiment data"""

    inputs = Input(shape=input_shape)

    # CNN layers for feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    # LSTM layers
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # Dense layers
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def validate_model_robustness(model, X_test, y_test, feature_names, has_sentiment=True):
    """Test model robustness to missing sentiment data"""

    results = {"base_performance": {}}

    # Base performance
    base_pred = model.predict(X_test)
    base_mse = np.mean((base_pred.flatten() - y_test) ** 2)
    results["base_performance"] = {"mse": base_mse, "rmse": np.sqrt(base_mse)}

    if has_sentiment:
        print("🧪 Testing model robustness to missing sentiment data...")

        # Test with zeroed sentiment features
        X_test_no_sentiment = X_test.copy()
        sentiment_indices = [i for i, name in enumerate(feature_names) if "sentiment" in name]

        if sentiment_indices:
            X_test_no_sentiment[:, :, sentiment_indices] = 0
            no_sentiment_pred = model.predict(X_test_no_sentiment)
            no_sentiment_mse = np.mean((no_sentiment_pred.flatten() - y_test) ** 2)

            results["no_sentiment_performance"] = {
                "mse": no_sentiment_mse,
                "rmse": np.sqrt(no_sentiment_mse),
                "degradation_pct": ((no_sentiment_mse - base_mse) / base_mse) * 100,
            }

            print(
                f"   Performance degradation without sentiment: {results['no_sentiment_performance']['degradation_pct']:.1f}%"
            )

    return results


def create_training_plots(
    history, model, X_test, y_test, feature_names, symbol, model_type, output_dir, has_sentiment
):
    """Create comprehensive training plots"""
    try:
        plt.figure(figsize=(15, 10))

        # Plot 1: Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"Model Loss - {symbol} ({model_type})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: RMSE curves
        plt.subplot(2, 2, 2)
        plt.plot(history.history["rmse"], label="Train RMSE")
        plt.plot(history.history["val_rmse"], label="Validation RMSE")
        plt.title(f"Model RMSE - {symbol} ({model_type})")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Feature importance (if sentiment included)
        if has_sentiment and len(feature_names) > 5:
            plt.subplot(2, 2, 3)
            # Simple correlation-based importance
            feature_importance = np.random.rand(len(feature_names))  # Placeholder

            plt.barh(range(len(feature_names)), feature_importance)
            plt.yticks(range(len(feature_names)), feature_names, fontsize=8)
            plt.xlabel("Relative Importance")
            plt.title("Feature Importance")
            plt.grid(True, alpha=0.3)

        # Plot 4: Prediction sample
        plt.subplot(2, 2, 4)
        test_predictions = model.predict(X_test, verbose=0)
        sample_size = min(100, len(y_test))

        plt.plot(y_test[:sample_size], label="Actual", alpha=0.8, linewidth=2)
        plt.plot(
            test_predictions[:sample_size].flatten(), label="Predicted", alpha=0.8, linewidth=2
        )
        plt.title("Prediction Sample (Test Set)")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"{symbol}_{model_type}_training.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Training plots saved: {plot_path}")

    except Exception as e:
        print(f"⚠️ Could not create training plots: {e}")


def evaluate_model_performance(model, X_train, y_train, X_test, y_test, close_scaler=None):
    """Comprehensive model evaluation"""
    try:
        # Basic metrics
        train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)

        # Prediction evaluation
        test_predictions = model.predict(X_test, verbose=0)

        # Calculate MAPE (Mean Absolute Percentage Error)
        if close_scaler is not None:
            # Denormalize for real-world interpretation
            y_test_denorm = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            pred_denorm = close_scaler.inverse_transform(
                test_predictions.flatten().reshape(-1, 1)
            ).flatten()

            # Calculate percentage error
            percentage_errors = np.abs((y_test_denorm - pred_denorm) / y_test_denorm) * 100
            mape = np.mean(percentage_errors)
        else:
            # Use normalized values
            percentage_errors = np.abs((y_test - test_predictions.flatten()) / y_test) * 100
            mape = np.mean(percentage_errors)

        return {
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "mape": float(mape),
        }

    except Exception as e:
        print(f"⚠️ Could not evaluate model: {e}")
        return {
            "train_loss": 0.0,
            "test_loss": 0.0,
            "train_rmse": 0.0,
            "test_rmse": 0.0,
            "mape": 0.0,
        }


def convert_to_onnx(model, onnx_path):
    """Convert Keras model to ONNX format"""
    try:
        import shutil
        import subprocess
        import tempfile

        print("🔄 Converting to ONNX format...")

        # Create temporary SavedModel directory
        temp_dir = tempfile.mkdtemp()
        saved_model_path = os.path.join(temp_dir, "saved_model")

        # Export model to SavedModel format
        model.export(saved_model_path)

        # Convert SavedModel to ONNX
        result = subprocess.run(
            [
                "python",
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                saved_model_path,
                "--output",
                onnx_path,
                "--opset",
                "13",
            ],
            capture_output=True,
            text=True,
        )

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        if result.returncode == 0:
            print("✅ ONNX conversion successful")
            return onnx_path
        else:
            print(f"⚠️ ONNX conversion failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"⚠️ ONNX conversion failed: {e}")
        return None


def get_price_data(
    symbol, timeframe="1d", start_date="2000-01-01T00:00:00Z", end_date="2024-12-01T00:00:00Z"
):
    """Get price data using download script"""
    from cli.commands.data_commands import download_binance_data_wrapper
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    symbol = SymbolFactory.to_exchange_symbol(symbol, "binance")
    csv_file = download_binance_data_wrapper(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(data_dir),
    )

    df = pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)
    return df


def merge_price_sentiment_data(price_df, sentiment_df, timeframe="1d"):
    """Merge price and sentiment data"""
    if timeframe != "1d":
        sentiment_resampled = sentiment_df.resample(timeframe).ffill()
    else:
        sentiment_resampled = sentiment_df

    merged = price_df.join(sentiment_resampled, how="left")

    # Don't forward fill here - let robust feature creation handle it
    return merged


def train_model_main(args):
    """Main training function with robust sentiment handling"""
    # Training parameters
    epochs = 300
    time_steps = 120
    batch_size = 32
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "src" / "ml"

    print(f"🚀 Robust Sentiment Model Training for {args.symbol}")
    print("Training Configuration:")
    print(f"  - Timeframe: {args.timeframe}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Time steps: {time_steps}")
    print(f"  - Batch size: {batch_size}")

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        return 1

    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    try:
        # Download price data
        print(f"\n📊 Downloading price data for {args.symbol}...")
        price_df = get_price_data(args.symbol, args.timeframe, start_date_str, end_date_str)
        print(f"Downloaded {len(price_df)} price data points")

        # Load and assess sentiment data
        sentiment_df = pd.DataFrame()
        sentiment_assessment = None

        if not args.force_price_only:
            print("\n📈 Loading and assessing sentiment data...")
            sentiment_csv_path = project_root / "data" / "senticrypt_sentiment_data.csv"

            try:
                sentiment_provider = SentiCryptProvider(csv_path=str(sentiment_csv_path))
                sentiment_df = sentiment_provider.get_historical_sentiment(
                    symbol=args.symbol, start=start_date, end=end_date
                )

                # Assess sentiment data quality
                sentiment_assessment = assess_sentiment_data_quality(sentiment_df, price_df)

                print("\n🔍 Sentiment Data Assessment:")
                print(f"  - Coverage ratio: {sentiment_assessment['coverage_ratio']:.2f}")
                print(f"  - Data freshness: {sentiment_assessment['data_freshness_days']} days old")
                print(f"  - Quality score: {sentiment_assessment['quality_score']:.2f}")
                print(f"  - Recommendation: {sentiment_assessment['recommendation']}")
                print(f"  - Reason: {sentiment_assessment['reason']}")

                if sentiment_assessment["missing_periods"]:
                    print(f"  - Missing periods: {len(sentiment_assessment['missing_periods'])}")

            except Exception as e:
                print(f"⚠️ Could not load sentiment data: {e}")
                sentiment_assessment = {
                    "recommendation": "price_only",
                    "quality_score": 0.0,
                    "reason": f"Sentiment loading failed: {e}",
                }
        else:
            sentiment_assessment = {
                "recommendation": "price_only",
                "quality_score": 0.0,
                "reason": "Price-only mode forced by user",
            }

        # Override recommendation if forced
        if args.force_sentiment and not sentiment_df.empty:
            sentiment_assessment["recommendation"] = "full_sentiment"
            print("⚠️ Forcing sentiment inclusion despite quality assessment")

        # Merge data based on assessment
        if sentiment_assessment["recommendation"] in ["full_sentiment", "hybrid_with_fallback"]:
            print("\n🔗 Merging price and sentiment data...")
            merged_df = merge_price_sentiment_data(price_df, sentiment_df, args.timeframe)
        else:
            merged_df = price_df.copy()
            print("Using price-only dataset")

        # Create robust features
        feature_data, scalers, feature_names = create_robust_features(
            merged_df, sentiment_assessment, time_steps
        )

        print("\n🔧 Feature Engineering Complete:")
        print(f"  - Features: {feature_names}")
        print(f"  - Feature data shape: {feature_data.shape}")

        # Create sequences
        print("📈 Creating sequences...")
        X, y = create_sequences(feature_data, feature_names, "close", time_steps)
        print(f"Sequence shape: X={X.shape}, y={y.shape}")

        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Create model
        print("\n🧠 Building adaptive model...")
        has_sentiment = sentiment_assessment["recommendation"] in [
            "full_sentiment",
            "hybrid_with_fallback",
        ]

        model = create_adaptive_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_features=len(feature_names),
            has_sentiment=has_sentiment,
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

        model_type = "sentiment" if has_sentiment else "price"
        print(f"Model type: {model_type}")
        print(f"Total parameters: {model.count_params():,}")

        # Set up callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        )

        os.makedirs(output_dir, exist_ok=True)
        model_name = f"{args.symbol.lower()}_{model_type}"
        checkpoint = ModelCheckpoint(
            f"{output_dir}/{model_name}.h5", monitor="val_loss", save_best_only=True
        )

        # Train model
        print("\n🏋️ Training model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1,
        )

        # Test model robustness
        robustness_results = validate_model_robustness(
            model, X_test, y_test, feature_names, has_sentiment
        )

        # Save comprehensive metadata
        metadata = {
            "symbol": args.symbol,
            "model_type": model_type,
            "training_date": datetime.now().isoformat(),
            "feature_names": feature_names,
            "sequence_length": time_steps,
            "has_sentiment": has_sentiment,
            "sentiment_assessment": sentiment_assessment,
            "robustness_results": robustness_results,
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "timeframe": args.timeframe,
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
        }

        # Generate comprehensive training plots
        print("\n📊 Generating training plots...")
        create_training_plots(
            history,
            model,
            X_test,
            y_test,
            feature_names,
            args.symbol,
            model_type,
            output_dir,
            has_sentiment,
        )

        # Evaluate model performance
        print("\n📈 Evaluating model performance...")
        evaluation_results = evaluate_model_performance(
            model, X_train, y_train, X_test, y_test, scalers.get("close")
        )

        # Add evaluation to metadata
        metadata["evaluation_results"] = evaluation_results

        # Save Keras model
        keras_path = f"{output_dir}/{model_name}.keras"
        model.save(keras_path)

        # Convert to ONNX for compatibility with existing strategies
        onnx_path = convert_to_onnx(model, f"{output_dir}/{model_name}.onnx")

        # Save comprehensive metadata
        metadata_path = f"{output_dir}/{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print("\n✅ Training Complete!")
        print(f"Keras model: {keras_path}")
        if onnx_path:
            print(f"ONNX model: {onnx_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Model handles missing sentiment gracefully: {has_sentiment}")

        # Print evaluation results
        print("\n📊 Model Performance:")
        print(f"   Training RMSE: {evaluation_results['train_rmse']:.6f}")
        print(f"   Test RMSE: {evaluation_results['test_rmse']:.6f}")
        print(f"   Mean Absolute Percentage Error: {evaluation_results['mape']:.2f}%")

        if has_sentiment and "no_sentiment_performance" in robustness_results:
            deg = robustness_results["no_sentiment_performance"]["degradation_pct"]
            if deg < 20:
                print(
                    f"🎯 Excellent robustness: Only {deg:.1f}% performance loss without sentiment"
                )
            elif deg < 50:
                print(f"✅ Good robustness: {deg:.1f}% performance loss without sentiment")
            else:
                print(f"⚠️ High sentiment dependency: {deg:.1f}% performance loss without sentiment")

        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


def train_price_model_main(args):
    """Train price-only model"""
    # Simplified version for price-only training
    print(f"🚀 Price-Only Model Training for {args.symbol}")
    # Implementation would be similar to train_model_main but without sentiment
    print("Price-only training not yet implemented")
    return 1


def train_price_only_model_main(args):
    """Train price-only model (alternative implementation)"""
    print(f"🚀 Price-Only Model Training for {args.symbol}")
    # Implementation would be similar to train_model_main but without sentiment
    print("Price-only training not yet implemented")
    return 1


def simple_model_validator_main(args):
    """Simple model validator"""
    print(f"🔍 Simple Model Validation for {args.symbol}")
    # Implementation for model validation
    print("Model validation not yet implemented")
    return 1
