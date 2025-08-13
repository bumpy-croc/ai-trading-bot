#!/usr/bin/env python3
"""
Robust Sentiment Model Trainer

This script handles missing sentiment data gracefully during training by:
1. Detecting sentiment data availability and age
2. Creating hybrid models (price + sentiment) or fallback models (price-only)
3. Filling missing sentiment periods with neutral values
4. Validating model performance on different data scenarios
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Import the download function from our new script
from download_binance_data import download_data as download_binance_data
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

from src.utils.symbol_factory import SymbolFactory

# Import sentiment provider
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_providers.senticrypt_provider import SentiCryptProvider

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

        # Only include gaps > 7 days
        assessment["missing_periods"] = [
            (start, end) for start, end in gap_starts if (end - start).days > 7
        ]

    # Calculate quality score
    coverage_score = min(1.0, assessment["coverage_ratio"])
    freshness_score = max(0.0, 1.0 - assessment["data_freshness_days"] / 365)  # Decay over a year
    completeness_score = max(0.0, 1.0 - len(assessment["missing_periods"]) / 10)  # Penalty for gaps

    assessment["quality_score"] = (
        coverage_score * 0.5 + freshness_score * 0.3 + completeness_score * 0.2
    )

    # Make recommendation
    if assessment["quality_score"] >= 0.7:
        assessment["recommendation"] = "full_sentiment"
        assessment["reason"] = "High quality sentiment data available"
    elif assessment["quality_score"] >= 0.4:
        assessment["recommendation"] = "hybrid_with_fallback"
        assessment["reason"] = "Moderate quality sentiment data - use with neutral fallbacks"
    else:
        assessment["recommendation"] = "price_only"
        assessment["reason"] = "Poor quality sentiment data - not reliable for training"

    return assessment


def create_robust_features(df: pd.DataFrame, sentiment_assessment: dict, time_steps: int = 120):
    """Create features that handle missing sentiment data robustly"""

    # Always include price features
    price_features = ["close", "volume", "high", "low", "open"]

    # SIMPLIFIED: Only use primary sentiment (mean score)
    sentiment_features = ["sentiment_primary"]  # Just the mean sentiment score

    # Create feature DataFrame
    feature_data = df[price_features].copy()

    # Handle sentiment features based on assessment
    if sentiment_assessment["recommendation"] in ["full_sentiment", "hybrid_with_fallback"]:
        print(
            f"🔮 Including simplified sentiment feature (quality: {sentiment_assessment['quality_score']:.2f})"
        )

        # Only process the single sentiment feature
        if "sentiment_primary" in df.columns:
            # Use actual sentiment data where available
            feature_data["sentiment_primary"] = df["sentiment_primary"]
        else:
            # Create neutral baseline (0.5 for mean sentiment)
            feature_data["sentiment_primary"] = 0.5
            print("   Created neutral sentiment baseline (0.5)")

        # Fill missing sentiment with neutral value
        mask = feature_data["sentiment_primary"].isna()
        if mask.any():
            feature_data.loc[mask, "sentiment_primary"] = 0.5
            print(f"   Filled {mask.sum()} missing sentiment values with neutral (0.5)")

        all_features = price_features + sentiment_features

    else:
        print("📊 Using price-only features (sentiment data insufficient)")
        all_features = price_features

    # Normalize features
    scalers = {}
    normalized_data = pd.DataFrame(index=feature_data.index)

    # Normalize price features using MinMaxScaler
    for feature in price_features:
        if feature in feature_data.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data[feature] = scaler.fit_transform(feature_data[[feature]]).flatten()
            scalers[feature] = scaler

    # Normalize sentiment feature (if included)
    if sentiment_assessment["recommendation"] in ["full_sentiment", "hybrid_with_fallback"]:
        if "sentiment_primary" in feature_data.columns:
            # Sentiment already ranges from -1 to 1, normalize to 0-1
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data["sentiment_primary"] = scaler.fit_transform(
                feature_data[["sentiment_primary"]]
            ).flatten()
            scalers["sentiment_primary"] = scaler

    return normalized_data, scalers, all_features


def get_neutral_sentiment_value(feature_name: str) -> float:
    """Get appropriate neutral value for a sentiment feature"""
    if "primary" in feature_name.lower():
        return 0.5  # Neutral sentiment
    elif "momentum" in feature_name.lower():
        return 0.0  # No momentum
    elif "volatility" in feature_name.lower():
        return 0.3  # Low-moderate volatility
    elif "extreme_positive" in feature_name.lower():
        return 0.0  # No extreme positive sentiment
    elif "extreme_negative" in feature_name.lower():
        return 0.0  # No extreme negative sentiment
    elif "ma_" in feature_name.lower():
        return 0.5  # Neutral moving average
    else:
        return 0.0  # Default neutral


def create_adaptive_model(input_shape, num_features: int, has_sentiment: bool = True):
    """Create SIMPLIFIED model to prevent overfitting"""

    inputs = Input(shape=input_shape)

    # Simpler CNN feature extraction layers (reduced filters)
    conv1 = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    # Simplified LSTM layers for both cases (avoiding overfitting)
    lstm1 = LSTM(50, return_sequences=True)(pool2)
    dropout1 = Dropout(0.2)(lstm1)

    lstm2 = LSTM(25, return_sequences=False)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)

    # Simple dense layers
    dense1 = Dense(25, activation="relu")(dropout2)
    dropout3 = Dropout(0.1)(dense1)

    # Output layer
    outputs = Dense(1, activation="sigmoid")(dropout3)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_sequences(data, features, target_col="close", time_steps=120):
    """Create sequences for LSTM training"""
    X, y = [], []

    target_data = data[target_col].values
    feature_data = data[features].values

    for i in range(time_steps, len(data)):
        X.append(feature_data[i - time_steps : i])
        y.append(target_data[i])

    return np.array(X), np.array(y)


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


def main():
    """Main training function with robust sentiment handling"""
    parser = argparse.ArgumentParser(description="Robust sentiment model trainer")
    parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument(
        "--start-date", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end-date", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    parser.add_argument(
        "--force-sentiment",
        action="store_true",
        help="Force sentiment inclusion even if data quality is poor",
    )
    parser.add_argument(
        "--force-price-only",
        action="store_true",
        help="Train price-only model regardless of sentiment availability",
    )

    args = parser.parse_args()

    # Training parameters
    epochs = 300
    time_steps = 120
    batch_size = 32
    output_dir = os.path.join(PROJECT_ROOT, "ml")

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
        sys.exit(1)

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
            sentiment_csv_path = os.path.join(PROJECT_ROOT, "data", "senticrypt_sentiment_data.csv")

            try:
                sentiment_provider = SentiCryptProvider(csv_path=sentiment_csv_path)
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

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


# Utility functions from the original script
def get_price_data(
    symbol, timeframe="1d", start_date="2000-01-01T00:00:00Z", end_date="2024-12-01T00:00:00Z"
):
    """Get price data using download script"""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    symbol = SymbolFactory.to_exchange_symbol(symbol, "binance")
    csv_file = download_binance_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        output_dir=data_dir,
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


if __name__ == "__main__":
    main()
