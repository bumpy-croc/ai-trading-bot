import ccxt
import pandas as pd
import argparse
import sys
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
import tf2onnx
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

# Import the download function from our new script
from download_binance_data import download_data as download_binance_data

# Import sentiment provider
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_providers.augmento_provider import AugmentoProvider

# Get project root directory (parent of scripts directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_price_data(symbol, timeframe='1d', start_date='2000-01-01T00:00:00Z', end_date='2024-12-01T00:00:00Z'):
    """
    Get price data using our download script and return as DataFrame
    """
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    
    # Download data and save to CSV
    csv_file = download_binance_data(
        symbol=symbol.replace('USDT', '/USDT'),  # Convert ETHUSDT to ETH/USDT
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        output_dir=data_dir
    )
    
    # Read the CSV and return DataFrame
    df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
    return df

def merge_price_sentiment_data(price_df, sentiment_df, timeframe='1d'):
    """
    Merge price and sentiment data, handling different timeframes
    """
    # Resample sentiment data to match price timeframe if needed
    if timeframe != '1d':
        # For intraday timeframes, we need to resample daily sentiment to match
        sentiment_resampled = sentiment_df.resample(timeframe).fillna(method='ffill')
    else:
        sentiment_resampled = sentiment_df
    
    # Merge on index (datetime)
    merged = price_df.join(sentiment_resampled, how='left')
    
    # Forward fill sentiment data for missing dates
    sentiment_cols = [col for col in merged.columns if col.startswith('sentiment_')]
    merged[sentiment_cols] = merged[sentiment_cols].fillna(method='ffill')
    
    # Fill any remaining NaN values with 0
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0)
    
    return merged

def create_features(df, time_steps=120):
    """
    Create features for ML model including price and sentiment data
    """
    # Price features
    price_features = ['close', 'volume', 'high', 'low', 'open']
    
    # Sentiment features - focus on key aggregated metrics
    key_sentiment_features = [
        'sentiment_combined_overall',
        'sentiment_combined_volume',
        'sentiment_twitter_overall',
        'sentiment_reddit_overall',
        'sentiment_twitter_volume',
        'sentiment_reddit_volume',
        'sentiment_twitter_positive_ratio',
        'sentiment_twitter_negative_ratio',
        'sentiment_reddit_positive_ratio',
        'sentiment_reddit_negative_ratio'
    ]
    
    # Include only available sentiment features
    available_sentiment_features = [f for f in key_sentiment_features if f in df.columns]
    
    # Combine all features
    all_features = price_features + available_sentiment_features
    feature_data = df[all_features].copy()
    
    # Normalize features
    scalers = {}
    normalized_data = pd.DataFrame(index=feature_data.index)
    
    # Normalize price features
    for feature in price_features:
        if feature in feature_data.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data[feature] = scaler.fit_transform(feature_data[[feature]]).flatten()
            scalers[feature] = scaler
    
    # Sentiment features handling
    for feature in available_sentiment_features:
        if feature in feature_data.columns:
            if 'overall' in feature:
                # Overall sentiment is already normalized to [-1, 1], scale to [0, 1]
                normalized_data[feature] = (feature_data[feature] + 1) / 2
                scalers[feature] = None  # Custom scaling
            elif 'ratio' in feature:
                # Ratios are already [0, 1]
                normalized_data[feature] = feature_data[feature]
                scalers[feature] = None
            else:
                # Volume features need standard scaling
                scaler = StandardScaler()
                normalized_data[feature] = scaler.fit_transform(feature_data[[feature]]).flatten()
                scalers[feature] = scaler
    
    return normalized_data, scalers, all_features

def create_sequences(data, features, target_col='close', time_steps=120):
    """
    Create sequences for LSTM training with multiple features
    """
    X, y = [], []
    
    target_data = data[target_col].values
    feature_data = data[features].values
    
    for i in range(time_steps, len(data)):
        X.append(feature_data[i-time_steps:i])
        y.append(target_data[i])
    
    return np.array(X), np.array(y)

def create_model_with_augmento_sentiment(input_shape, num_features):
    """
    Create a sophisticated model that leverages Augmento sentiment features
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    dropout1 = Dropout(0.2)(pool1)
    
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(dropout1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    dropout2 = Dropout(0.2)(pool2)
    
    # LSTM layers for sequence modeling
    lstm1 = LSTM(100, return_sequences=True)(dropout2)
    dropout3 = Dropout(0.3)(lstm1)
    
    lstm2 = LSTM(50, return_sequences=False)(dropout3)
    dropout4 = Dropout(0.3)(lstm2)
    
    # Dense layers for final prediction
    dense1 = Dense(50, activation='relu')(dropout4)
    dropout5 = Dropout(0.2)(dense1)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(dropout5)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def main():
    """Main training function with Augmento sentiment integration"""
    parser = argparse.ArgumentParser(description='Train a neural network model with Augmento sentiment data for cryptocurrency price prediction')
    parser.add_argument('symbol', help='Trading pair symbol (e.g., ETHUSDT, BTCUSDT, SOLUSDT)')
    parser.add_argument('--start-date', type=str, default='2019-04-15', 
                       help='Start date for training data (YYYY-MM-DD format, default: 2019-04-15)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='End date for training data (YYYY-MM-DD format, default: 2024-01-01)')
    parser.add_argument('--timeframe', type=str, default='1d',
                       help='Timeframe for data (default: 1d)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Augmento API key for real-time data access')
    parser.add_argument('--no-sentiment', action='store_true',
                       help='Train without sentiment data (price-only model)')
    args = parser.parse_args()
    
    # Hardcoded parameters
    epochs = 300
    time_steps = 120
    batch_size = 32
    
    # Use project root for output directory
    output_dir = os.path.join(PROJECT_ROOT, 'ml')
    
    print(f"🚀 Starting model training for {args.symbol} with Augmento sentiment analysis")
    print(f"Timeframe: {args.timeframe}")
    print(f"Training epochs: {epochs}")
    print(f"Time steps: {time_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Sentiment enabled: {not args.no_sentiment}")
    if args.api_key:
        print("🔑 Using authenticated API access for real-time sentiment data")
    else:
        print("⚠️ No API key provided - using limited historical data (30+ days old)")
    
    # Parse and format the date arguments
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("Please use YYYY-MM-DD format for dates")
        sys.exit(1)
    
    start_date_str = start_date.strftime('%Y-%m-%dT00:00:00Z')
    end_date_str = end_date.strftime('%Y-%m-%dT23:59:59Z')
    
    print(f"Date range: {start_date_str} to {end_date_str}")
    
    try:
        # Load price data
        print(f"\n📊 Downloading price data for {args.symbol}...")
        price_df = get_price_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        print(f"Downloaded {len(price_df)} price data points")
        
        # Load sentiment data if enabled
        sentiment_df = pd.DataFrame()
        if not args.no_sentiment:
            print(f"\n📈 Loading Augmento sentiment data...")
            
            # Adjust dates for API limitation if no key provided
            sentiment_start = start_date
            sentiment_end = end_date
            
            if not args.api_key:
                print("⚠️ No API key - adjusting dates for historical data limitation")
                max_date = datetime.now() - timedelta(days=35)
                if end_date > max_date:
                    sentiment_end = max_date
                    sentiment_start = max(start_date, sentiment_end - timedelta(days=min(365, (end_date - start_date).days)))
                    print(f"Adjusted sentiment date range: {sentiment_start.strftime('%Y-%m-%d')} to {sentiment_end.strftime('%Y-%m-%d')}")
            
            # Initialize sentiment provider
            sentiment_provider = AugmentoProvider(api_key=args.api_key)
            
            # Get multi-source sentiment data
            sentiment_df = sentiment_provider.get_multi_source_sentiment(
                symbol=args.symbol,
                start=sentiment_start,
                end=sentiment_end,
                sources=['twitter', 'reddit']  # Most reliable sources
            )
            
            if not sentiment_df.empty:
                print(f"Loaded {len(sentiment_df)} sentiment data points with {len(sentiment_df.columns)} features")
                
                # Show sample of sentiment data
                print("📈 Sample sentiment metrics:")
                key_cols = ['sentiment_combined_overall', 'sentiment_combined_volume', 
                           'sentiment_twitter_overall', 'sentiment_reddit_overall']
                available_cols = [col for col in key_cols if col in sentiment_df.columns]
                if available_cols:
                    recent_data = sentiment_df[available_cols].tail(3).round(3)
                    print(recent_data)
                
            else:
                print("⚠️ No sentiment data available, falling back to price-only model")
                args.no_sentiment = True
        
        # Merge price and sentiment data
        if not args.no_sentiment and not sentiment_df.empty:
            print(f"\n🔗 Merging price and sentiment data...")
            merged_df = merge_price_sentiment_data(price_df, sentiment_df, args.timeframe)
            print(f"Merged dataset has {len(merged_df)} records with {len(merged_df.columns)} features")
        else:
            merged_df = price_df.copy()
            print(f"Using price-only dataset with {len(merged_df)} records")
        
        # Create features
        print("🔧 Creating features...")
        feature_data, scalers, feature_names = create_features(merged_df, time_steps)
        
        print(f"Features: {feature_names}")
        print(f"Feature data shape: {feature_data.shape}")
        
        # Create sequences
        print("📈 Creating sequences...")
        X, y = create_sequences(feature_data, feature_names, 'close', time_steps)
        
        print(f"Sequence shape: X={X.shape}, y={y.shape}")
        
        # Split the data (80% for training)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create and train the model
        print("\n🧠 Building and training model...")
        
        # Choose model architecture based on whether sentiment is included
        if not args.no_sentiment and len(feature_names) > 5:
            model = create_model_with_augmento_sentiment(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_features=len(feature_names)
            )
            model_suffix = "augmento_sentiment"
        else:
            # Fallback to simpler model for price-only
            model = Sequential()
            model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', 
                           input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(100, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(100, return_sequences=False))
            model.add(Dropout(0.3))
            model.add(Dense(units=1, activation='sigmoid'))
            model_suffix = "price_only"
        
        model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        print(f"Model architecture: {model_suffix}")
        print(f"Total parameters: {model.count_params():,}")
        
        # Set up callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
        )
        
        # Checkpoint to save the best model
        os.makedirs(output_dir, exist_ok=True)
        model_name = f'best_model_{args.symbol.lower()}_{model_suffix}'
        checkpoint = ModelCheckpoint(
            f'{output_dir}/{model_name}.h5', 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=False
        )
        
        # Model training
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            validation_data=(X_test, y_test), 
            batch_size=batch_size, 
            callbacks=[early_stopping, checkpoint], 
            verbose=2
        )
        
        # Plot the training history
        print("\n📊 Generating training plots...")
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Loss
        plt.subplot(2, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {args.symbol} ({model_suffix})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot 2: RMSE
        plt.subplot(2, 3, 2)
        plt.plot(history.history['rmse'], label='Train RMSE')
        plt.plot(history.history['val_rmse'], label='Validation RMSE')
        plt.title(f'Model RMSE - {args.symbol} ({model_suffix})')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        
        # Plot 3: Feature importance (if sentiment included)
        if not args.no_sentiment and len(feature_names) > 5:
            plt.subplot(2, 3, 3)
            # Simple feature correlation with target
            correlations = []
            for feature in feature_names:
                if feature in feature_data.columns:
                    corr = feature_data[feature].corr(feature_data['close'])
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            
            plt.barh(range(len(feature_names)), correlations)
            plt.yticks(range(len(feature_names)), feature_names, fontsize=8)
            plt.xlabel('Absolute Correlation with Price')
            plt.title('Feature Importance (Correlation)')
        
        # Plot 4: Prediction sample
        plt.subplot(2, 3, 4)
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        sample_size = min(100, len(y_test))
        plt.plot(y_test[:sample_size], label='Actual', alpha=0.7)
        plt.plot(test_predictions[:sample_size], label='Predicted', alpha=0.7)
        plt.title('Prediction Sample (Test Set)')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Price')
        plt.legend()
        
        # Plot 5: Sentiment overview (if available)
        if not args.no_sentiment and not sentiment_df.empty:
            plt.subplot(2, 3, 5)
            if 'sentiment_combined_overall' in sentiment_df.columns:
                sentiment_df['sentiment_combined_overall'].plot(alpha=0.7)
                plt.title('Combined Sentiment Over Time')
                plt.xlabel('Date')
                plt.ylabel('Sentiment Score')
        
        # Plot 6: Sentiment vs Price correlation (if available)
        if not args.no_sentiment and not sentiment_df.empty:
            plt.subplot(2, 3, 6)
            if 'sentiment_combined_overall' in sentiment_df.columns:
                # Align sentiment with price data for correlation
                aligned_data = merged_df[['close', 'sentiment_combined_overall']].dropna()
                if len(aligned_data) > 10:
                    correlation = aligned_data['close'].corr(aligned_data['sentiment_combined_overall'])
                    plt.scatter(aligned_data['sentiment_combined_overall'], aligned_data['close'], alpha=0.5)
                    plt.xlabel('Sentiment Score')
                    plt.ylabel('Price')
                    plt.title(f'Sentiment vs Price (r={correlation:.3f})')
        
        plt.tight_layout()
        plot_path = os.path.join(PROJECT_ROOT, 'ml', f'{args.symbol}_{model_suffix}_{args.timeframe}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved as '{plot_path}'")
        
        # Save model in multiple formats
        keras_path = f'{output_dir}/model_{args.symbol.lower()}_{model_suffix}.keras'
        model.save(keras_path)
        print(f"Keras model saved as '{keras_path}'")
        
        # Convert to ONNX
        print("\n🔄 Converting model to ONNX format...")
        try:
            import tempfile
            import subprocess
            import shutil
            
            # Create a temporary SavedModel directory
            temp_dir = tempfile.mkdtemp()
            saved_model_path = os.path.join(temp_dir, 'saved_model')
            model.export(saved_model_path)
            
            # Convert SavedModel to ONNX
            onnx_path = f'{output_dir}/model_{args.symbol.lower()}_{model_suffix}.onnx'
            result = subprocess.run([
                'python', '-m', 'tf2onnx.convert',
                '--saved-model', saved_model_path,
                '--output', onnx_path,
                '--opset', '13'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"ONNX model saved as '{onnx_path}'")
            else:
                print(f"ONNX conversion failed: {result.stderr}")
            
            # Clean up
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
        
        # Evaluate the model
        print("\n📈 Evaluating model performance...")
        train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training   - Loss: {train_loss:.6f}, RMSE: {train_rmse:.6f}")
        print(f"Testing    - Loss: {test_loss:.6f}, RMSE: {test_rmse:.6f}")
        
        # Calculate additional metrics
        test_predictions = model.predict(X_test, verbose=0)
        
        # Denormalize for better interpretation
        close_scaler = scalers['close']
        y_test_denorm = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        pred_denorm = close_scaler.inverse_transform(test_predictions.flatten().reshape(-1, 1)).flatten()
        
        # Calculate percentage error
        percentage_errors = np.abs((y_test_denorm - pred_denorm) / y_test_denorm) * 100
        mean_percentage_error = np.mean(percentage_errors)
        
        print(f"Mean Absolute Percentage Error: {mean_percentage_error:.2f}%")
        
        # Save metadata
        metadata = {
            'symbol': args.symbol,
            'model_type': model_suffix,
            'timeframe': args.timeframe,
            'features': feature_names,
            'training_period': f"{args.start_date} to {args.end_date}",
            'total_samples': len(X),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'train_loss': float(train_loss),
            'test_loss': float(test_loss),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'mean_percentage_error': float(mean_percentage_error),
            'sentiment_enabled': not args.no_sentiment,
            'api_key_used': args.api_key is not None,
            'sentiment_provider': 'Augmento' if not args.no_sentiment else None
        }
        
        import json
        metadata_path = f'{output_dir}/model_{args.symbol.lower()}_{model_suffix}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Model training completed successfully for {args.symbol}!")
        print(f"📁 Models saved in: {output_dir}")
        print(f"📊 Training plot: {plot_path}")
        print(f"📋 Metadata: {metadata_path}")
        
        if not args.no_sentiment:
            sentiment_features = [f for f in feature_names if f.startswith('sentiment_')]
            print(f"🎯 Augmento sentiment features included: {len(sentiment_features)}")
            print(f"   Key features: {sentiment_features[:5]}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 