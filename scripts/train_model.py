import ccxt
import pandas as pd
import argparse
import sys
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
import tf2onnx
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

# Import the download function from our new script
from download_binance_data import download_data as download_binance_data

# Get project root directory (parent of scripts directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_close_data(symbol, timeframe='1d', start_date='2000-01-01T00:00:00Z', end_date='2024-12-01T00:00:00Z'):
    """
    Get close price data using our download script and return as numpy array
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
    
    # Read the CSV and return close values
    df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
    return df['close'].values

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train a neural network model for cryptocurrency price prediction')
    parser.add_argument('symbol', help='Trading pair symbol (e.g., ETHUSDT, BTCUSDT, SOLUSDT)')
    args = parser.parse_args()
    
    # Hardcoded parameters
    timeframe = '1d'
    epochs = 300
    time_steps = 120
    batch_size = 32
    
    # Use project root for output directory
    output_dir = os.path.join(PROJECT_ROOT, 'ml')
    
    print(f"🚀 Starting model training for {args.symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Training epochs: {epochs}")
    print(f"Time steps: {time_steps}")
    print(f"Batch size: {batch_size}")
    
    # Determine date range first
    end_date = datetime(2025, 4, 30)
    start_date = datetime(2020, 1, 1)
    start_date_str = start_date.strftime('%Y-%m-%dT00:00:00Z')
    end_date_str = end_date.strftime('%Y-%m-%dT23:59:59Z')
    
    print(f"Date range: {start_date_str} to {end_date_str}")
    
    try:
        # Load data
        print(f"\n📊 Downloading data for {args.symbol}...")
        data = get_close_data(
            symbol=args.symbol,
            timeframe=timeframe,
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        print(f"Downloaded {len(data)} data points")
        
        # Normalize the data
        print("🔧 Normalizing data...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.reshape(-1, 1))
        
        # Function to create samples from the sequence
        def create_samples(dataset, time_steps=time_steps):
            X, y = [], []
            for i in range(time_steps, len(dataset)):
                X.append(dataset[i-time_steps:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)
        
        # Prepare training and test data
        print("📈 Preparing training data...")
        X, y = create_samples(data, time_steps)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM
        
        # Split the data (80% for training)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train the model
        print("\n🧠 Building and training model...")
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Set up early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )
        
        # Checkpoint to save the best model
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = ModelCheckpoint(
            f'{output_dir}/best_model_{args.symbol.lower()}.h5', 
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
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['rmse'], label='Train RMSE')
        plt.plot(history.history['val_rmse'], label='Validation RMSE')
        plt.title(f'Model Training History - {args.symbol}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/RMSE')
        plt.legend()
        plot_path = os.path.join(PROJECT_ROOT, f'{args.symbol}_{timeframe}.png')
        plt.savefig(plot_path)
        print(f"Training plot saved as '{plot_path}'")
        
        # Convert the model to ONNX
        print("\n🔄 Converting model to ONNX format...")
        try:
            # Save model first in .keras format to avoid warnings
            model.save(f'{output_dir}/model_{args.symbol.lower()}.keras')
            
            # Use the correct tf2onnx API
            import tempfile
            
            # Create a temporary SavedModel directory
            temp_dir = tempfile.mkdtemp()
            saved_model_path = os.path.join(temp_dir, 'saved_model')
            model.export(saved_model_path)
            
            # Convert SavedModel to ONNX using the correct API
            import subprocess
            result = subprocess.run([
                'python', '-m', 'tf2onnx.convert',
                '--saved-model', saved_model_path,
                '--output', f'{output_dir}/model_{args.symbol.lower()}.onnx',
                '--opset', '13'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"ONNX model saved as 'model_{args.symbol.lower()}.onnx'")
            else:
                print(f"ONNX conversion failed: {result.stderr}")
                raise Exception("ONNX conversion subprocess failed")
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
            print("Model training completed successfully, but ONNX conversion was skipped.")
            # Save in Keras format as backup
            model.save(f'{output_dir}/model_{args.symbol.lower()}.keras')
            print(f"Model saved in Keras format as 'model_{args.symbol.lower()}.keras'")
        
        # Evaluate the model
        print("\n📈 Evaluating model performance...")
        train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
        print(f"Training - Loss: {train_loss:.6f}, RMSE: {train_rmse:.6f}")
        print(f"Testing  - Loss: {test_loss:.6f}, RMSE: {test_rmse:.6f}")
        
        print(f"\n✅ Model training completed successfully for {args.symbol}!")
        print(f"📁 Models saved in: {output_dir}")
        print(f"📊 Training plot: {plot_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()