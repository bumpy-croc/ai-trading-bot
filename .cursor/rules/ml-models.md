---
description: Trading Bot ML Model Training & Integration Guide
globs: 
alwaysApply: false
---

# 🧠 Trading Bot ML Models

## Model Overview

Two main model types for price prediction:
1. **Price Prediction Models** (`btcusdt_price.*`) - 5 features (OHLCV)
2. **Sentiment-Enhanced Models** (`btcusdt_sentiment.*`) - 13 features (5 price + 8 sentiment)

---

## Model Architecture

### Neural Network Structure
```python
# CNN + LSTM + Dense architecture
model = Sequential([
    # CNN layers for feature extraction
    Conv1D(64, 3, activation='relu', input_shape=(120, features)),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    
    # LSTM layers for temporal patterns
    LSTM(100, return_sequences=True),
    LSTM(100),
    
    # Dense layers for prediction
    Dense(50, activation='relu'),
    Dense(1, activation='linear')
])
```

### Input Features
```python
# Price-only model (5 features)
price_features = ['open', 'high', 'low', 'close', 'volume']

# Sentiment-enhanced model (13 features)
all_features = price_features + [
    'sentiment_score', 'sentiment_volume', 'sentiment_momentum',
    'sentiment_freshness', 'sentiment_consensus', 'sentiment_extremes',
    'sentiment_trend', 'sentiment_volatility'
]
```

---

## Model Training

### Training Script
```bash
# Basic training
python scripts/train_model.py BTCUSDT

# With sentiment data
python scripts/train_model.py BTCUSDT --force-sentiment

# Custom parameters
python scripts/train_model.py BTCUSDT --epochs 100 --batch-size 32 --validation-split 0.2
```

### Training Configuration
```python
# Training parameters
config = {
    'sequence_length': 120,      # 120 time steps
    'features': 5,              # OHLCV for price model
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
    'loss': 'mse',
    'optimizer': 'adam'
}
```

### Data Preprocessing
```python
def prepare_data(df: pd.DataFrame, sequence_length: int = 120) -> tuple:
    """Prepare data for model training"""
    # Normalize features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 3])  # Predict close price
    
    return np.array(X), np.array(y), scaler
```

---

## Model Validation

### Validation Scripts
```bash
# Validate model performance
python scripts/simple_model_validator.py

# Model performance analysis
python scripts/analyze_model_performance.py

# Compare model versions
python scripts/compare_models.py --model1 v1 --model2 v2
```

### Validation Metrics
```python
# Key metrics to monitor
metrics = {
    'mse': mean_squared_error(y_true, y_pred),
    'mae': mean_absolute_error(y_true, y_pred),
    'r2': r2_score(y_true, y_pred),
    'directional_accuracy': calculate_directional_accuracy(y_true, y_pred)
}
```

---

## Live Trading Integration

### ONNX Export
```python
# Export to ONNX for fast inference
import onnx
import tf2onnx

# Convert TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, "src/ml/btcusdt_price.onnx")
```

### Real-time Inference
```python
def predict_price(model, data: np.ndarray) -> tuple:
    """Make real-time price prediction"""
    # Preprocess input data
    scaled_data = scaler.transform(data)
    sequence = scaled_data[-120:].reshape(1, 120, -1)
    
    # Make prediction
    prediction = model.predict(sequence)[0]
    
    # Calculate confidence (based on model uncertainty)
    confidence = calculate_prediction_confidence(prediction, model)
    
    return prediction, confidence
```

### Graceful Fallback
```python
def get_ml_prediction(df: pd.DataFrame, index: int) -> tuple:
    """Get ML prediction with fallback"""
    try:
        # Try sentiment-enhanced model first
        if sentiment_data_available(df, index):
            prediction, confidence = predict_with_sentiment(df, index)
        else:
            # Fallback to price-only model
            prediction, confidence = predict_price_only(df, index)
            
        return prediction, confidence
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return None, 0.0
```

---

## Model Management

### Model Files
```
src/ml/
├── btcusdt_price.h5              # TensorFlow model
├── btcusdt_price.onnx            # ONNX model (for inference)
├── btcusdt_price_metadata.json   # Training metadata
├── btcusdt_price_training.png    # Training visualization
├── btcusdt_sentiment.h5          # Sentiment-enhanced model
├── btcusdt_sentiment.onnx        # ONNX model
├── btcusdt_sentiment_metadata.json
└── btcusdt_sentiment_training.png
```

### Model Metadata
```json
{
    "model_type": "price_prediction",
    "version": "1.0",
    "training_date": "2024-01-15",
    "features": ["open", "high", "low", "close", "volume"],
    "sequence_length": 120,
    "performance": {
        "mse": 0.000123,
        "mae": 0.0089,
        "r2": 0.85,
        "directional_accuracy": 0.72
    },
    "training_params": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}
```

---

## Model Retraining

### Retraining Triggers
- Monthly scheduled retraining
- Performance degradation (accuracy < 0.6)
- Market regime changes
- New data availability

### Retraining Process
```bash
# Safe retraining (staging environment)
python scripts/safe_model_trainer.py

# Validate new model
python scripts/simple_model_validator.py

# Compare with current model
python scripts/compare_models.py --model1 current --model2 new

# Deploy if better
python scripts/deploy_model.py --model new
```

---

## Performance Monitoring

### Model Drift Detection
```python
def detect_model_drift(predictions: list, actuals: list) -> bool:
    """Detect if model performance is degrading"""
    recent_accuracy = calculate_accuracy(predictions[-100:], actuals[-100:])
    historical_accuracy = calculate_accuracy(predictions[:-100], actuals[:-100])
    
    # Flag if recent accuracy drops significantly
    return recent_accuracy < historical_accuracy * 0.9
```

### Confidence Calibration
```python
def calibrate_confidence(predictions: list, actuals: list) -> float:
    """Calibrate prediction confidence"""
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    mean_error = np.mean(errors)
    
    # Higher error = lower confidence
    return max(0.1, 1.0 - mean_error / 0.01)
```

---

## Best Practices

### 1. Data Quality
- Use high-quality, clean data
- Handle missing values appropriately
- Validate data consistency

### 2. Model Validation
- Use out-of-sample testing
- Validate across different market conditions
- Monitor for overfitting

### 3. Production Deployment
- Use ONNX for fast inference
- Implement graceful fallbacks
- Monitor model performance continuously

### 4. Risk Management
- Don't rely solely on ML predictions
- Combine with traditional indicators
- Implement position size limits

---

**For detailed implementation guides, use:**
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["commands"])` - Complete command reference