---
description: Trading Bot ML Models & Training Guide
globs: 
alwaysApply: false
---

# 🧠 Trading Bot ML Models & Training

## 🎯 ML System Overview

The trading bot uses sophisticated neural networks for price prediction, combining technical indicators with sentiment analysis for enhanced accuracy.

---

## 🏗️ Model Architecture

### **Model Types**

#### **1. Price Prediction Models** (`btcusdt_price.*`)
**Purpose**: Predict future price movements using historical price data

**Architecture**:
- **Input**: 120 time steps × 5 features (OHLCV)
- **Layers**: CNN + LSTM + Dense
- **Output**: Single price prediction
- **Training**: Supervised learning with MSE loss

**Features**:
```python
price_features = [
    'close_normalized',    # MinMax normalized closing price
    'volume_normalized',   # MinMax normalized volume
    'high_normalized',     # MinMax normalized high price
    'low_normalized',      # MinMax normalized low price
    'open_normalized'      # MinMax normalized open price
]
```

#### **2. Sentiment-Enhanced Models** (`btcusdt_sentiment.*`)
**Purpose**: Predict future price movements using price + sentiment data

**Architecture**:
- **Input**: 120 time steps × 13 features (5 price + 8 sentiment)
- **Layers**: CNN + LSTM + Dense
- **Output**: Price prediction with confidence score
- **Training**: Supervised learning with custom loss function

**Features**:
```python
# Price Features (MinMax normalization)
price_features = ['close', 'volume', 'high', 'low', 'open']

# Sentiment Features (StandardScaler)
sentiment_features = [
    'sentiment_score',           # Primary sentiment score
    'sentiment_momentum',        # Sentiment change rate
    'sentiment_volatility',      # Sentiment volatility
    'extreme_positive',          # Extreme positive sentiment flag
    'extreme_negative',          # Extreme negative sentiment flag
    'sentiment_ma_3',           # 3-day sentiment moving average
    'sentiment_ma_7',           # 7-day sentiment moving average
    'sentiment_ma_14'           # 14-day sentiment moving average
]
```

---

## 🔧 Model Training Workflow

### **Data Preparation**
```python
def prepare_training_data(symbol: str, start_date: datetime, end_date: datetime):
    """Prepare training data with feature engineering"""
    
    # 1. Fetch historical price data
    price_data = binance_provider.get_historical_data(symbol, '1h', start_date, end_date)
    
    # 2. Fetch sentiment data
    sentiment_data = senticrypt_provider.get_historical_sentiment(symbol, start_date, end_date)
    
    # 3. Merge datasets
    combined_data = price_data.join(sentiment_data, how='left')
    
    # 4. Feature engineering
    combined_data = engineer_features(combined_data)
    
    # 5. Create sequences
    X, y = create_sequences(combined_data, sequence_length=120)
    
    return X, y
```

### **Feature Engineering**
```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for ML models"""
    
    # Price normalization (MinMax)
    price_features = ['close', 'volume', 'high', 'low', 'open']
    for feature in price_features:
        df[f'{feature}_normalized'] = df[feature].rolling(
            window=120, min_periods=1
        ).apply(
            lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.5,
            raw=True
        )
    
    # Sentiment features (if available)
    if 'sentiment_score' in df.columns:
        # Sentiment momentum
        df['sentiment_momentum'] = df['sentiment_score'].diff(3)
        
        # Sentiment volatility
        df['sentiment_volatility'] = df['sentiment_score'].rolling(7).std()
        
        # Extreme sentiment flags
        df['extreme_positive'] = (df['sentiment_score'] > 0.7).astype(float)
        df['extreme_negative'] = (df['sentiment_score'] < -0.7).astype(float)
        
        # Sentiment moving averages
        for period in [3, 7, 14]:
            df[f'sentiment_ma_{period}'] = df['sentiment_score'].rolling(period).mean()
    
    return df
```

### **Model Architecture Definition**
```python
def create_model(input_shape: tuple, use_sentiment: bool = False):
    """Create neural network model"""
    
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        
        # LSTM layers for sequence modeling
        LSTM(100, return_sequences=True),
        LSTM(50, return_sequences=False),
        
        # Dense layers for final prediction
        Dense(25, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Price prediction
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

---

## 🚀 Training Commands

### **Basic Model Training**
```bash
# Train price prediction model
python scripts/train_model.py BTCUSDT

# Train sentiment-enhanced model
python scripts/train_model.py BTCUSDT --force-sentiment

# Train with custom parameters
python scripts/train_model.py BTCUSDT --epochs 100 --batch-size 32
```

### **Safe Training (Staging Environment)**
```bash
# Safe training with validation
python scripts/safe_model_trainer.py

# Validate models before deployment
python scripts/simple_model_validator.py
```

### **Model Validation**
```bash
# Validate model performance
python scripts/simple_model_validator.py

# Check model files
ls -la ml/btcusdt_*

# View model metadata
cat ml/btcusdt_sentiment_metadata.json
```

---

## 📊 Model Performance Metrics

### **Training Metrics**
- **MSE (Mean Squared Error)**: Primary loss function
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Model fit quality
- **Validation Loss**: Out-of-sample performance

### **Trading Performance**
- **Prediction Accuracy**: Percentage of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted returns using model predictions
- **Win Rate**: Percentage of profitable trades using model signals
- **Maximum Drawdown**: Largest peak-to-trough decline

### **Model Validation**
```python
def validate_model_performance(model_path: str, test_data: pd.DataFrame):
    """Validate model performance on test data"""
    
    # Load model
    model = load_model(model_path)
    
    # Generate predictions
    predictions = model.predict(test_data)
    
    # Calculate metrics
    mse = mean_squared_error(test_data['actual'], predictions)
    mae = mean_absolute_error(test_data['actual'], predictions)
    r2 = r2_score(test_data['actual'], predictions)
    
    # Direction accuracy
    direction_accuracy = np.mean(
        np.sign(np.diff(test_data['actual'])) == np.sign(np.diff(predictions))
    )
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }
```

---

## 🔄 Live Trading Integration

### **Real-Time Inference**
```python
def generate_prediction(self, df: pd.DataFrame, index: int):
    """Generate real-time prediction using ONNX model"""
    
    # Prepare input sequence
    sequence = self._prepare_sequence(df, index)
    
    # Run ONNX inference
    prediction = self.ort_session.run(
        None, {self.input_name: sequence}
    )[0][0]
    
    # Calculate confidence
    confidence = self._calculate_confidence(prediction, df, index)
    
    return prediction, confidence
```

### **Sequence Preparation**
```python
def _prepare_sequence(self, df: pd.DataFrame, index: int) -> np.ndarray:
    """Prepare input sequence for model inference"""
    
    # Get sequence window
    start_idx = max(0, index - self.sequence_length + 1)
    sequence_data = df.iloc[start_idx:index + 1]
    
    # Select features
    if self.use_sentiment:
        feature_columns = self.price_features + self.sentiment_features
    else:
        feature_columns = self.price_features
    
    # Extract features
    sequence = sequence_data[feature_columns].values
    
    # Pad if necessary
    if len(sequence) < self.sequence_length:
        padding = np.zeros((self.sequence_length - len(sequence), len(feature_columns)))
        sequence = np.vstack([padding, sequence])
    
    # Reshape for model input
    sequence = sequence.reshape(1, self.sequence_length, len(feature_columns))
    
    return sequence
```

### **Confidence Calculation**
```python
def _calculate_confidence(self, prediction: float, df: pd.DataFrame, index: int) -> float:
    """Calculate prediction confidence score"""
    
    # Base confidence on recent prediction accuracy
    recent_accuracy = self._get_recent_accuracy(df, index)
    
    # Adjust for market volatility
    volatility = df.iloc[index]['atr'] / df.iloc[index]['close']
    volatility_factor = 1.0 / (1.0 + volatility * 10)  # Lower confidence in high volatility
    
    # Adjust for sentiment quality (if using sentiment)
    if self.use_sentiment and 'sentiment_freshness' in df.columns:
        sentiment_quality = df.iloc[index]['sentiment_freshness']
    else:
        sentiment_quality = 1.0
    
    # Combine factors
    confidence = recent_accuracy * volatility_factor * sentiment_quality
    
    return np.clip(confidence, 0.0, 1.0)
```

---

## 🛠️ Model Management

### **Model Files Structure**
```
ml/
├── btcusdt_price.h5              # Keras model (training)
├── btcusdt_price.keras           # Keras model (deployment)
├── btcusdt_price.onnx            # ONNX model (inference)
├── btcusdt_price_metadata.json   # Training metadata
├── BTCUSDT_price_training.png    # Training visualization
├── btcusdt_sentiment.h5          # Sentiment model (training)
├── btcusdt_sentiment.keras       # Sentiment model (deployment)
├── btcusdt_sentiment.onnx        # Sentiment model (inference)
├── btcusdt_sentiment_metadata.json # Training metadata
└── BTCUSDT_sentiment_training.png  # Training visualization
```

### **Model Metadata**
```json
{
    "symbol": "BTCUSDT",
    "model_type": "sentiment_enhanced",
    "training_date": "2024-01-15T10:30:00Z",
    "sequence_length": 120,
    "feature_names": ["close_normalized", "volume_normalized", ...],
    "training_metrics": {
        "mse": 0.001234,
        "mae": 0.0289,
        "r2": 0.856,
        "direction_accuracy": 0.723
    },
    "hyperparameters": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}
```

---

## 🔧 Model Optimization

### **Hyperparameter Tuning**
```python
def optimize_hyperparameters(data: pd.DataFrame):
    """Optimize model hyperparameters using grid search"""
    
    param_grid = {
        'lstm_units': [50, 100, 150],
        'dense_units': [25, 50, 100],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001]
    }
    
    best_params = None
    best_score = -999
    
    for params in generate_param_combinations(param_grid):
        model = create_model_with_params(params)
        score = cross_validate_model(model, data)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params
```

### **Feature Selection**
```python
def select_features(df: pd.DataFrame, target: str = 'close'):
    """Select most important features for model training"""
    
    # Calculate feature importance
    feature_importance = calculate_feature_importance(df, target)
    
    # Select top features
    top_features = feature_importance.head(10).index.tolist()
    
    return top_features
```

---

## 🚨 Model Safety & Validation

### **Pre-Deployment Checklist**
- [ ] **Training Validation**: Model performs well on validation set
- [ ] **Out-of-Sample Testing**: Test on unseen data
- [ ] **Backtesting**: Validate with historical trading simulation
- [ ] **Paper Trading**: Test in live environment with paper money
- [ ] **Performance Monitoring**: Track model drift and degradation

### **Model Monitoring**
```python
def monitor_model_performance(model_path: str, live_data: pd.DataFrame):
    """Monitor model performance in live trading"""
    
    # Track prediction accuracy
    accuracy_history = []
    
    # Check for model drift
    drift_detected = detect_model_drift(model_path, live_data)
    
    # Alert if performance degrades
    if drift_detected:
        send_alert("Model performance degradation detected")
    
    return accuracy_history
```

### **Model Retraining Triggers**
- **Performance Degradation**: Accuracy drops below threshold
- **Market Regime Change**: Significant change in market conditions
- **Data Drift**: Statistical properties of input data change
- **Scheduled Retraining**: Monthly retraining with latest data

---

## 🔄 Model Deployment

### **Safe Deployment Process**
```bash
# 1. Train model in staging
python scripts/safe_model_trainer.py

# 2. Validate model
python scripts/simple_model_validator.py

# 3. Deploy to production
cp ml/staging/btcusdt_sentiment.* ml/

# 4. Restart trading engine
python scripts/run_live_trading.py ml_with_sentiment --paper-trading
```

### **Rollback Procedure**
```bash
# If model performance degrades
cp ml/backup/btcusdt_sentiment.* ml/

# Restart with previous model
python scripts/run_live_trading.py ml_with_sentiment --paper-trading
```

---

**For detailed implementation guides, use:**
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["commands"])` - Complete command reference