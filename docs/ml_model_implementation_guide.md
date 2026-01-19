# ML Model Implementation Guide

**Purpose:** Guide for using, training, and extending the ML model architectures
**Audience:** Developers, data scientists, contributors
**Last Updated:** November 2025

---

## Quick Start

### Training a New Model Architecture

```bash
# Train Attention-LSTM on BTCUSDT
atb train model BTCUSDT --model-type attention_lstm --days 365 --epochs 100

# Train TCN (faster training)
atb train model BTCUSDT --model-type tcn --days 365 --epochs 80

# Train with specific variant (lightweight, default, deep)
atb train model BTCUSDT --model-type attention_lstm --variant lightweight --days 180
```

### Using Models in Code

```python
from src.ml.training_pipeline.models import create_model

# Create Attention-LSTM model
model = create_model('attention_lstm', input_shape=(60, 15))

# Create lightweight TCN
model = create_model('tcn', input_shape=(60, 15), variant='lightweight')

# Create with custom parameters
model = create_model('attention_lstm', input_shape=(60, 15),
                     lstm_units=[256, 128], dropout=0.3)

# Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=50)
```

---

## Available Model Architectures

### 1. CNN-LSTM (Baseline)

**Type:** `'cnn_lstm'` or `'adaptive'`
**Description:** Original hybrid model combining CNN feature extraction with LSTM temporal modeling

**Usage:**
```python
model = create_model('cnn_lstm', input_shape=(60, 15), has_sentiment=True)
```

**When to Use:**
- Baseline comparison
- Proven performance
- Good balance of speed/accuracy

---

### 2. Attention-LSTM ⭐ RECOMMENDED

**Type:** `'attention_lstm'`
**Description:** LSTM with multi-head attention mechanism

**Variants:**
- `'default'`: Balanced (128, 64 LSTM units)
- `'lightweight'`: Faster (64, 32 LSTM units)
- `'deep'`: More accurate (256, 128, 64 LSTM units)

**Usage:**
```python
# Default variant
model = create_model('attention_lstm', input_shape=(60, 15))

# Lightweight variant
model = create_model('attention_lstm', input_shape=(60, 15), variant='lightweight')

# Custom configuration
model = create_model('attention_lstm', input_shape=(60, 15),
                     lstm_units=[128, 64],
                     num_attention_heads=4,
                     attention_key_dim=64,
                     dropout=0.2)
```

**Expected Performance:**
- 12-15% improvement in MAE/MSE vs vanilla LSTM
- R² > 0.94 on financial data
- Inference: 50-100ms

**When to Use:**
- When accuracy is critical
- When interpretability needed (attention weights)
- Good for multi-step predictions

---

### 3. Temporal Convolutional Network (TCN) ⭐ RECOMMENDED

**Type:** `'tcn'`
**Description:** Dilated causal convolutions with residual connections

**Variants:**
- `'default'`: Balanced (64 filters, 5 layers)
- `'lightweight'`: Faster (32 filters, 4 layers)
- `'deep'`: More accurate (128 filters, 6 layers)

**Usage:**
```python
# Default variant
model = create_model('tcn', input_shape=(60, 15))

# Custom configuration
model = create_model('tcn', input_shape=(60, 15),
                     num_filters=64,
                     kernel_size=5,
                     num_layers=5,
                     dropout=0.2)

# TCN with attention
model = create_model('tcn_attention', input_shape=(60, 15))
```

**Expected Performance:**
- Competitive with LSTM
- 3-5x faster training
- Inference: <50ms

**Receptive Field:**
```python
from src.ml.training_pipeline.models_tcn import calculate_receptive_field

# Calculate how far back model can see
rf = calculate_receptive_field(kernel_size=5, num_layers=5)
print(f"Receptive field: {rf} timesteps")  # 125 timesteps
```

**When to Use:**
- When training speed critical
- Real-time inference requirements
- Shorter timeframes (1h, 4h)

---

### 4. LightGBM (Gradient Boosting)

**Description:** Fast gradient boosting baseline with manual feature engineering

**Usage:**
```python
from src.ml.training_pipeline.models_lightgbm import (
    create_lightgbm_model,
    engineer_features_for_lgb,
    train_lightgbm_with_early_stopping,
    get_feature_importance
)

# Engineer features
features_df, feature_names = engineer_features_for_lgb(price_df)
X = features_df[feature_names].dropna()
y = features_df['close'].loc[X.index]

# Split data
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# Create and train model
model = create_lightgbm_model(n_estimators=1000, learning_rate=0.05)
model = train_lightgbm_with_early_stopping(model, X_train, y_train, X_val, y_val)

# Get feature importance
importance = get_feature_importance(model, feature_names, top_n=20)
print(importance)
```

**Expected Performance:**
- 10-100x faster training than LSTM
- Competitive accuracy
- Inference: <10ms

**When to Use:**
- Quick baseline
- Feature importance analysis
- Ensemble component
- Directional prediction

---

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| Maximum accuracy | Attention-LSTM (deep) | Best accuracy, interpretable |
| Real-time inference | TCN | Fastest inference (<50ms) |
| Fast experimentation | LightGBM | 10-100x faster training |
| Multi-horizon forecasting | TFT (future) | Built for multi-horizon |
| Balanced performance | Attention-LSTM (default) | Good accuracy + speed |
| Resource-constrained | TCN (lightweight) | Low memory, fast |

---

## Training Pipeline Integration

### Step 1: Modify Training Configuration

```python
# In src/ml/training_pipeline/config.py or via CLI
config = TrainingConfig(
    symbol="BTCUSDT",
    model_type="attention_lstm",  # New parameter
    variant="default",             # New parameter
    sequence_length=60,
    epochs=100,
    batch_size=32,
)
```

### Step 2: Update Pipeline to Use Factory

```python
# In src/ml/training_pipeline/pipeline.py

from src.ml.training_pipeline.models import create_model, get_model_callbacks

# Replace direct model creation with factory
model_type = ctx.config.model_type or "cnn_lstm"  # Default to baseline
variant = ctx.config.variant or "default"

model = create_model(
    model_type=model_type,
    input_shape=(ctx.config.sequence_length, len(feature_names)),
    variant=variant,
    has_sentiment=has_sentiment  # For CNN-LSTM compatibility
)

callbacks = get_model_callbacks(model_type, patience=15)
history = model.fit(train_ds, validation_data=val_ds, epochs=ctx.config.epochs, callbacks=callbacks)
```

---

## Benchmarking Models

### Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmark/test_model_architectures.py -v

# Run specific benchmark
pytest tests/benchmark/test_model_architectures.py::test_attention_lstm_vs_baseline -v

# Run comprehensive comparison
pytest tests/benchmark/test_model_architectures.py::test_comprehensive_model_comparison -v

# Run inference speed benchmark
pytest tests/benchmark/test_model_architectures.py::test_inference_speed_benchmark -v
```

### Benchmark Output Example

```
=== Comprehensive Model Architecture Comparison ===
Model              Train(s)   Infer(ms)       RMSE        MAE     MAPE%        DA%
------------------------------------------------------------------------------------
cnn_lstm             120.50      45.30     0.0245     0.0189      3.45      62.30
attention_lstm       135.20      68.50     0.0215     0.0165      2.98      65.10
tcn                   42.30      28.70     0.0220     0.0170      3.05      64.20
tcn_attention         85.40      52.30     0.0218     0.0168      3.01      64.80

=== Best Performers ===
Best RMSE: attention_lstm (0.0215)
Best MAE: attention_lstm (0.0165)
Best Directional Accuracy: attention_lstm (65.10%)
Fastest Training: tcn (42.30s)
Fastest Inference: tcn (28.70ms)
```

### Adding Custom Benchmarks

```python
# In tests/benchmark/test_model_architectures.py

@pytest.mark.benchmark
@pytest.mark.slow
def test_my_custom_benchmark(benchmark_data, benchmark_config):
    """Test custom model configuration."""
    X_train, y_train, X_val, y_val = benchmark_data

    result = benchmark_model(
        "attention_lstm",
        "deep",
        X_train, y_train, X_val, y_val,
        benchmark_config
    )

    logger.info(f"Custom test RMSE: {result.rmse:.4f}")
    assert result.rmse < 0.05  # Your acceptance criteria
```

---

## Adding New Model Architectures

### Step 1: Create Model Module

Create `src/ml/training_pipeline/models_yourmodel.py`:

```python
"""Your new model architecture."""

import tensorflow as tf
from tensorflow.keras import Model, layers

def create_yourmodel(
    input_shape: tuple[int, int],
    # Your hyperparameters
    hidden_units: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Create your custom model.

    Args:
        input_shape: Shape of input sequences (sequence_length, num_features)
        hidden_units: Number of hidden units
        dropout: Dropout rate
        learning_rate: Learning rate

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Your architecture here
    x = layers.Dense(hidden_units, activation='relu')(inputs)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model

# Recommended hyperparameters
RECOMMENDED_HYPERPARAMETERS = {
    "default": {
        "hidden_units": 128,
        "dropout": 0.2,
        "learning_rate": 1e-3,
    }
}
```

### Step 2: Register in Factory

Update `src/ml/training_pipeline/models.py`:

```python
def create_model(model_type: str, input_shape: tuple[int, int], variant: str = "default", **kwargs):
    # ... existing code ...

    # Add your model
    elif model_type_lower == "yourmodel":
        try:
            from src.ml.training_pipeline.models_yourmodel import create_yourmodel
        except ImportError:
            raise ImportError("yourmodel requires models_yourmodel module")

        return create_yourmodel(input_shape, **kwargs)

    # ... rest of code ...

# Update registry
AVAILABLE_MODELS = {
    # ... existing models ...
    "yourmodel": "Your custom model architecture",
}
```

### Step 3: Add Tests

Create `tests/unit/test_yourmodel.py`:

```python
import pytest
import numpy as np

pytest.importorskip("tensorflow")

from src.ml.training_pipeline.models import create_model

def test_yourmodel_creation():
    """Test model creation."""
    model = create_model('yourmodel', input_shape=(60, 15))
    assert model is not None
    assert model.input_shape == (None, 60, 15)

def test_yourmodel_training():
    """Test model can train."""
    model = create_model('yourmodel', input_shape=(10, 5))

    # Generate synthetic data
    X = np.random.randn(100, 10, 5).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)

    # Train for 1 epoch
    history = model.fit(X, y, epochs=1, verbose=0)

    assert 'loss' in history.history
    assert len(history.history['loss']) == 1
```

### Step 4: Add Benchmarks

Add to `tests/benchmark/test_model_architectures.py`:

```python
@pytest.mark.benchmark
@pytest.mark.slow
def test_yourmodel_performance(benchmark_data, benchmark_config):
    """Benchmark your custom model."""
    X_train, y_train, X_val, y_val = benchmark_data

    result = benchmark_model(
        "yourmodel",
        "default",
        X_train, y_train, X_val, y_val,
        benchmark_config
    )

    logger.info(f"YourModel Results:")
    logger.info(f"  RMSE: {result.rmse:.4f}")
    logger.info(f"  MAE: {result.mae:.4f}")

    assert result.rmse > 0
```

---

## Interpretability & Debugging

### Extracting Attention Weights (Attention-LSTM)

```python
from src.ml.training_pipeline.models_attention_lstm import extract_attention_weights

# Load trained model
model = tf.keras.models.load_model('path/to/model.keras')

# Extract attention weights for visualization
attention_weights = extract_attention_weights(model, X_test[:10])

# Visualize what model focuses on
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(attention_weights[0], cmap='viridis')
plt.title('Attention Weights - What Model Focuses On')
plt.xlabel('Features')
plt.ylabel('Timesteps')
plt.show()
```

### Feature Importance (LightGBM)

```python
from src.ml.training_pipeline.models_lightgbm import get_feature_importance

# After training LightGBM
importance_df = get_feature_importance(model, feature_names, top_n=20)

print("Top 20 Most Important Features:")
print(importance_df)

# Visualize
import matplotlib.pyplot as plt

importance_df.plot(x='feature', y='importance', kind='barh', figsize=(10, 8))
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

### Model Performance Visualization

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Val RMSE')
plt.title('Model RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Hyperparameter Tuning

### Recommended Approach

1. **Start with defaults** - Use recommended hyperparameters from each model
2. **Grid search key params** - Focus on learning rate, dropout, layer sizes
3. **Use validation loss** - Optimize for validation performance, not training
4. **Early stopping** - Prevent overfitting with patience=10-20

### Example: Tuning Attention-LSTM

```python
from itertools import product

# Define hyperparameter grid
lstm_units_options = [[128, 64], [256, 128], [64, 32]]
dropout_options = [0.2, 0.3]
attention_heads_options = [2, 4, 8]

best_val_loss = float('inf')
best_params = None

for lstm_units, dropout, num_heads in product(lstm_units_options, dropout_options, attention_heads_options):
    model = create_model(
        'attention_lstm',
        input_shape=(60, 15),
        lstm_units=lstm_units,
        dropout=dropout,
        num_attention_heads=num_heads
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=get_model_callbacks('attention_lstm', patience=10),
        verbose=0
    )

    val_loss = min(history.history['val_loss'])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {
            'lstm_units': lstm_units,
            'dropout': dropout,
            'num_attention_heads': num_heads
        }

print(f"Best params: {best_params}")
print(f"Best val_loss: {best_val_loss:.4f}")
```

---

## Production Deployment

### Step 1: Export to ONNX

```python
import tf2onnx
import onnx

# Load trained model
model = tf.keras.models.load_model('path/to/model.keras')

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Save ONNX model
onnx.save(onnx_model, 'model.onnx')
```

### Step 2: Validate ONNX Export

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
sess = ort.InferenceSession('model.onnx')

# Test inference
X_test_sample = X_test[:1].astype(np.float32)
predictions = sess.run(None, {sess.get_inputs()[0].name: X_test_sample})

print(f"ONNX prediction: {predictions[0][0]}")

# Compare with Keras model
keras_pred = model.predict(X_test_sample, verbose=0)
print(f"Keras prediction: {keras_pred[0][0]}")

# Should be very close
assert np.allclose(predictions[0], keras_pred, atol=1e-5)
```

### Step 3: Memory & Speed Profiling

```python
import time
import psutil
import os

# Memory profiling
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# Load model
model = tf.keras.models.load_model('path/to/model.keras')

mem_after = process.memory_info().rss / 1024 / 1024  # MB
model_memory = mem_after - mem_before

print(f"Model memory usage: {model_memory:.2f} MB")

# Inference speed profiling
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = model.predict(X_test[:1], verbose=0)
    times.append((time.perf_counter() - start) * 1000)

print(f"Avg inference time: {np.mean(times):.2f} ms")
print(f"Min inference time: {np.min(times):.2f} ms")
print(f"Max inference time: {np.max(times):.2f} ms")

# Check production readiness
assert np.mean(times) < 100, "Inference too slow for production (<100ms target)"
assert model_memory < 100, "Model too large for Railway (< 100MB target)"
```

---

## Troubleshooting

### Common Issues

**1. Model not found error:**
```python
ValueError: Unknown model_type: attention_lstm
```
**Solution:** Ensure model module exists in `src/ml/training_pipeline/models_attention_lstm.py`

**2. Memory errors during training:**
```python
ResourceExhaustedError: OOM when allocating tensor
```
**Solutions:**
- Reduce batch size: `batch_size=16` instead of `32`
- Use lightweight variant: `variant='lightweight'`
- Enable mixed precision: `config.mixed_precision=True`

**3. Slow training:**
**Solutions:**
- Use TCN instead of LSTM (3-5x faster)
- Enable GPU: Verify with `tf.config.list_physical_devices('GPU')`
- Reduce sequence length: `sequence_length=30` instead of `60`

**4. Overfitting (val_loss >> train_loss):**
**Solutions:**
- Increase dropout: `dropout=0.3` instead of `0.2`
- Add more regularization
- Use more training data
- Enable early stopping

**5. Underfitting (high train_loss and val_loss):**
**Solutions:**
- Increase model capacity: Use `variant='deep'`
- Train longer: Increase `epochs`
- Check feature engineering
- Verify data quality

---

## Best Practices

### 1. Always Use Validation Set
```python
# Good: Train/Val/Test split
X_train, y_train, X_val, y_val = split_sequences(sequences, targets)
```

### 2. Monitor Multiple Metrics
```python
# Track RMSE, MAE, directional accuracy
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae')
    ]
)
```

### 3. Use Early Stopping
```python
callbacks = get_model_callbacks(model_type, patience=15)
history = model.fit(..., callbacks=callbacks)
```

### 4. Save Best Model
```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)
```

### 5. Version Your Models
```bash
# Model registry automatically versions
atb train model BTCUSDT --model-type attention_lstm
# Saved to: src/ml/models/BTCUSDT/price/2025-11-21_14h_v1/
```

---

## Additional Resources

- **Full Research Report:** `docs/ml_model_research_report.md`
- **Architecture Details:** `docs/ml_architecture_research.md`
- **Training Pipeline:** `docs/prediction.md`
- **Model Registry:** `docs/model_registry_usage.md`

---

**Document Version:** 1.0
**Last Updated:** November 2025
**Author:** AI Research (Claude)
**Status:** Complete
