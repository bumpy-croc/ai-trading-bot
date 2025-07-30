# Prediction Engine - Phase 3: Model Integration

## Overview

Phase 3 of the Prediction Engine implements centralized ONNX model loading, inference, and caching, replacing the direct model loading patterns in trading strategies.

## Components Implemented

### 1. Configuration System (`config.py`)
- `PredictionConfig`: Centralized configuration using constants from `src/config/constants.py`
- Configurable model registry path, confidence thresholds, caching TTL, etc.

### 2. ONNX Model Runner (`models/onnx_runner.py`)
- `OnnxRunner`: Handles ONNX model loading and inference
- `ModelPrediction`: Structured prediction results with confidence and direction
- Automatic metadata loading from `*_metadata.json` files
- Input preparation and output processing with normalization support

### 3. Model Registry (`models/registry.py`)
- `PredictionModelRegistry`: Centralized model management (replaces old `ModelRegistry`)
- Automatic discovery and loading of ONNX models from `src/ml/`
- Default model selection with priority for price models
- Model metadata access and management functions

### 4. Caching System (`utils/caching.py`)
- `ModelCache`: TTL-based caching for predictions
- `@cache_prediction`: Decorator for automatic prediction caching
- Configurable cache expiration times

### 5. Comprehensive Tests (`tests/predictions/test_models.py`)
- Unit tests for all components with >90% coverage
- Mocked ONNX runtime for isolated testing
- Test cases for error handling and edge cases

## Current Status

✅ **Phase 3 Complete**: All model integration components implemented and tested

### Models Successfully Loaded
- `btcusdt_price.onnx` (default model)
- `btcusdt_sentiment.onnx`
- `btcusdt_price_v2.onnx`

### Legacy Components Removed
- ❌ Old `src/ml/model_registry.py` (deleted)
- ✅ Updated `src/ml/__init__.py` to remove old imports

## Usage Example

```python
from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry
import numpy as np

# Initialize prediction engine
config = PredictionConfig()
registry = PredictionModelRegistry(config)

# Get available models
models = registry.list_models()
print(f"Available models: {models}")

# Make a prediction
features = np.random.rand(120, 5).astype(np.float32)
prediction = registry.predict('btcusdt_price', features)

print(f"Price: {prediction.price}")
print(f"Confidence: {prediction.confidence}")
print(f"Direction: {prediction.direction}")
print(f"Inference time: {prediction.inference_time}ms")
```

## Integration with Strategies

The strategies (`MlBasic`, `MlAdaptive`, `MlWithSentiment`) currently still use direct ONNX loading. These will be updated in **Phase 4: Strategy Integration** to use the new prediction engine.

## Next Steps

- **Phase 4**: Update strategies to use `PredictionModelRegistry`
- **Phase 5**: Testing & validation to ensure identical results
- **Future**: Feature engineering pipeline (Phase 2) and ensemble methods