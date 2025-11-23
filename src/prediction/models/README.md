# Prediction Models

Core prediction model components for ONNX runtime and model registry.

## Modules

- `onnx_runner.py`: ONNX model loader and inference engine
- `registry.py`: Model discovery, selection, and metadata helpers for all strategies

## Usage

```python
import numpy as np

from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry

config = PredictionConfig.from_config_manager()
registry = PredictionModelRegistry(config)

bundle = registry.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")
features = np.random.rand(120, 5).astype(np.float32)
prediction = bundle.runner.predict(features)
print(f"Price: {prediction.price}, Confidence: {prediction.confidence}")
```

## Model Discovery

Models are automatically discovered from the `src/ml/models` registry:
- `model.onnx` (and optional `model.keras`) files contain the trained models
- `metadata.json`, `feature_schema.json`, and optional `metrics.json` capture configuration, schema, and evaluation artifacts