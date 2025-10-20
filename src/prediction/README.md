# Prediction Engine

Centralized ONNX model loading, inference, and caching for all ML strategies.

## Components
- `config.py`: `PredictionConfig` (paths, thresholds, cache TTL)
- `models/onnx_runner.py`: ONNX loader and inference (`OnnxRunner`)
- `models/registry.py`: `PredictionModelRegistry` (discovers models in `src/ml` at project root)
- `utils/caching.py`: TTL-based prediction cache and decorators

Note: `sitecustomize.py` adds both project root and `src/` to `sys.path`, so imports like `from prediction...` resolve when running scripts.

## Usage
```python
from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry
import numpy as np

config = PredictionConfig()
registry = PredictionModelRegistry(config)

features = np.random.rand(120, 5).astype(np.float32)
pred = registry.predict('btcusdt_price', features)
print(pred.price, pred.confidence, pred.direction)
```

## Status
- Strategies currently load ONNX directly; migration to `PredictionModelRegistry` is planned.
- Models are organized in two structures:
  - Flat structure (legacy): `src/ml/btcusdt_price.onnx`, `btcusdt_price_v2.onnx`, `btcusdt_sentiment.onnx`, `ethusdt_sentiment.onnx`
  - Nested structure (current): `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/model.onnx` (e.g., `BTCUSDT/basic/2025-09-17_1h_v1/model.onnx`)
- Metadata files are included in each versioned model directory.
