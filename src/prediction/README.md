# Prediction Engine

> **Last Updated**: 2025-10-26  
> **Related Documentation**: See [docs/prediction.md](../../docs/prediction.md) for comprehensive guide

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

## Model Storage

Models are available in two structures:
- **Flat (legacy)**: `src/ml/*.onnx` (e.g., `btcusdt_price.onnx`, `btcusdt_price_v2.onnx`, `btcusdt_sentiment.onnx`, `ethusdt_sentiment.onnx`) along with legacy artifacts (`.h5`, `.keras`, and `*_metadata.json` such as `btcusdt_price_metadata.json`)
- **Nested (current)**: `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/model.onnx` (e.g., `BTCUSDT/basic/2025-09-17_1h_v1/model.onnx`) with colocated `metadata.json`

## Status
- Strategies currently load ONNX directly from the legacy paths; migration to exclusive use of `PredictionModelRegistry` is planned.
- Both storage layouts remain in place for backward compatibility:
  - Flat structure (legacy, archival compatibility): `src/ml/btcusdt_price.onnx`, `btcusdt_price_v2.onnx`, `btcusdt_sentiment.onnx`, `ethusdt_sentiment.onnx`
  - Nested structure (current default for component strategies): `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/model.onnx` (e.g., `BTCUSDT/basic/2025-09-17_1h_v1/model.onnx`)
- Metadata lives alongside each model—legacy files keep the `*_metadata.json` naming, while the registry uses a single `metadata.json` per versioned directory.
