# Prediction Engine

> **Last Updated**: 2025-12-19  
> **Related Documentation**: See [docs/prediction.md](../../docs/prediction.md) for comprehensive guide

Centralized ONNX model loading, inference, and caching for all ML strategies.

## Components
- `config.py`: `PredictionConfig` (model registry paths, cache knobs, thresholds).
- `engine.py`: `PredictionEngine` that wires feature extraction, registry selection, and caching.
- `features/`: FeaturePipeline modules (technical, sentiment, market microstructure).
- `models/onnx_runner.py`: `OnnxRunner` executing ONNX artifacts with provider negotiation and caching.
- `models/registry.py`: `PredictionModelRegistry` that discovers bundles under `src/ml/models/`.
- `utils/caching.py`: TTL-based prediction cache plus helpers shared by runners and the engine.

`sitecustomize.py` injects both the project root and `src/` onto `sys.path`, so imports like `from prediction.engine import PredictionEngine` resolve for CLI and ad-hoc scripts alike.

## Usage
```python
import numpy as np

from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry

config = PredictionConfig.from_config_manager()
registry = PredictionModelRegistry(config)
bundle = registry.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")

# Build a zeroed feature tensor that matches the bundle schema (sequence_length x features)
sequence_length = int(bundle.metadata.get("sequence_length", 120))
feature_count = len(bundle.metadata.get("feature_names", [])) or len(
    bundle.feature_schema.get("features", [])
) or 5
features = np.zeros((1, sequence_length, feature_count), dtype=np.float32)

prediction = bundle.runner.predict(features)
print(bundle.version_id, prediction.price, prediction.confidence)
```

`PredictionModelRegistry` exposes additional helpers:
- `list_bundles()` – enumerate every discovered `(symbol, timeframe, model_type)` tuple.
- `select_bundle(symbol=..., model_type=..., timeframe=...)` – fetch the active bundle (latest symlink wins).
- `reload_models()` – refresh in-memory bundles after adding artifacts.
- `invalidate_cache(model_name=None)` – forward eviction requests to `PredictionCacheManager`.

## Model Storage

All models are now stored in the **structured registry format**:
- **Registry structure**: `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/model.onnx` (e.g., `BTCUSDT/basic/2025-09-17_1h_v1/model.onnx`)
- Each model directory contains: `model.onnx`, `model.keras`, `saved_model/`, `metadata.json`, and `feature_schema.json`
- The `latest/` symlink in each type directory points to the current production version

## Status

All strategies now exclusively use `PredictionModelRegistry` for model loading:
- Legacy flat structure has been removed from `src/ml/` root
- All signal generators (`MLSignalGenerator`, `MLBasicSignalGenerator`) use the registry by default
- Strategy factory functions (`create_ml_basic_strategy`, `create_ml_adaptive_strategy`, `create_ml_sentiment_strategy`) only accept registry parameters

## Migration (Completed October 2025)

The dual-mode architecture has been removed:
- ✅ Removed `DEFAULT_USE_PREDICTION_ENGINE` constant
- ✅ Removed `model_path` and `use_prediction_engine` parameters from all strategies
- ✅ Deleted legacy symlinks from `src/ml/` root
- ✅ All strategies now use registry-based model loading exclusively
