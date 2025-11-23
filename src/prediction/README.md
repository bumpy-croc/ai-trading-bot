# Prediction Engine

> **Last Updated**: 2025-11-23  
> **Related Documentation**: See [docs/prediction.md](../../docs/prediction.md) for comprehensive guide

Centralized ONNX model loading, inference, and caching for all ML strategies.

## Components
- `config.py`: `PredictionConfig` (paths, thresholds, cache TTL)
- `models/onnx_runner.py`: ONNX loader and inference (`OnnxRunner`)
- `models/registry.py`: `PredictionModelRegistry` (discovers models in `src/ml/models/`)
- `utils/caching.py`: TTL-based prediction cache and decorators

Note: `sitecustomize.py` adds both project root and `src/` to `sys.path`, so imports like `from prediction...` resolve when running scripts.

## Usage

### Via `PredictionEngine`
```python
from src.prediction.engine import PredictionEngine
from src.prediction.config import PredictionConfig

# candles_dataframe is a pandas DataFrame with OHLCV (and optional sentiment) columns.
config = PredictionConfig.from_config_manager()
engine = PredictionEngine(config=config)

prediction = engine.predict(candles_dataframe)
print(prediction.price, prediction.confidence, prediction.model_name)
```

### Direct registry access (manual runners)
```python
import numpy as np

from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry

config = PredictionConfig.from_config_manager()
registry = PredictionModelRegistry(config)

bundle = registry.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")
features = np.random.rand(120, 5).astype(np.float32)
prediction = bundle.runner.predict(features)
print(prediction.price, prediction.confidence, prediction.direction)
```

## Model Storage

All models are now stored in the **structured registry format**:
- **Registry structure**: `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/model.onnx` (e.g., `BTCUSDT/basic/2025-09-17_1h_v1/model.onnx`)
- Each model directory contains: `model.onnx`, `metadata.json`, and `feature_schema.json`
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
