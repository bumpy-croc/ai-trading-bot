# ML Models

Pretrained models and training artifacts used by strategies and the prediction engine.

## Structure

All models are now managed through the **PredictionModelRegistry** using a structured versioned layout:

### Registry Structure
Located at `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/` for versioned model management:
- `BTCUSDT/basic/2025-09-17_1h_v1/` - Bitcoin basic model with metadata
- `BTCUSDT/sentiment/2025-09-17_1h_v1/` - Bitcoin sentiment model with metadata
- `ETHUSDT/sentiment/2025-09-17_1h_v1/` - Ethereum sentiment model with metadata

Example registry layout:
```
models/
├── BTCUSDT/
│   ├── basic/
│   │   ├── 2025-09-17_1h_v1/
│   │   │   ├── model.onnx
│   │   │   ├── metadata.json
│   │   │   └── feature_schema.json
│   │   └── latest/ -> 2025-09-17_1h_v1/
│   └── sentiment/
│       ├── 2025-09-17_1h_v1/
│       │   ├── model.onnx
│       │   ├── metadata.json
│       │   └── feature_schema.json
│       └── latest/ -> 2025-09-17_1h_v1/
└── ETHUSDT/
    └── sentiment/
        ├── 2025-09-17_1h_v1/
        │   ├── model.onnx
        │   ├── metadata.json
        │   └── feature_schema.json
        └── latest/ -> 2025-09-17_1h_v1/
```

Each versioned directory contains:
- `model.onnx` - ONNX inference model
- `model.keras` - Original Keras model (optional)
- `metadata.json` - Model metadata (training params, metrics, features)
- `feature_schema.json` - Feature schema for validation

The `latest/` symlink points to the current production version for each model type.

## Available Models

All models are discovered automatically by the PredictionModelRegistry. Select models by:
- Symbol (e.g., "BTCUSDT", "ETHUSDT")
- Type (e.g., "basic", "sentiment")
- Timeframe (e.g., "1h", "1d")
- Version (optional – defaults to the target of the `latest/` symlink)

## Usage

### Via Prediction Engine
```python
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine

# candles_dataframe is a pandas DataFrame with OHLCV (and optional sentiment/prediction) columns.
config = PredictionConfig.from_config_manager()
engine = PredictionEngine(config=config)
prediction = engine.predict(candles_dataframe)
print(prediction.price, prediction.confidence)
```

### Manual bundle access
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

### Via Strategy Components
All ML strategies now automatically use the registry:
```python
from src.strategies.ml_basic import create_ml_basic_strategy

strategy = create_ml_basic_strategy(
    name="MlBasic",
    sequence_length=120,
    model_name=None,  # Auto-discovers from registry
    model_type="basic",
    timeframe="1h"
)
```

## Training New Models

Use the CLI to train and register new models:
```bash
# Train a new model (automatically registers in the nested structure)
atb train model BTCUSDT --timeframe 1h --start-date 2023-01-01 --end-date 2024-12-01

# The trainer creates a new version directory and updates the latest symlink
```

See [docs/prediction.md](../../docs/prediction.md) for training details and model deployment workflows.

## Migration Notes

As of October 2025, all legacy flat-structure files (symlinks in `src/ml/` root) have been removed. All strategies now exclusively use the PredictionModelRegistry to load models from the `models/` subdirectory.
