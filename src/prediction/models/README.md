# Prediction Models

Core prediction model components for the ONNX runtime and the structured model registry.

## Modules

- `onnx_runner.py`: Thin wrapper around ONNX Runtime with caching hooks.
- `registry.py`: Discovers versioned bundles in `src/ml/models`, loads metadata, and exposes selection helpers.
- `exceptions.py`: Purpose-built exception types for loader and selection failures.

## Registry layout

Each bundle lives under `src/ml/models/{SYMBOL}/{MODEL_TYPE}/{VERSION}/`:

- `model.onnx` – primary inference artifact loaded by `OnnxRunner`.
- `model.keras` / `saved_model/` – retained for optional retraining or audits.
- `metadata.json` – training configuration (timeframe, hyperparameters, lineage).
- `feature_schema.json` – declarative schema required by the feature pipeline.
- `metrics.json` (optional) – evaluation summary surfaced by `atb models compare`.
- `latest/` – symlink pointing to the active production version for a given symbol/model type.

The registry path is configurable via `PredictionConfig.model_registry_path`, allowing alternate locations in tests or deployments.

## Selecting a bundle

```python
import numpy as np

from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry

cfg = PredictionConfig.from_config_manager()
registry = PredictionModelRegistry(cfg)

# Inspect discovered bundles
for bundle in registry.list_bundles():
    print(bundle.key, bundle.metadata.get("timeframe"))

# Load the production (latest) BTCUSDT basic bundle
bundle = registry.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")

# Run inference with prepared features
features = np.random.rand(120, 5).astype(np.float32)
prediction = bundle.runner.predict(features)

print(f"Version: {bundle.version_id}")
print(f"Metadata: {bundle.metadata}")
print(f"Prediction output: {prediction}")
```

`StrategyModel` instances returned by the registry expose parsed metadata, metrics, the feature schema, and the underlying runner so
strategies can reuse the same interface across backtesting and live trading.

## Command-line helpers

The `atb models` command family uses the same registry internals:

- `atb models list` – show all bundles grouped by `symbol/timeframe/model_type`.
- `atb models compare BTCUSDT 1h basic` – print `metrics.json` for a selected bundle.
- `atb models validate` – reload every bundle to surface corrupt or missing files.
- `atb models promote BTCUSDT basic 2025-10-30_12h_v1` – repoint the `latest` symlink for production use.

These tools are the preferred way to audit artifacts before wiring them into strategies or live-control workflows.