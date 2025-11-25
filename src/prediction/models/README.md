# Prediction Models

Core prediction model components for ONNX runtime execution and bundle discovery.

## Modules

- `onnx_runner.py`: Loads ONNX artifacts, negotiates execution providers, and exposes `predict()`.
- `registry.py`: Discovers structured bundles inside `src/ml/models`, returning `StrategyModel` objects.

## StrategyModel bundle contents

Every bundle returned by `PredictionModelRegistry` includes:

- `symbol`, `timeframe`, `model_type`, and `version_id`.
- `metadata`: Parsed `metadata.json` (training parameters, horizons, lineage).
- `feature_schema`: Parsed `feature_schema.json` describing required inputs.
- `metrics`: Optional `metrics.json` surfaced by `atb models compare`.
- `runner`: An `OnnxRunner` instance ready to perform inference.
- `directory`: Filesystem path to the bundle for advanced tooling.

## Registry helpers

- `list_bundles()` – Enumerate every discovered `StrategyModel`.
- `select_bundle(symbol=..., model_type=..., timeframe=...)` – Fetch the active (latest) bundle for that triple.
- `select_many([...])` – Resolve multiple bundles atomically, raising when any are missing.
- `reload_models()` – Rescan the registry after copying artifacts.
- `invalidate_cache(model_name=None)` – Forward cache eviction to `PredictionCacheManager`.

## Example

```python
from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry

config = PredictionConfig.from_config_manager()
registry = PredictionModelRegistry(config)

for bundle in registry.list_bundles():
    print(f"{bundle.symbol}/{bundle.model_type} {bundle.timeframe} -> {bundle.version_id}")

btc_basic = registry.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")
print(btc_basic.runner.model_path)
```

## Model Discovery

Models are automatically discovered from the structured registry rooted at `src/ml/models`:
- `model.onnx` files contain the inference artifact used by `OnnxRunner`.
- `model.keras` and `saved_model/` preserve the TensorFlow export for retracing.
- `metadata.json`, `feature_schema.json`, and optional `metrics.json` drive selection, validation, and CLI summaries.

CLI helpers under `atb models` (`list`, `compare`, `validate`, `promote`) and `atb live-control` reuse this registry, so documentation changes here should stay in sync with [docs/prediction.md](../../docs/prediction.md).