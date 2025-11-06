# Prediction & models

> **Last Updated**: 2025-11-06  
> **Related Documentation**: [Backtesting](backtesting.md), [Live trading](live_trading.md)

Machine-learning inference and model lifecycle management live under `src/prediction` and `src/ml`. The goal is to keep training
isolated from live execution while still exposing predictions to strategies in a consistent way.

## Prediction engine

`PredictionEngine` (`src/prediction/engine.py`) orchestrates feature extraction, model selection, and inference:

- Builds features through `FeaturePipeline` (technical, sentiment, and market microstructure inputs).
- Technical indicators and normalization live in `src.tech.features.technical`; `src/prediction/features/technical.py` now re-exports the extractor for compatibility.
- Selects models from `PredictionModelRegistry`, supporting per-strategy bundles and latest-version symlinks.
- Optionally caches results in the database (`PredictionCacheManager`) to avoid duplicate inferences during tight polling loops.
- Supports ensemble aggregation and regime-aware confidence adjustments when enabled in `PredictionConfig`.

Usage example:

```python
from datetime import datetime, timedelta

from src.data_providers.binance_provider import BinanceProvider
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine

config = PredictionConfig.from_config_manager()
engine = PredictionEngine(config=config)

provider = BinanceProvider()
end = datetime.utcnow()
start = end - timedelta(days=90)
df = provider.get_historical_data("BTCUSDT", "1h", start, end)
result = engine.predict(df)
print(result.price, result.confidence, result.model_name)
```

## Model registry

The registry (`src/prediction/models/registry.py`) loads model bundles from the path declared in `PredictionConfig.model_registry_path`.
Each bundle stores weights, metadata, and optional metrics.

### Model Storage Locations

All models are now stored exclusively in the **structured registry**:
- `src/ml/models/SYMBOL/TYPE/VERSION/model.onnx` - Versioned model structure

Example models:
- `BTCUSDT/basic/2025-09-17_1h_v1/` - BTC price prediction (basic)
- `BTCUSDT/sentiment/2025-09-17_1h_v1/` - BTC with sentiment analysis
- `ETHUSDT/sentiment/2025-09-17_1h_v1/` - ETH with sentiment analysis

The `latest/` symlink in each type directory (e.g., `BTCUSDT/basic/latest/`) points to the current production version. All strategies now load models exclusively through the `PredictionModelRegistry`.

### Model Management Commands

Helper commands under `python -m cli models` provide operational visibility:

- `python -m cli models list` – list all discovered bundles grouped by symbol/timeframe/model type.
- `python -m cli models compare BTCUSDT 1h price` – print the metrics metadata for the selected bundle.
- `python -m cli models validate` – reload all bundles to surface missing files or corrupt artifacts.
- `python -m cli models promote BTCUSDT price 2024-03-01` – repoint the `latest` symlink to a specific version.

## Training and deployment

`python -m cli train` now writes models directly into the registry at `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}` and refreshes the `latest`
symlink used by the prediction engine. Operations teams can still trigger training from the live-control CLI, which simply wraps the
same pipeline:

```bash
# Train a price-only model on the last 365 days and update the latest bundle
python -m cli live-control train --symbol BTCUSDT --days 365 --epochs 50
```

To roll back, repoint the `latest` symlink with either `python -m cli models promote …` or `python -m cli live-control deploy-model --model-path BTCUSDT/basic/2025-09-17_1h_v1`.
Listing available bundles uses the same registry information:

```bash
python -m cli live-control list-models
```

Models are stored in `src/ml/models` by default. Metadata JSON files capture training parameters so dashboards and audits can tie
strategy performance back to the model version in use.

### Training CLI options

Use the following knobs when running `python -m cli train model` locally:

- `--epochs`, `--batch-size`, and `--sequence-length` adjust hyperparameters without editing code.
- `--skip-plots`, `--skip-robustness`, and `--skip-onnx` let you bypass the slowest diagnostics when you only need a quick experiment. Leave them off for production artifacts so the metadata and ONNX bundle stay in sync.
- `--disable-mixed-precision` falls back to float32 math if you encounter GPU/MPS precision glitches. Mixed precision remains enabled by default when a GPU is present to speed up long jobs.

The defaults remain equivalent to the legacy behavior (300 epochs, batch size 32, sequence length 120, diagnostics on, ONNX on), so unattended jobs continue to produce identical artifacts unless you override the flags explicitly.
