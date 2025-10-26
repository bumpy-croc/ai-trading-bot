# Prediction & models

> **Last Updated**: 2025-10-26  
> **Related Documentation**: [Backtesting](backtesting.md), [Live trading](live_trading.md)

Machine-learning inference and model lifecycle management live under `src/prediction` and `src/ml`. The goal is to keep training
isolated from live execution while still exposing predictions to strategies in a consistent way.

## Prediction engine

`PredictionEngine` (`src/prediction/engine.py`) orchestrates feature extraction, model selection, and inference:

- Builds features through `FeaturePipeline` (technical, sentiment, and market microstructure inputs).
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

Models are stored in two locations:
- **Legacy location**: `src/ml/*.onnx` (root level) - Currently used by strategies
- **Structured registry**: `src/ml/models/SYMBOL/TYPE/VERSION/model.onnx` - New versioned model structure

Available models include:
- `btcusdt_price.onnx`, `btcusdt_price_v2.onnx` - BTC price prediction
- `btcusdt_sentiment.onnx` - BTC with sentiment analysis
- `ethusdt_sentiment.onnx` - ETH with sentiment analysis

### Model Management Commands

Helper commands under `atb models` provide operational visibility:

- `atb models list` – list all discovered bundles grouped by symbol/timeframe/model type.
- `atb models compare BTCUSDT 1h price` – print the metrics metadata for the selected bundle.
- `atb models validate` – reload all bundles to surface missing files or corrupt artifacts.
- `atb models promote BTCUSDT price 2024-03-01` – repoint the `latest` symlink to a specific version.

## Training and deployment

`SafeModelTrainer` (`src/ml/safe_model_trainer.py`) wraps training scripts so new models are prepared in `/tmp/ai-trading-bot-staging`
before being deployed:

```bash
# Train a price-only model on 365 days of data
atb live-control train --symbol BTCUSDT --days 365 --epochs 50 --auto-deploy
```

The trainer performs backups of the current live bundle, runs validation, and creates a deployment package. `atb live-control deploy-model`
promotes the staged bundle to the live models directory and can optionally close open positions before the swap.

Models are stored in `src/ml/models` by default. Metadata JSON files capture training parameters so dashboards and audits can tie
strategy performance back to the model version in use.
