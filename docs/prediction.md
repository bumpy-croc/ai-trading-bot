# Prediction & models

> **Last Updated**: 2026-07-07  
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
from datetime import UTC, datetime, timedelta

from src.data_providers.binance_provider import BinanceProvider
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine

config = PredictionConfig.from_config_manager()
engine = PredictionEngine(config=config)

provider = BinanceProvider()
end = datetime.now(UTC)
start = end - timedelta(days=90)
df = provider.get_historical_data("BTCUSDT", "1h", start, end)
result = engine.predict(df)
print(result.price, result.confidence, result.model_name)
```

## Model registry

The registry (`src/prediction/models/registry.py`) loads model bundles from the path declared in `PredictionConfig.model_registry_path`.
Each bundle contains:

- `model.onnx` – the ONNX runtime artifact used by inference-heavy workflows.
- `model.keras` and (optionally) a `saved_model/` export – retained for retracing or fine-tuning.
- `metadata.json` – training parameters, evaluation summaries, and lineage (symbol, timeframe, model type).
- `feature_schema.json` – canonical schema describing the features the model expects at inference time.
- `metrics.json` (optional) – lightweight rollups surfaced by the CLI compare command.

Bundles are keyed by `(symbol, timeframe, model_type)` and can optionally expose a `latest/` symlink per model type so production
strategies always resolve to the current version without editing code.

### Model Storage Locations

All models are stored exclusively in the structured registry:
- `src/ml/models/SYMBOL/TYPE/VERSION/` – versioned directories that include the ONNX model, Keras SavedModel, metadata, and feature schema.

Example models:
- `BTCUSDT/basic/2025-10-30_12h_v1/` – BTC price prediction (basic, 1h timeframe)
- `BTCUSDT/sentiment/2025-09-17_1h_v1/` – BTC with sentiment analysis
- `ETHUSDT/sentiment/2025-09-17_1h_v1/` – ETH with sentiment analysis

The `latest/` symlink in each type directory (e.g., `BTCUSDT/basic/latest/`) points to the current production version. All strategies load models exclusively through the `PredictionModelRegistry`.

### Model Management Commands

Helper commands under `atb models` provide operational visibility:

- `atb models list` – list all discovered bundles grouped by symbol/timeframe/model type.
- `atb models compare BTCUSDT 1h basic` – print the `metrics.json` payload for the selected bundle (`model_type` is `basic`, `sentiment`, etc.); if the file is absent the command returns `{}`.
- `atb models validate` – reload all bundles to surface missing files or corrupt artifacts.
- `atb models promote BTCUSDT basic 2025-10-30_12h_v1` – repoint the `latest` symlink for `BTCUSDT/basic` to a specific version directory.

## Training and deployment

`atb train model` writes models directly into the registry at `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}` and refreshes the `latest`
symlink used by the prediction engine. Operations teams can also trigger training from the live-control CLI, which simply wraps the
same pipeline:

```bash
# Train a price-only model on the last 365 days and update the latest bundle
atb live-control train --symbol BTCUSDT --days 365 --epochs 50
```

To roll back, repoint the `latest` symlink with either `atb models promote …` or `atb live-control deploy-model --model-path BTCUSDT/basic/2025-09-17_1h_v1`.
Listing available bundles uses the same registry information:

```bash
atb live-control list-models
```

Models are stored in `src/ml/models` by default. Metadata JSON files capture training parameters so dashboards and audits can tie
strategy performance back to the model version in use.

### Training CLI options

Use the following knobs when running `atb train model` locally:

- `--epochs`, `--batch-size`, and `--sequence-length` adjust hyperparameters without editing code.
- `--skip-plots`, `--skip-robustness`, and `--skip-onnx` let you bypass the slowest diagnostics when you only need a quick experiment. Leave them off for production artifacts so the metadata and ONNX bundle stay in sync.
- `--disable-mixed-precision` falls back to float32 math if you encounter GPU/MPS precision glitches. Mixed precision remains enabled by default when a GPU is present to speed up long jobs.
- `--force-price-only` trains against the production price-only contract: the 5 `PriceOnlyFeatureExtractor` features (rolling causal min-max normalization) with `close_normalized` as the regression target — the same contract `atb train price` and live inference use. Bundles still land in the `price/` namespace; promotion into `basic/` stays an explicit step.

The defaults remain equivalent to the legacy behavior (300 epochs, batch size 32, sequence length 120, diagnostics on, ONNX on), so unattended jobs continue to produce identical artifacts unless you override the flags explicitly.

### Training data sourcing (single-source, fail-loud)

Training corpora are loaded through the year-based parquet cache (`atb data prefill-cache`)
before any network fetch; ranges missing from the cache are fetched from Binance and cached, so
repeated runs — e.g. multi-entrant tournaments — train on identical rows without re-downloading.
There is **no third-party fallback at training time**: if Binance/cache cannot fully cover the
requested window (open-time boundary slack, calendar-day start check, ≥99% expected-bar
coverage), training fails loudly instead of silently switching data sources mid-corpus (#909).
Fix a failure by prefilling the cache, passing `--input-data-s3`, or adjusting the date range.

## Cloud training (AWS SageMaker)

`atb train cloud` runs the same training pipeline on SageMaker spot GPU instances (`src/ml/cloud/`). Because Binance blocks
AWS IPs, the CLI always assembles the corpus locally (parquet cache first, Binance for gaps — see
"Training data sourcing" above), uploads it to S3 as the job's data channel, and the container trains
from that channel — both in blocking mode and with `--no-wait`.

```bash
# Fixed-cutoff experiment window (dates are UTC; --days and --start-date are mutually exclusive)
atb train cloud BTCUSDT --start-date 2026-05-01 --end-date 2026-06-01 --epochs 50 --force-price-only

# Production-style retrain: last 365 days ending now
atb train cloud BTCUSDT --days 365 --epochs 300

# Architecture selection (tournament entrants): lstm, cnn_lstm, attention_lstm,
# tcn, tcn_attention, tft x {default, lightweight, deep}
atb train cloud ETHUSDT --model-type tcn_attention --model-variant deep --force-price-only

# Async round trip
atb train cloud BTCUSDT --no-wait            # uploads data, submits, prints job id
atb train cloud-status <JOB_NAME>            # poll status
atb train cloud-status <JOB_NAME> --sync     # download + sync bundle into the registry
atb train cloud-list [BTCUSDT]               # list job outputs in S3 (newest first)
```

### Model tournaments run cloud-first

Architecture/model tournaments are run as **parallel SageMaker jobs, not sequential local
training** (Board decision 2026-07-06, #918): a 5-entrant sweep costs roughly $0.10–0.50 total
and finishes in under an hour of wall-clock time. Submit one `atb train cloud … --no-wait
--model-type <entrant>` job per entrant with an identical fixed `--start-date`/`--end-date`
window, then sync and evaluate the bundles. Local training remains the fallback only for
protocol-experimental runs that need unpushed pipeline patches. Before a tournament, confirm the
ECR image is fresh (see the rebuild note below) — the container bakes in `src/ml/training_pipeline/`.

### Namespace and promotion flow

Cloud bundles record `model_type: price`, so syncs land in `src/ml/models/{SYMBOL}/price/{VERSION}` and only move
`price/latest`. Live strategies load `{SYMBOL}/basic/latest`, which cloud training **never** touches — promotion to live is
always an explicit, separate step:

```bash
# Copy the bundle into basic/ WITHOUT touching basic/latest (safe default)
atb train cloud-promote BTCUSDT 2026-07-05_10h30m00s_v1 --to basic

# Only when you intend to change what live strategies load:
atb train cloud-promote BTCUSDT 2026-07-05_10h30m00s_v1 --to basic --set-latest
```

Version IDs include seconds (`YYYY-MM-DD_HHhMMmSSs_vN`), so parallel cloud jobs cannot collide; if a name somehow already
exists locally, the sync writes to a `-2`/`-3` suffixed sibling instead of overwriting the existing bundle.

### Cost expectations and operational notes

- Spot `ml.g4dn.xlarge` (default) bills ~$0.21–0.25/hr; measured runs: a 2-epoch smoke job was 36 billable seconds
  (~$0.0074), a full-history 1h retrain ≈ $0.37 (~1.3h). Instance startup (~3–4 min) is not billed.
- Requires `SAGEMAKER_ROLE_ARN`, `SAGEMAKER_S3_BUCKET`, `AWS_REGION`, and `SAGEMAKER_DOCKER_IMAGE` in the environment
  (one-time infra setup: `src/ml/cloud/README.md`).
- **Rebuild the ECR image whenever feature engineering changes** (`./src/ml/cloud/build-and-push.sh`) — the container bakes
  in `src/ml/training_pipeline/`, so a stale image trains with features that diverge from current inference code
  (train/serve skew).

## macOS GPU inference verification

macOS users can confirm that ONNX Runtime is activating the CoreML/MPS execution providers introduced in [issue #156](https://github.com/bumpy-croc/ai-trading-bot/issues/156) with the following steps:

1. **Install the GPU-enabled ONNX Runtime build.**
   ```bash
   pip install onnxruntime-silicon
   ```
   The `onnxruntime` PyPI package only enables CPU execution on Apple Silicon. The `onnxruntime-silicon` wheel ships the CoreML and MPS providers required for GPU acceleration.

2. **Inspect the detected providers.**
   ```bash
   python -m src.prediction.models.execution_providers --include-missing
   ```
   The command prints every provider exposed by the host runtime followed by the prioritized list used by the trading bot. On an Apple Silicon Mac with `onnxruntime-silicon` installed you should see `CoreMLExecutionProvider` and `MPSExecutionProvider` in both lists.

3. **(Optional) Validate against a model.**
   ```bash
   python -m src.prediction.models.execution_providers --model path/to/model.onnx
   ```
   When a model path is supplied, the helper loads the session with the preferred providers and echoes the providers ONNX Runtime actually activated. This confirms that the GPU-capable backend is used instead of falling back to CPU.

4. **Run the prediction unit tests.**
   ```bash
   pytest tests/unit/predictions/test_models.py tests/unit/predictions/test_prediction_caching.py -k provider
   ```
   The focused tests validate that the provider utility feeds the ONNX runner and caching layers correctly.

If any of the above steps omit the GPU providers, reinstall `onnxruntime-silicon`, ensure the Python environment is using that interpreter, and repeat the checks.

## Bear-market validation gate (#801)

Models trained predominantly on the long-biased 2023–2025 market can go long into
bear-market dead-cat bounces. Before a candidate model takes the `latest` symlink
it is now scored on a fixed set of historical windows and blocked if it fails.

- **Windows & thresholds** live in `config/validation_windows.json` (bear 2022,
  Oct 2025–Feb 2026 crash, Feb–Jun 2026 chop by default). Each window has a
  `max_drawdown_pct` cap and a `min_trades` floor. These are config, not code —
  update them as regimes evolve.
- **Harness**: `src/ml/validation/bear_validation.py::BearValidationHarness`
  backtests the model per window (reusing `ExperimentRunner`) and reports Sharpe,
  max-drawdown, win-rate and trade count. `mock`/`fixture` providers give
  deterministic, network-free runs for CI.
- **Gate**: `src/ml/validation/gate.py::promote_version_if_valid`. The registry
  resolves a model only by the `latest` symlink, so the gate scores the
  *candidate* by flipping `latest` to it, validating, and rolling back to the
  previously-live version on failure (canary-with-rollback — fail-safe). It
  writes an auditable `validation_audit.json` next to the version. A run that
  cannot execute (missing data) is *inconclusive* → soft-pass with a loud
  warning, unless `VALIDATION_REQUIRED` is truthy (then it blocks/rolls back).

### CLI

```bash
# Score a model without deploying (exit non-zero only on outright failure)
atb live-control validate-model --symbol BTCUSDT --model-type basic

# Deploy is validation-gated; --skip-validation is an audited human override
atb live-control deploy-model --model-path BTCUSDT/basic/<version>

# --auto-deploy keeps the freshly trained model only if validation passes;
# on failure 'latest' rolls back to the pre-training model
atb live-control train --symbol BTCUSDT --auto-deploy
```

> **Out of scope / human sign-off**: this gate does not retrain models and never
> flips a live-trading symbol's `latest` without the promotion passing (or an
> explicit `--skip-validation`). Actual retraining on bear-inclusive windows and
> live promotion remain human/ml-engineer actions.
