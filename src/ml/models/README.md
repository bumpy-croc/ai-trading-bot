# Model Registry

Structured storage for versioned models organized by symbol, model type, and version. Every bundle ships the artifacts required for
inference (`model.onnx`), retraining (`model.keras` + `saved_model/`), and operational introspection (`metadata.json`,
`feature_schema.json`).

## Directory Structure

```
models/
├── SYMBOL/                 # Trading pair (e.g., BTCUSDT, ETHUSDT)
│   └── TYPE/               # Model type (basic, sentiment, adaptive, etc.)
│       ├── VERSION/        # Version identifier (e.g., 2025-10-30_12h_v1)
│       │   ├── model.onnx
│       │   ├── model.keras
│       │   ├── metadata.json
│       │   ├── feature_schema.json
│       │   └── saved_model/
│       └── latest -> VERSION   # Symlink to the active production bundle
```

Example structure:

```
models/
├── BTCUSDT/
│   ├── basic/
│   │   ├── 2025-10-30_12h_v1/
│   │   │   ├── model.onnx
│   │   │   ├── model.keras
│   │   │   ├── metadata.json
│   │   │   ├── feature_schema.json
│   │   │   └── saved_model/
│   │   └── latest -> 2025-10-30_12h_v1
│   └── sentiment/
│       ├── 2025-09-17_1h_v1/
│       └── latest -> 2025-09-17_1h_v1
└── ETHUSDT/
    └── sentiment/
        ├── 2025-09-17_1h_v1/
        └── latest -> 2025-09-17_1h_v1
```

## Model Types

- **basic/** – Price-only prediction models using technical indicators
- **sentiment/** – Models with sentiment features (Fear & Greed Index integration)
- **adaptive/** – Models with regime-aware adaptation capabilities

## Version Naming Convention

Versions follow the pattern: `YYYY-MM-DD_TIMEFRAME_vN`
- **Date**: Model training date
- **Timeframe**: Target timeframe (e.g., 1h, 4h, 1d)
- **Version**: Sequential version number

## Usage

Models are loaded by the `PredictionModelRegistry`, which auto-discovers this structured layout and resolves `latest/`
symlinks per `(symbol, model_type)` automatically. Legacy flat directories are no longer scanned; move artifacts into this
hierarchy to make them available to strategies.

For training new models and deploying to this registry, see [docs/prediction.md](../../../docs/prediction.md).
