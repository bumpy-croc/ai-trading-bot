# Model Registry

Structured storage for versioned models organized by symbol, model type, and version.

## Directory Structure

```
models/
├── SYMBOL/              # Trading pair (e.g., BTCUSDT, ETHUSDT)
│   └── TYPE/            # Model type (e.g., basic, sentiment, adaptive)
│       └── VERSION/     # Version identifier (e.g., 2025-09-17_1h_v1)
│           ├── model.onnx         # ONNX inference model
│           ├── metadata.json      # Model metadata
│           └── training_info.json # Training parameters and metrics
```

Example structure:
```
models/
├── BTCUSDT/
│   ├── basic/
│   │   └── 2025-09-17_1h_v1/
│   │       └── model.onnx
│   └── sentiment/
│       └── 2025-09-17_1h_v1/
│           └── model.onnx
└── ETHUSDT/
    └── sentiment/
        └── 2025-09-17_1h_v1/
            └── model.onnx
```

## Model Types

- **basic/** - Price-only prediction models using technical indicators
- **sentiment/** - Models with sentiment features (Fear & Greed Index integration)
- **adaptive/** - Models with regime-aware adaptation capabilities

## Version Naming Convention

Versions follow the pattern: `YYYY-MM-DD_TIMEFRAME_vN`
- **Date**: Model training date
- **Timeframe**: Target timeframe (e.g., 1h, 4h, 1d)
- **Version**: Sequential version number

## Current Models

### BTCUSDT
- `basic/2025-09-17_1h_v1` - Basic price prediction (1h timeframe)
- `sentiment/2025-09-17_1h_v1` - Price prediction with sentiment (1h timeframe)

### ETHUSDT
- `sentiment/2025-09-17_1h_v1` - Price prediction with sentiment (1h timeframe)

## Usage

Models are loaded by the `PredictionModelRegistry` which auto-discovers and manages model versions. The registry supports both this structured format and legacy models in the parent `ml/` directory.

For training new models and deploying to this registry, see [docs/prediction.md](../../../docs/prediction.md).
