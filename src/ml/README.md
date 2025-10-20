# ML Models

Pretrained models and training artifacts used by strategies and the prediction engine.

## Structure

Models are organized in two formats:

### Flat Structure (Legacy)
Located at `src/ml/` root:
- `*.onnx` - Legacy ONNX inference models (e.g., `btcusdt_price.onnx`, `btcusdt_sentiment.onnx`)
- `*.keras` / `*.h5` - Keras training artifacts
- `*_metadata.json` - Model metadata files consumed by the prediction engine
- `*_training.png` - Training performance visualizations

### Nested Structure (Current)
Located at `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/` for versioned model management:
- `BTCUSDT/basic/2025-09-17_1h_v1/` - Bitcoin basic model with metadata
- `BTCUSDT/sentiment/2025-09-17_1h_v1/` - Bitcoin sentiment model with metadata
- `ETHUSDT/sentiment/2025-09-17_1h_v1/` - Ethereum sentiment model with metadata

Example registry layout:
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

Each versioned directory contains:
- `model.onnx` - ONNX inference model
- `metadata.json` - Model metadata (training params, metrics, features)

## Notes
- The prediction engine auto-discovers models in both structures
- Nested structure is preferred for new models
- See [docs/prediction.md](../../docs/prediction.md) for training details

## Available Models

### BTC Models
- `btcusdt_price.onnx` - Basic price prediction
- `btcusdt_price_v2.onnx` - Enhanced price prediction (v2)
- `btcusdt_sentiment.onnx` - Price prediction with sentiment features

### ETH Models
- `ethusdt_sentiment.onnx` - ETH price prediction with sentiment features

## Usage

The prediction engine auto-discovers models in this folder. Both legacy (root-level) and structured registry models are supported.

See [docs/prediction.md](../../docs/prediction.md) for training details and model deployment workflows.
