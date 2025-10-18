# ML Models

Pretrained models and training artifacts used by strategies and the prediction engine.

## Directory Structure

### Root Level (Legacy)
- `*.onnx` - Legacy ONNX inference models (e.g., `btcusdt_price.onnx`, `btcusdt_sentiment.onnx`)
- `*.keras` / `*.h5` - Keras training artifacts
- `*_metadata.json` - Model metadata files consumed by the prediction engine
- `*_training.png` - Training performance visualizations

### Structured Registry (`models/`)
New model registry structure for versioned model management:
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
