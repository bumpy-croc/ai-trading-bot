# ML Models

Pretrained models and training artifacts used by strategies and the prediction engine.

## Structure

Models are organized in two formats:

### Flat Structure (Legacy)
Located at `src/ml/` root:
- `btcusdt_price.onnx`, `btcusdt_price_v2.onnx` - Bitcoin price prediction models
- `btcusdt_sentiment.onnx` - Bitcoin sentiment-aware model
- `ethusdt_sentiment.onnx` - Ethereum sentiment-aware model
- Corresponding `.h5` and `.keras` training artifacts

### Nested Structure (Current)
Located at `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/`:
- `BTCUSDT/basic/2025-09-17_1h_v1/` - Bitcoin basic model with metadata
- `BTCUSDT/sentiment/2025-09-17_1h_v1/` - Bitcoin sentiment model with metadata
- `ETHUSDT/sentiment/2025-09-17_1h_v1/` - Ethereum sentiment model with metadata

Each versioned directory contains:
- `model.onnx` - ONNX inference model
- `metadata.json` - Model metadata (training params, metrics, features)

## Notes
- The prediction engine auto-discovers models in both structures
- Nested structure is preferred for new models
- See [docs/prediction.md](../../docs/prediction.md) for training details
