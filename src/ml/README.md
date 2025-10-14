# ML Models

Pretrained models and training artifacts used by strategies and the prediction engine.

## Contents
- `.onnx`: inference models used in production
- `.keras`/`.h5`: training artifacts
- `*_metadata.json`: model metadata consumed by the prediction engine

## Notes
- The prediction engine auto-discovers models in this folder
- See [docs/prediction.md](../../docs/prediction.md) for training details
