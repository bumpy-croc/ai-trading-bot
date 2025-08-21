# ML Models

Pretrained models and training artifacts used by strategies and the prediction engine.

## Contents
- `.onnx`: inference models used in production
- `.keras`/`.h5`: training artifacts
- `*_metadata.json`: model metadata consumed by the prediction engine

## Notes
- The prediction engine auto-discovers models in this folder
- See `docs/MODEL_TRAINING_AND_INTEGRATION_GUIDE.md` for training details
