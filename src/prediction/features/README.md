# Prediction Features

Compatibility shims and helpers that bridge legacy prediction pipelines with the
new `src/tech` feature stack.

## Structure

- `base.py`, `technical.py`, `market.py`, `price_only.py`, `sentiment.py` –
  thin wrappers that re-export the canonical implementations in
  `src.tech.features.*`. Importing from either location continues to work, but
  new code should depend on the `src.tech` packages directly.
- `schemas.py` – Shared pydantic helpers for validating feature schemas saved in
  model bundles.
- `selector.py` – Home of `FeatureSelector`, the utility that slices a prepared
  feature DataFrame into the exact ordering required by a model’s
  `feature_schema.json`.

## FeatureSelector example

```python
import pandas as pd

from src.prediction.features.selector import FeatureSelector

schema = {
    "sequence_length": 120,
    "features": [
        {"name": "close_normalized", "required": True, "normalization": {"mean": 0.0, "std": 1.0}},
        {"name": "rsi", "required": True},
    ],
}

selector = FeatureSelector(schema)
features_df = pd.DataFrame({
    "close_normalized": [0.1] * 200,
    "rsi": [55] * 200,
})
tensor = selector.select(features_df)
print(tensor.shape)  # (120, 2)
```

## When to use this package

- Maintain backward compatibility for modules that still import from
  `src.prediction.features.*`.
- Load feature schemas embedded in model metadata without reaching into
  `src.tech` internals.
- Convert enriched DataFrames produced by the `TechnicalFeatureExtractor` into
  numpy tensors ready for the ONNX runtime.

For new feature builders, implement them inside `src/tech/features` so training
and inference share the exact same code path, then optionally expose a shim here
if older modules still expect the historical import location.
