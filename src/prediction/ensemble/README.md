# Prediction Ensembles

Utilities for combining multiple `ModelPrediction` objects into a single
decision.

## Modules

- `__init__.py` – Defines `SimpleEnsembleAggregator` (mean/median/weighted) and
  the `EnsembleResult` dataclass returned after aggregation.

## SimpleEnsembleAggregator

```python
from src.prediction.ensemble import SimpleEnsembleAggregator
from src.prediction.models.onnx_runner import ModelPrediction

aggregator = SimpleEnsembleAggregator(method="weighted")
prediction = aggregator.aggregate([
    ModelPrediction(price=100.5, confidence=0.6, direction=1, model_name="model_a", inference_time=0.01),
    ModelPrediction(price=101.2, confidence=0.8, direction=1, model_name="model_b", inference_time=0.02),
    ModelPrediction(price=99.4, confidence=0.4, direction=-1, model_name="model_c", inference_time=0.01),
])

print(prediction.price, prediction.confidence, prediction.direction)
```

- `method="mean"` – arithmetic mean of price/confidence.
- `method="median"` – median of price/confidence.
- `method="weighted"` – confidence-weighted averages (automatically falls back
  to equal weights when all confidences are zero).

Direction is determined via majority vote (`1` for bullish, `-1` for bearish,
`0` on ties).

## When to use

- Blend predictions from multiple ONNX bundles before handing the result to a
  trading strategy.
- Perform quick what-if experiments on ensemble weighting without wiring up a
  full strategy component.

The aggregator intentionally has no dependencies on the strategy subsystem,
which keeps it easy to reuse inside notebooks, CLI tooling, and unit tests.
