## Model Training and Integration Guide

This guide explains how to train or retrain models and integrate them with the prediction engine and `MlBasic` strategy, without degrading performance.

### Core concepts
- **Prediction Engine**: Orchestrates feature extraction (`FeaturePipeline`) and model inference (`PredictionModelRegistry` → `OnnxRunner`).
- **Feature Extractors**:
  - `PriceOnlyFeatureExtractor`: outputs only normalized OHLCV features in fixed order for price-only models.
  - `TechnicalFeatureExtractor`: outputs normalized prices + technical indicators.
- **Models**: ONNX files located in `src/ml`, discovered by the model registry.
- **Strategy**: `MlBasic` uses the price-only features; entry/exit and position size remain in the strategy.

### Canonical input for MlBasic (price-only)
- **Feature count**: 5 (normalized OHLCV), in this exact order:
  - `close_normalized`, `volume_normalized`, `high_normalized`, `low_normalized`, `open_normalized`
- **Sequence length**: 120 bars
- **Normalization window**: 120 (rolling min–max)
- **Normalization formula** (per bar, per feature):
  - If `max(window) != min(window)`: `(x_t - min(window)) / (max(window) - min(window))`
  - Else: `0.5`
- **Model output semantics (MlBasic)**: Predicts next close on normalized scale; strategy denormalizes using the last 120-bar window.

### Where this is defined
- Price-only features and order:
  ```1:29:src/prediction/features/price_only.py
  class PriceOnlyFeatureExtractor(FeatureExtractor):
      def __init__(..., normalization_window=DEFAULT_NORMALIZATION_WINDOW):
          self._feature_names = [
              'close_normalized', 'volume_normalized',
              'high_normalized', 'low_normalized', 'open_normalized',
          ]
  ```
- Defaults:
  ```20:22:src/config/constants.py
  DEFAULT_SEQUENCE_LENGTH = 120
  DEFAULT_NORMALIZATION_WINDOW = 120
  ```

### Prediction Engine usage in `MlBasic`
- Enable via env or config:
  - `USE_PREDICTION_ENGINE=1` (default `False`)
  - Optional: `PREDICTION_ENGINE_MODEL_NAME` (else defaults to ONNX filename stem, e.g., `btcusdt_price`).
- When enabled, `MlBasic` uses `PriceOnlyFeatureExtractor` to guarantee 5 features in the correct order and passes sequences of shape `(1, 120, 5)` to the engine session.
- If the selected engine model reports a different `feature_count`, the strategy safely falls back to its original ONNX session to preserve behavior.

### ONNX packaging and metadata
- Place files in `src/ml`:
  - `your_model.onnx`
  - `your_model_metadata.json`
- Recommended metadata fields:
  ```json
  {
    "sequence_length": 120,
    "feature_count": 5,
    "price_normalization": { "mean": 0.0, "std": 1.0 },
    "normalization_params": {}
  }
  ```
- The engine reads metadata to validate inputs and optionally adjust normalization/denormalization.

### Training/retraining guidelines
- **Match input shape**: Train with `(sequence_length=120, feature_count=5)` if targeting `MlBasic` price-only.
- **Match feature order**: Use the exact order listed above; keep it consistent between training and inference.
- **Normalization**: Use rolling min–max over the same window as inference (120 bars) if predicting normalized next close. If you change normalization or predict a different target (returns, direction), update the strategy logic or move to engine-level prediction outputs.
- **Timeframe & instrument**: Train on the same timeframe (e.g., 1h) and instrument (e.g., BTCUSDT) you will backtest/live trade.
- **Data hygiene**: Avoid leakage; set robust train/valid/test splits aligned with time.
- **Output semantics**:
  - Price regression (normalized next close): no strategy change required.
  - Returns or classification: update `MlBasic` entry/exit/position-size logic or consume engine’s `confidence`/`direction` outputs.
- **Model naming**: By default, the engine uses the ONNX filename stem as the model name; keep a clear, stable naming convention (e.g., `btcusdt_price_v2`).

### Extending feature sets (beyond 5)
- Create a new extractor under `src/prediction/features/` that outputs your trained feature set in the exact order used during training.
- Update `MlBasic` (engine-enabled path) to use that extractor instead of `PriceOnlyFeatureExtractor`.
- Ensure model metadata (`feature_count`) matches the new extractor output.

### Integration checklist (before replacing a model)
- [ ] ONNX exported to `src/ml/your_model.onnx`
- [ ] Metadata JSON created with `sequence_length`, `feature_count`, and any normalization hints
- [ ] Feature extractor produces exactly the features in the same order as training
- [ ] `USE_PREDICTION_ENGINE=1` smoke test passes and returns are ≥ baseline
- [ ] Optional: Set `PREDICTION_ENGINE_MODEL_NAME` to your model name for clarity

### Quick start: plugging in a new price-only model
1. Export ONNX and metadata to `src/ml` (use `sequence_length=120`, `feature_count=5`).
2. Enable engine: `USE_PREDICTION_ENGINE=1`.
3. Run unit tests: `pytest -q tests/unit/strategies/test_ml_basic_unit.py -n 4`.
4. If returns drop or dimension errors appear:
   - Verify metadata and feature order.
   - Ensure extractor matches training features.
   - Confirm timeframe alignment.

### Troubleshooting
- INVALID_ARGUMENT (Got: X Expected: Y): feature dimension mismatch; fix extractor/model metadata; `MlBasic` will temporarily fall back to its original session.
- Performance drop:
  - Check timeframe/instrument alignment, feature drift, normalization consistency.
  - Compare denormalization scale to training.
- No model found: ensure files are in `src/ml` and naming matches expectations; set `PREDICTION_ENGINE_MODEL_NAME` if needed.

### Testing commands
- MlBasic unit tests:
  - `pytest -q tests/unit/strategies/test_ml_basic_unit.py -n 4`
- MlBasic logging tests:
  - `pytest -q tests/unit/strategies/test_ml_basic_logging_unit.py -n 4`

### CI considerations
- Keep engine disabled by default; enable in dedicated jobs to validate new models.
- Enforce a minimum return threshold vs. baseline to avoid regressions.
- Add unit tests to validate strategy core functionality and logging behavior.
- Keep smoke test threshold unchanged; add a performance budget test once batching is implemented.

### Roadmap for fuller integration
- Migrate `MlBasic` to call `engine.predict(...)` directly once output scale/semantics are confirmed identical.
- Introduce extractor profiles per strategy (price-only vs. technical) and bind models accordingly via config.
- Add model versioning metadata and performance benchmarks for auto-selection.

### Engine integration checklist
- [ ] Enable via `USE_PREDICTION_ENGINE=1` and select model (`PREDICTION_ENGINE_MODEL_NAME` or filename stem)
- [ ] Ensure extractor matches model features (count, order); for MlBasic use `PriceOnlyFeatureExtractor`
- [ ] Confirm metadata (`sequence_length`, `feature_count`) present
- [ ] Run unit tests to validate strategy functionality
- [ ] Run smoke test (returns ≥ baseline)
- [ ] Add performance budget test if needed (batching on)
