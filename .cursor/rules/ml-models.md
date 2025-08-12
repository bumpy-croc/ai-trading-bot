---
description: ML models and integration (concise)
alwaysApply: false
---

### Model files (`src/ml/`)
- Price models: `btcusdt_price.(h5|keras|onnx)` with `btcusdt_price_metadata.json`
- Sentiment models: `btcusdt_sentiment.(h5|keras|onnx)` with `btcusdt_sentiment_metadata.json`

### Integration
- Real-time inference via ONNX in prediction engine (`src/prediction/models/onnx_runner.py`).
- Use confidence for position sizing (see strategies and risk manager).
- Fallback: if sentiment unavailable, use price-only model.

### Training/validation scripts
- Train: `python scripts/train_model.py`
- Validate: `python scripts/simple_model_validator.py`
- Analyze: `python scripts/safe_model_trainer.py`, `scripts/analyze_btc_data.py`

### Best practices
- Prevent overfitting; use time-based validation; keep features minimal.
- Store artifacts in `src/ml/` with metadata JSON.