# Strategies

Built-in strategies using indicators and/or ML predictions.

## Included
- `ml_basic.py`: ONNX price model predictions
- `ml_with_sentiment.py`: Price + sentiment model
- `ml_adaptive.py`: Adaptive ML logic and sizing

## Usage (with backtester)
```bash
python scripts/run_backtest.py ml_basic --symbol BTCUSDT --days 90
python scripts/run_backtest.py ml_with_sentiment --symbol BTCUSDT --days 365
```

## Create your own
- Subclass `strategies.base.BaseStrategy`
- Implement `generate_signals()` and position sizing methods