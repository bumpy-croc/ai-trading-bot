# Strategies

Built-in strategies using indicators and/or ML predictions.

## Included
- `ml_basic.py`: ONNX price model predictions

## Usage (with backtester)
```bash
python scripts/run_backtest.py ml_basic --symbol BTCUSDT --days 90
```

## Create your own
- Subclass `strategies.base.BaseStrategy`
- Implement `generate_signals()` and position sizing methods