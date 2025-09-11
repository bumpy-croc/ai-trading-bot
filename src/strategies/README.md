# Strategies

Built-in strategies using indicators and/or ML predictions.

## Included
- `ml_basic.py`: ONNX price model predictions
- `ml_adaptive.py`: Adaptive ML strategy with regime detection
- `bull.py`: Bull market optimized strategy  
- `bear.py`: Bear market optimized strategy

## Usage (with backtester)
```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 90
```

## Create your own
- Subclass `strategies.base.BaseStrategy`
- Implement `generate_signals()` and position sizing methods
