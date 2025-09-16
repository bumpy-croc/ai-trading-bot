# Strategies

Built-in strategies using indicators and/or ML predictions.

## Included
- `ml_basic.py`: ONNX price model predictions (price-only)
- `ml_sentiment.py`: ONNX model predictions with sentiment analysis
- `ml_adaptive.py`: Adaptive ML strategy with regime detection
- `bull.py`: Bull market optimized strategy  
- `bear.py`: Bear market optimized strategy

## Usage (with backtester)
```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_sentiment --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 90
```

## Strategy Details

### ML Sentiment Strategy
- Uses models trained with both price data and Fear & Greed Index sentiment
- Adaptive position sizing based on sentiment confidence
- Enhanced prediction accuracy during volatile market conditions
- Robust fallback when sentiment data is unavailable
- Supports both BTC and ETH models with sentiment integration

## Create your own
- Subclass `strategies.base.BaseStrategy`
- Implement required methods: `calculate_indicators()`, `check_entry_conditions()`, `check_exit_conditions()`, `calculate_position_size()`, `calculate_stop_loss()`, `get_parameters()`
