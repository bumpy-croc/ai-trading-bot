# Strategies

Built-in strategies using indicators and/or ML predictions.

## Included
- `ml_basic.py`: ONNX price model predictions (price-only)
- `ml_sentiment.py`: ONNX model predictions with sentiment analysis
- `ml_adaptive.py`: Adaptive ML strategy with regime detection
- `bull.py`: Bull market optimized strategy  
- `bear.py`: Bear market optimized strategy
- `ensemble_weighted.py`: Weighted ensemble strategy combining multiple strategies
- `momentum_leverage.py`: Aggressive momentum-based strategy with pseudo-leverage

## Usage (with backtester)
```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_sentiment --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ensemble_weighted --symbol BTCUSDT --timeframe 1h --days 90
atb backtest momentum_leverage --symbol BTCUSDT --timeframe 1h --days 90
```

## Strategy Details

### ML Sentiment Strategy
- Uses models trained with both price data and Fear & Greed Index sentiment
- Adaptive position sizing based on sentiment confidence
- Enhanced prediction accuracy during volatile market conditions
- Robust fallback when sentiment data is unavailable
- Supports both BTC and ETH models with sentiment integration

### Ensemble Weighted Strategy
- Combines multiple strategies (ML Basic, ML Adaptive, Bull, Bear) using weighted voting
- Performance-based dynamic weighting that adapts over time
- Aggressive position sizing (up to 80% allocation) with pseudo-leverage
- Advanced momentum and trend indicators for enhanced entry timing
- Wide stop losses (6%) and high profit targets (20%) to capture major moves

### Momentum Leverage Strategy
- Pure momentum-based approach with pseudo-leverage for beating buy-and-hold
- Ultra-aggressive position sizing (40-95% allocation) based on momentum strength
- Multi-timeframe momentum analysis (3, 7, 20 periods)
- Trend confirmation using exponential moving averages
- Volatility-based position scaling for optimal risk management

## Create your own
- Subclass `strategies.base.BaseStrategy`
- Implement required methods: `calculate_indicators()`, `check_entry_conditions()`, `check_exit_conditions()`, `calculate_position_size()`, `calculate_stop_loss()`, `get_parameters()`
