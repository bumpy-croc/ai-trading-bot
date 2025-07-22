# ML Adaptive Strategy: Bear Market Improvements

## Overview

The ML Adaptive strategy is an enhanced version of the ML Basic strategy, specifically designed to handle extreme market volatility and bear markets. It incorporates lessons learned from the catastrophic failure during the 2020 COVID crash where the original strategy lost 47.64% of capital.

## Key Improvements

### 1. Volatility-Based Position Sizing

**Problem Solved**: Fixed 10% position sizing regardless of market conditions led to excessive risk during volatile periods.

**Solution**:
- Base position size: 10% (normal conditions)
- Minimum position size: 2% (extreme conditions)
- Maximum position size: 15% (optimal conditions)
- Position size inversely scales with volatility
- Further reductions based on:
  - Market regime (25% size in crisis, 50% in volatile markets)
  - Prediction confidence
  - Consecutive losses (20% reduction per loss)

### 2. Market Regime Detection

**Problem Solved**: No awareness of changing market conditions.

**Solution**: Four market regimes identified:
- **Normal**: Volatility < 2% daily
- **Volatile**: 2% < Volatility < 5% daily
- **Crisis**: Volatility > 10% daily or ATR > 15%
- **Recovery**: 24 hours post-crisis

Each regime has different:
- Entry thresholds
- Position sizing
- Stop loss levels
- Risk parameters

### 3. Dynamic Stop Loss Management

**Problem Solved**: Fixed 2% stop loss too tight for volatile markets.

**Solution**:
- Base stop loss: 2% (normal markets)
- Crisis stop loss: Up to 8% (based on ATR)
- Calculation: max(base_stop_loss, ATR × 1.5)
- Regime adjustments:
  - Crisis: 2x multiplier
  - Volatile: 1.5x multiplier

### 4. Crisis Mode Protocols

**Problem Solved**: No special handling for extreme market events.

**Solution**:
- Crisis detection based on volatility thresholds
- Doubled confidence requirements for entries
- Emergency exit if losing > 50% of stop loss
- 24-hour cooldown period post-crisis
- Additional entry filters:
  - Price must be above lower Bollinger Band
  - MACD histogram must be positive

### 5. Risk Management Enhancements

**Problem Solved**: No daily loss limits or consecutive loss tracking.

**Solution**:
- Maximum daily loss: 5% of capital
- Maximum consecutive losses: 3 trades
- Daily loss tracking with automatic trading suspension
- Consecutive loss tracking with position size reduction

### 6. Enhanced Entry Conditions

**Problem Solved**: Low win rate (35.6%) and poor market timing.

**Solution**: Multi-factor entry validation:
- ML prediction confidence thresholds
- RSI < 70 (avoid overbought)
- Trend direction alignment
- Market regime filters
- Crisis mode extra caution

## Configuration Parameters

```python
# Risk Management
base_stop_loss_pct = 0.02      # 2% base
max_stop_loss_pct = 0.08       # 8% maximum
min_stop_loss_pct = 0.015      # 1.5% minimum

# Position Sizing
base_position_size = 0.10      # 10% base
min_position_size = 0.02       # 2% minimum
max_position_size = 0.15       # 15% maximum

# Volatility Thresholds
volatility_low_threshold = 0.02     # 2% daily
volatility_high_threshold = 0.05    # 5% daily
volatility_crisis_threshold = 0.10  # 10% daily

# Risk Limits
max_daily_loss_pct = 0.05          # 5% daily maximum
max_consecutive_losses = 3          # Stop after 3 losses
crisis_mode_cooldown_hours = 24     # Wait after crisis
```

## Expected Improvements

Based on the 2020 analysis, the ML Adaptive strategy should achieve:

1. **Reduced Maximum Drawdown**: From 53.9% to < 20%
2. **Improved Win Rate**: From 35.6% to > 50%
3. **Positive Sharpe Ratio**: From -2.70 to > 0
4. **Crisis Survival**: No strategy shutdown during extreme events

## Technical Indicators Used

- **ATR (Average True Range)**: For volatility-based stop loss
- **RSI**: To avoid overbought conditions
- **Bollinger Bands**: Support/resistance and volatility
- **MACD**: Trend confirmation
- **Moving Averages**: 20, 50, 200 periods for trend analysis

## Usage

```python
from strategies.ml_adaptive import MlAdaptive

# Initialize strategy
strategy = MlAdaptive(
    name="MlAdaptive",
    model_path="src/ml/btcusdt_price.onnx",
    sequence_length=120
)

# Use in backtesting or live trading
engine = Backtester(
    data_provider=data_provider,
    strategy=strategy,
    initial_balance=10000,
    commission=0.001
)
```

## Testing

The strategy includes comprehensive unit tests covering:
- Market regime detection
- Volatility calculations
- Dynamic stop loss adjustment
- Position sizing logic
- Crisis mode behavior
- Daily loss limits
- Consecutive loss tracking

Run tests with:
```bash
python -m pytest tests/test_ml_adaptive.py -v
```

## Future Enhancements

1. **Sentiment Integration**: Add sentiment indicators for better market timing
2. **Hedging Strategies**: Implement options or inverse positions during crisis
3. **ML Model Retraining**: Include bear market data in training set
4. **Mean Reversion Filters**: Identify oversold bounce opportunities
5. **Multi-Timeframe Analysis**: Combine multiple timeframes for confirmation