# Technical Indicators

Pure function implementations of common technical indicators for price analysis.

## Overview

This module provides efficient, vectorized implementations of technical indicators used throughout the trading system. All indicators are pure functions that operate on pandas DataFrames or Series.

## Available Indicators

### Trend Indicators
- **SMA (Simple Moving Average)** - Average price over N periods
- **EMA (Exponential Moving Average)** - Weighted average favoring recent prices
- **WMA (Weighted Moving Average)** - Linearly weighted average
- **DEMA (Double Exponential Moving Average)** - Reduced lag EMA
- **TEMA (Triple Exponential Moving Average)** - Further reduced lag
- **HMA (Hull Moving Average)** - Weighted moving average with minimal lag

### Momentum Indicators
- **RSI (Relative Strength Index)** - Momentum oscillator (0-100)
- **MACD (Moving Average Convergence Divergence)** - Trend and momentum
- **Stochastic Oscillator** - Momentum indicator comparing close to range
- **Williams %R** - Momentum indicator similar to stochastic
- **CCI (Commodity Channel Index)** - Cyclical trend indicator
- **ROC (Rate of Change)** - Momentum as percentage change

### Volatility Indicators
- **ATR (Average True Range)** - Volatility measurement
- **Bollinger Bands** - Volatility bands around moving average
- **Keltner Channels** - ATR-based volatility bands
- **Standard Deviation** - Statistical volatility measure
- **Historical Volatility** - Annualized price volatility

### Volume Indicators
- **OBV (On-Balance Volume)** - Cumulative volume flow
- **Volume SMA** - Average volume over period
- **VWAP (Volume Weighted Average Price)** - Intraday average price weighted by volume
- **Money Flow Index (MFI)** - Volume-weighted RSI
- **Accumulation/Distribution** - Volume flow indicator

### Support/Resistance
- **Pivot Points** - Traditional pivot levels
- **Fibonacci Retracement** - Key retracement levels
- **Donchian Channels** - Highest high and lowest low bands

## Usage

### Basic Usage
```python
from src.indicators.technical import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands
)

# Simple moving averages (adds ma_20, ma_50 columns)
df = calculate_moving_averages(df, periods=[20, 50])

# RSI (pass DataFrame or Series)
rsi = calculate_rsi(df, period=14)
df['rsi'] = rsi

# ATR for volatility (adds 'atr' column)
df = calculate_atr(df, period=14)

# Bollinger Bands (adds bb_middle, bb_upper, bb_lower columns)
df = calculate_bollinger_bands(df, period=20, std_dev=2.0)

print(df[['close', 'ma_20', 'ma_50', 'rsi', 'atr', 'bb_upper', 'bb_lower']].tail())
```

### Strategy Integration
```python
from src.strategies.components.signal_generators.base import SignalGenerator
from src.indicators.technical import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_atr
)

class MySignalGenerator(SignalGenerator):
    def generate_signal(self, df, index, regime=None):
        # Calculate indicators on-demand for the current candle
        df_with_indicators = calculate_moving_averages(df, periods=[20, 50])
        df_with_indicators['rsi'] = calculate_rsi(df_with_indicators, period=14)
        df_with_indicators = calculate_atr(df_with_indicators, period=14)
        
        # Generate signal based on indicators
        # ... signal logic here ...
        
        return signal
```

### Vectorized Operations
All indicators are optimized for pandas DataFrames and use vectorized operations for efficiency:

```python
import pandas as pd
from src.indicators.technical import calculate_rsi, calculate_atr

# Load large dataset
df = pd.read_csv('BTCUSDT_1h.csv', parse_dates=['timestamp'], index_col='timestamp')

# Calculate indicators efficiently on entire dataset
df['rsi'] = calculate_rsi(df, period=14)
df = calculate_atr(df, period=14)

# No loops needed - fully vectorized
```

## Performance Considerations

- All indicators use **vectorized pandas operations** for speed
- **Minimal memory allocation** - operations performed in-place where possible
- **NaN handling** - Early periods return NaN until sufficient data available
- **Type hints** - Full type annotations for IDE support

## Testing

```bash
# Run indicator tests
pytest tests/unit/indicators/ -v

# Test specific indicator
pytest tests/unit/indicators/test_technical.py::test_calc_sma -v
```

## Notes

- **Pure functions** - No side effects, deterministic outputs
- **Pandas-native** - Works seamlessly with DataFrame workflows
- **Battle-tested** - Extensively used in production strategies
- **Standard definitions** - Industry-standard indicator calculations
