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
    calc_sma, 
    calc_ema,
    calc_rsi, 
    calc_atr,
    calc_macd,
    calc_bollinger_bands
)

# Simple moving average
sma_20 = calc_sma(df['close'], window=20)
sma_50 = calc_sma(df['close'], window=50)

# Exponential moving average
ema_12 = calc_ema(df['close'], window=12)
ema_26 = calc_ema(df['close'], window=26)

# RSI
rsi = calc_rsi(df['close'], window=14)

# ATR for volatility
atr = calc_atr(df, window=14)

# MACD
macd, signal, histogram = calc_macd(
    df['close'], 
    fast=12, 
    slow=26, 
    signal=9
)

# Bollinger Bands
upper, middle, lower = calc_bollinger_bands(
    df['close'],
    window=20,
    num_std=2
)
```

### Strategy Integration
```python
from src.strategies.base import BaseStrategy
from src.indicators.technical import calc_sma, calc_rsi, calc_atr

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, df):
        # Calculate multiple indicators
        df['sma_20'] = calc_sma(df['close'], window=20)
        df['sma_50'] = calc_sma(df['close'], window=50)
        df['rsi'] = calc_rsi(df['close'], window=14)
        df['atr'] = calc_atr(df, window=14)
        
        # Calculate MACD
        macd, signal, histogram = calc_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        return df
```

### Vectorized Operations
All indicators are optimized for pandas DataFrames and use vectorized operations for efficiency:

```python
import pandas as pd
from src.indicators.technical import calc_rsi, calc_atr

# Load large dataset
df = pd.read_csv('BTCUSDT_1h.csv', parse_dates=['timestamp'], index_col='timestamp')

# Calculate indicators efficiently on entire dataset
df['rsi'] = calc_rsi(df['close'], window=14)
df['atr'] = calc_atr(df, window=14)

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
