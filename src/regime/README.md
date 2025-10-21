# Regime Detection

Market regime detection and analysis for adaptive trading strategies.

## Overview

This module provides market regime detection capabilities to help strategies adapt to different market conditions (bull, bear, sideways, high volatility). The system uses multiple technical indicators and statistical measures to classify market regimes in real-time.

## Features

- **Market regime classification** - Bull, bear, sideways, and transitional states
- **Volatility regime detection** - High, normal, and low volatility periods
- **Trend strength analysis** - Quantifies trend strength and direction
- **Dynamic strategy adaptation** - Enable regime-aware strategy switching
- **Confidence scoring** - Provides confidence levels for regime classifications
- **Multi-timeframe analysis** - Analyze regimes across different timeframes

## Modules

- `detector.py` - Core regime detection logic with `RegimeDetector` class
- `enhanced_detector.py` - Enhanced detector with machine learning capabilities

## Configuration

```python
from src.regime.detector import RegimeConfig

config = RegimeConfig(
    slope_window=50,               # Window for trend slope calculation
    band_window=20,                # Window for Bollinger bands
    atr_window=14,                 # Window for ATR calculation
    atr_percentile_lookback=252,   # Lookback for ATR percentile
    trend_threshold=0.0,           # Threshold for trend detection
    r2_min=0.2,                    # Minimum R² for trend confidence
    atr_high_percentile=0.7,       # Percentile threshold for high volatility
    hysteresis_k=3,                # Confirmations required to switch regime
    min_dwell=12                   # Minimum bars to stay in regime
)
```

## Usage

### Basic Regime Detection
```python
from src.regime.detector import RegimeDetector
import pandas as pd

# Load your OHLCV data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Detect regimes
detector = RegimeDetector()
df_with_regimes = detector.annotate(df)

# Access regime information
print(df_with_regimes[['trend_label', 'vol_label', 'regime_label', 'regime_confidence']].tail())
```

### Enhanced Detection
```python
from src.regime.enhanced_detector import EnhancedRegimeDetector, EnhancedRegimeConfig

# Configure enhanced detector
config = EnhancedRegimeConfig(
    slope_window=40,
    atr_window=14,
    rsi_window=14,
    volume_sma_window=20
)

detector = EnhancedRegimeDetector(config)
df_enhanced = detector.annotate(df)

# Enhanced detector adds additional columns for momentum and volume analysis
print(df_enhanced.columns)
```

### Regime-Aware Backtesting
```bash
# Strategies handle regime awareness internally during backtests
atb backtest ml_basic --symbol BTCUSDT --days 365
```

### Live Trading with Regime Switching
```python
from src.live.regime_strategy_switcher import RegimeStrategyMapping

# Configure strategy mapping per regime
mapping = RegimeStrategyMapping(
    bull_strategy='bull',
    bear_strategy='bear',
    sideways_strategy='ml_basic'
)

# Engine will automatically switch strategies based on detected regime
```

## Regime Types

| Regime | Description | Characteristics |
|--------|-------------|-----------------|
| **Bull** | Strong uptrend | Rising prices, high momentum, low volatility |
| **Bear** | Strong downtrend | Falling prices, negative momentum, increasing volatility |
| **Sideways** | Range-bound | Minimal trend, mean-reverting, moderate volatility |
| **High Volatility** | Volatile market | Large price swings, unstable trends |
| **Transitional** | Changing regime | Mixed signals, uncertain direction |

## Indicators Used

- **ATR (Average True Range)** - Volatility measurement
- **ADX (Average Directional Index)** - Trend strength
- **Moving Average Slopes** - Trend direction
- **RSI (Relative Strength Index)** - Momentum
- **Bollinger Band Width** - Volatility expansion/contraction
- **Volume Analysis** - Confirmation signals

## Documentation

See [docs/backtesting.md](../../docs/backtesting.md#regime-detection) for an overview of regime switching workflows and CLI
tooling.
