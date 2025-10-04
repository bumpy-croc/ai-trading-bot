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
    volatility_threshold=0.03,     # 3% volatility threshold
    trend_threshold=0.02,          # 2% trend strength threshold
    lookback_periods=50,           # 50 periods for analysis
    use_ml_enhancements=False      # Use ML-based regime detection
)
```

## Usage

### Basic Regime Detection
```python
from src.regime.detector import RegimeDetector

detector = RegimeDetector()
regime, confidence = detector.detect_current_regime(price_data)
print(f"Current regime: {regime} (confidence: {confidence:.2%})")
```

### Enhanced Detection with ML
```python
from src.regime.enhanced_detector import EnhancedRegimeDetector

detector = EnhancedRegimeDetector()
result = detector.detect_regime_with_features(
    price_data,
    volume_data,
    include_features=['volatility', 'trend', 'momentum']
)
print(f"Regime: {result.regime}, Features: {result.features}")
```

### Regime-Aware Backtesting
```bash
# Enable regime-aware strategy switching in backtests
atb backtest ml_basic --regime-aware --symbol BTCUSDT --days 365
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

See [docs/REGIME_DETECTION_MVP.md](../../docs/REGIME_DETECTION_MVP.md) for detailed information on regime detection algorithms and usage patterns.