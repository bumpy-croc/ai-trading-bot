# Regime Detection

Market regime detection and analysis for adaptive trading strategies.

## Overview

This module provides market regime detection capabilities to help strategies adapt to different market conditions (bull, bear, sideways).

## Features

- Market regime classification
- Volatility regime detection  
- Trend strength analysis
- Dynamic strategy adaptation

## Usage

```python
from regime import RegimeDetector

detector = RegimeDetector()
regime = detector.detect_current_regime(price_data)
```