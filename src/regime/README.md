# Regime Detection

Market regime detection utilities used across the trading system.

## Overview

The `src.regime` package provides:

- `RegimeDetector` – core regime annotations based on price action and volatility.
- `EnhancedRegimeDetector` – strategy-facing wrapper with state tracking and
  stability metrics.
- Calibration helpers to tune detector parameters against labelled data.
- Evaluation and plotting utilities for quantifying detection accuracy.

## Basic usage

```python
from src.regime import RegimeDetector

# df must contain open/high/low/close/volume columns indexed by time
annotated = RegimeDetector().annotate(df)
print(annotated[["trend_label", "vol_label", "regime_label", "regime_confidence"]].tail())
```

## Enhanced detector for strategies

```python
from src.regime import EnhancedRegimeDetector

detector = EnhancedRegimeDetector()
regime_context = detector.detect_regime(annotated, -1)
print(regime_context.get_regime_label(), regime_context.confidence)
```

The enhanced detector maintains regime history, transition tracking and exposes
helper methods for recent statistics.

## Calibrating and evaluating accuracy

```python
from src.regime import calibrate_regime_detector

# Provide labelled columns (e.g. produced from expert annotations or research data)
calibration = calibrate_regime_detector(
    annotated_df,
    target_trend_col="target_trend",
    target_vol_col="target_vol",
)

print(calibration.metrics)  # overall, trend and volatility accuracy
```

To evaluate a specific configuration and visualise accuracy over time:

```python
from src.regime import evaluate_regime_accuracy, plot_regime_accuracy

metrics, evaluation_frame = evaluate_regime_accuracy(
    annotated_df,
    target_trend_col="target_trend",
    target_vol_col="target_vol",
)

fig = plot_regime_accuracy(evaluation_frame)
fig.savefig("regime_accuracy.png")
```

These helpers make it straightforward to quantify how well a configuration
matches labelled data and to reason about performance drift.
