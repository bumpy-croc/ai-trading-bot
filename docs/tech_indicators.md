# Technical Indicator Toolkit

Shared indicator math now lives under `src/tech/indicators/core.py`. These helpers
power prediction features, risk management, dashboards, and trading engines.
All functions accept pandas Series/DataFrames and return vectorized results so
callers can chain them without side effects.

## Available Helpers

- **Trend**: `calculate_moving_averages`, `calculate_ema`, and the derived trend
  strength helpers embedded in the feature extractor.
- **Momentum**: `calculate_rsi`, `calculate_macd` (with fast/slow/signal spans and
  histogram output).
- **Volatility**: `calculate_atr`, `calculate_bollinger_bands`, and
  `detect_market_regime` (volatility- and trend-aware regime tagging).
- **Support/Resistance**: `calculate_support_resistance` returns the latest
  swing levels over a configurable lookback.

Additions should follow the same pattern: accept `pd.DataFrame`/`Series`, avoid
mutation, and document column expectations.

## Usage Example

    from src.tech.indicators.core import (
        calculate_atr,
        calculate_bollinger_bands,
        calculate_moving_averages,
        calculate_rsi,
    )

    df = calculate_moving_averages(raw_df, periods=[20, 50])
    df["rsi"] = calculate_rsi(df, period=14)
    df = calculate_atr(df, period=14)
    df = calculate_bollinger_bands(df, period=20, std_dev=2.0)
    print(df[["close", "ma_20", "ma_50", "rsi", "atr", "bb_upper", "bb_lower"]].tail())

## Tests

Run `pytest tests/unit/indicators -v` to exercise the math helpers. Indicator
extraction helpers in `src.tech.adapters.row_extractors` are covered by
`tests/unit/trading/test_shared_indicators.py` and
`tests/unit/backtesting/test_utils.py`.
