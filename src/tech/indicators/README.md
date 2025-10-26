# src/tech/indicators

Pure indicator math lives here. Each function should:

- Accept pandas `DataFrame` or `Series` inputs using standard OHLCV column
  names (`open`, `high`, `low`, `close`, `volume`).
- Return new pandas objects with deterministic columns and no I/O side effects.
- Avoid referencing prediction-specific modules or caches so the same function
  can be used by trading engines, dashboards, and tests.

Organize helpers by domain (moving averages, oscillators, volatility, trend,
support/resistance). Keep implementations vectorized and well-documented. When
adding a new function, update `src/tech/indicators/__init__.py` to export it and
extend the relevant docs (`docs/tech_indicators.md`).
