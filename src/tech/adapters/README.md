# src/tech/adapters

Adapters translate indicator/feature values into shapes that engines, logs, and
UIs expect. Examples include extracting the latest indicator snapshot from a
DataFrame row or serializing sentiment metrics.

Guidelines:

- Accept pandas inputs and return plain Python dictionaries or lightweight
  dataclasses that are easy to JSON-encode.
- Keep column lists centralized here so backtesting and live trading share the
  same behavior instead of copy/pasting helpers.
- Avoid logging or I/O inside adapters; return the data and let callers decide
  how to persist it.

The `row_extractors.py` module exposes `extract_indicators`,
`extract_sentiment_data`, and `extract_ml_predictions`, which replace the legacy
helpers that used to live in `src/trading/shared` and the bespoke versions inside
`src/backtesting/utils.py`. Engines and dashboards now import from this package
so indicator snapshots stay consistent everywhere.
