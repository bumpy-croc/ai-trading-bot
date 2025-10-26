# src/tech

This package is the single home for technical-analysis primitives that are
shared by prediction, risk, trading, and monitoring code. It is split into
three layers:

1. `src/tech/indicators` contains pure mathematical helpers (moving averages,
   oscillators, volatility measures, support/resistance utilities). These
   functions must be deterministic, accept pandas objects, and avoid side
   effects so they can be reused everywhere.
2. `src/tech/features` provides higher-level feature builders that combine the
   core indicators into ML-ready tensors or enriched DataFrames. Keep these
   modules free of prediction-engine specifics such as caching or registry
   lookups so they remain portable.
3. `src/tech/adapters` houses wrappers that expose indicator/feature values to
   other systems (for example, serializing the latest indicator row for trading
   engines or dashboards).

When adding new indicators or derived features, place the raw math in
`src/tech/indicators`, shareable feature builders in `src/tech/features`, and
interface-specific extraction helpers in `src/tech/adapters`. Each subdirectory
has its own README with extension guidelines. This structure follows
`docs/execplans/indicator_refactor_plan.md`.
