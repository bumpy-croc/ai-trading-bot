# src/tech/features

Feature builders combine the primitives in `src/tech/indicators` into
higher-level datasets for ML models or analytics. Typical responsibilities
include normalization, derived volatility/trend metrics, and schema validation.

Guidelines:

- Depend only on `src.tech.indicators` and shared config/constants, never on
  prediction-engine classes such as the registry or cache.
- Keep extractors stateless and deterministic, returning pandas DataFrames so
  downstream components can convert to numpy/torch as needed.
- Document any new extractor in this README and in `docs/prediction.md` so
  model authors know how to enable it via the feature pipeline.

The `TechnicalFeatureExtractor` lives here and is re-exported for compatibility
from `src/prediction/features/technical.py`.
