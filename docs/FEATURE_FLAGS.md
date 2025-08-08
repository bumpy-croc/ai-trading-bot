## Feature Flags: Usage and Promotion

This project uses a minimal, environment-friendly feature flag system to gate features without releasing them to production by default.

### Sources and Precedence
Highest to lowest precedence:
1. `FEATURE_<UPPER_SNAKE_KEY>` – single-flag emergency override (e.g., `FEATURE_USE_PREDICTION_ENGINE=true`).
2. `FEATURE_FLAGS_OVERRIDES` – JSON overrides set per environment (only diffs).
3. `feature_flags.json` – Git-tracked defaults at the project root (same across environments).
4. Code-supplied default when calling the helper API.

Flags can be boolean or string (for limited multi-choice scenarios). Constants remain in `src/config/constants.py` and are not environment-overridable.

### Defaults file (git-tracked)
`feature_flags.json` at repo root:

```json
{
  "use_prediction_engine": false
}
```

### Environment overrides (Railway)
Set once per environment:

```json
// FEATURE_FLAGS_OVERRIDES (develop)
{ "use_prediction_engine": true }

// FEATURE_FLAGS_OVERRIDES (staging)
{ "use_prediction_engine": true }

// FEATURE_FLAGS_OVERRIDES (main)
{}
```

Emergency per-flag override:
- `FEATURE_USE_PREDICTION_ENGINE=false`

### Helper API
Use from `src/config/feature_flags.py`:

```python
from src.config.feature_flags import is_enabled, get_flag

if is_enabled("use_prediction_engine", default=False):
    ...

bucket = get_flag("experiment_bucket", default="control")  # string flag example
```

### Current initial flag
- `use_prediction_engine` (bool) – gates usage of prediction engine in strategies like `MlBasic`. Default off in repo, on in develop/staging via overrides.

### Promotion to Stable (removing a flag)
When a feature is ready for production by default:
1. Make the feature code unconditional (remove the `is_enabled(...)` gate).
2. Remove the key from `feature_flags.json`.
3. Remove the flag from all environment overrides and any `FEATURE_<KEY>` variables.
4. Update tests to assert the new default path. Optionally add a test to detect references to removed flags.
5. Document in release notes that the feature is now default and the flag is removed.

Optional soft landing: keep a no-op reference for one release that logs a deprecation warning, then fully remove in the next release.

### Notes
- Use flags only for enabling/disabling features by environment. Do not use flags for core operational constants.
- Keep `FEATURE_FLAGS_OVERRIDES` small: only include diffs from `feature_flags.json`.

