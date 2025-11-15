# Configuration

> **Last Updated**: 2025-11-15  
> **Related Documentation**: [Development workflow](development.md), [Database](database.md)

The configuration system centralises access to environment settings so that every service (CLI commands, backtesting, live
engine, dashboards) reads values in a consistent order. The entry point is the `ConfigManager` defined in
`src/config/config_manager.py`.

## Provider chain

`ConfigManager` loads values from multiple providers and walks them in priority order:

1. `RailwayProvider` – reads deployment secrets when running on Railway.
2. `EnvVarProvider` – regular environment variables (exported in the shell, CI, Docker, etc.).
3. `DotEnvProvider` – keys defined inside a project-level `.env` file for local development.

Creating the manager without arguments automatically wires these providers. Each provider exposes `is_available()` so the
manager can skip layers that are not relevant for the current environment.

## Access patterns

Use the global helper to reuse the singleton instance:

```python
from src.config.config_manager import get_config

config = get_config()
database_url = config.get_required("DATABASE_URL")
log_level = config.get("LOG_LEVEL", default="INFO")
cache_ttl = config.get_int("CACHE_TTL_HOURS", default=24)
```

Typed helpers (`get_bool`, `get_int`, `get_float`, `get_list`) apply safe conversions and fall back to the default when a value
is missing or malformed. `get_all()` merges every provider (lower priority wins) which is useful for diagnostics.

To inspect all loaded values at runtime you can call:

```python
summary = config.get_config_summary()
print(summary["sources"])        # active providers
print(summary["total_keys"])      # number of resolved keys
```

## Feature flags

Feature toggles live outside static constants so experiments can be promoted without code changes. The helper functions live in
`src/config/feature_flags.py` and resolve values with the following precedence:

1. `FEATURE_<NAME>` emergency overrides (per-flag environment variables).
2. `FEATURE_FLAGS_OVERRIDES` JSON overrides defined per environment.
3. Repository defaults stored in `feature_flags.json`.

Current defaults (see `feature_flags.json`):

- `use_prediction_engine` (bool) – drives whether strategies call the prediction engine.
- `enable_regime_detection` (bool) – gates regime-aware overlays.
- `optimizer_canary_fraction` (string) – fraction of optimizer runs allowed to auto-apply.
- `optimizer_auto_apply` (bool) – controls whether successful optimizer suggestions are enacted automatically.

Usage:

```python
from src/config.feature_flags import is_enabled, get_flag

if is_enabled("use_prediction_engine", default=False):
    enable_prediction_mode()

canary_bucket = get_flag("optimizer_canary_fraction", default="0.0")
auto_apply = is_enabled("optimizer_auto_apply", default=False)
```

`resolve_all()` returns a merged dictionary that is useful when exposing diagnostics endpoints.

## Constants and overrides

Runtime defaults such as `DEFAULT_INITIAL_BALANCE`, `DEFAULT_CHECK_INTERVAL`, and risk tuning values live in
`src/config/constants.py`. Treat them as sensible defaults – short-term adjustments should be supplied via configuration
or strategy parameters instead of editing the constant file directly. For example, the backtest CLI allows overriding
initial balance and risk per trade via command-line flags.

## Local configuration workflow

1. Copy `.env.example` to `.env` and fill in secrets (API keys, database URL, webhook endpoints):
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```
2. Export sensitive overrides in the shell for automation: `export DATABASE_URL=postgresql://...`.
3. In a Python shell, call `get_config().get_all()` to review the resolved values and confirm the expected source order:
   ```python
   from src.config.config_manager import get_config
   config = get_config()
   print(config.get_all())
   ```

Because `ConfigManager` is thread-safe, long-running components such as the live trading engine can call `refresh()` to reload
providers after secrets rotate without restarting the process.
