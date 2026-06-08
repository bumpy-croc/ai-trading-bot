# Configuration

> **Last Updated**: 2025-11-10  
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
# For quick experiments, prefer a default so the snippet is runnable without
# exporting environment variables. Use `get_required` in production paths.
database_url = config.get("DATABASE_URL", default="postgresql://...")  # set your real URL
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

Usage:

```python
from src.config.feature_flags import is_enabled, get_flag

if is_enabled("use_prediction_engine", default=False):
    enable_prediction_mode()

bucket = get_flag("optimizer_bucket", default="control")
```

`resolve_all()` returns a merged dictionary that is useful when exposing diagnostics endpoints.

### Feature flag reference

Every feature flag in the project. Each resolves via the precedence above; the
`FEATURE_<UPPER_SNAKE_KEY>` env var is the per-flag override (**setting one in Railway triggers a
redeploy/restart**). Boolean flags accept `true/on/1/yes/enabled` (and `false/off/0/no/disabled`).

**Live-engine flags** — change live trading / runtime behaviour:

| Env var | Flag key | Default | Values | What it does |
|---|---|---|---|---|
| `FEATURE_WS_USER_HARD_RECONNECT` | `ws_user_hard_reconnect` | `off` | boolean | **#723 Phase 2 (experimental).** When the margin user-data WebSocket is stuck in `REST_DEGRADED`, the throttled recovery probe performs a full **hard reconnect** — teardown + fresh `AsyncClient`/`ws_api` (`hard_reconnect_user()`) — instead of an in-place reconnect, to attempt real recovery from the #616/#723 "reconnect doesn't restore events" failure. **High-risk WS/event-loop subsystem; ships OFF.** Each hard reconnect emits 2 benign asyncio teardown artifacts (`Error stopping client`, `Task was destroyed but it is pending!`). Recovery is still gated on a real user event (#717); REST polling covers order/balance while degraded, so leaving it OFF is safe. |
| `FEATURE_ORPHANED_BORROW_SWEEP_MODE` | `orphaned_borrow_sweep_mode` | `off` | `off` / `dry_run` / `active` | **#702/#703.** Periodic reconciler sweep that repays an orphaned margin borrow (a base-asset loan with no tracked position). `off` = no-op; `dry_run` = detect + log only, no money moved; `active` = **repay** (a money-mover — requires the #705 base-asset exchange-mutation lock and serialises against entry/exit). Deployed as `dry_run` during validation, then `active`. |
| `FEATURE_USE_PREDICTION_ENGINE` | `use_prediction_engine` | `true` | boolean | Gates whether the strategy uses the ML prediction engine for signal generation (`src/strategies/components/strategy.py`). Off ⇒ the strategy falls back to its non-prediction signal path. Also referenced by the experiment runner/promotion. |
| `FEATURE_ENABLE_REGIME_DETECTION` | `enable_regime_detection` | `false` | boolean | Enables market-regime detection in the live engine (`src/engines/live/trading_engine.py`, read directly via `os.getenv`). |

**Experiment / optimizer flags** — used by the A/B experiment + parameter-optimizer promotion subsystem (`src/experiments/`); repo defaults live in `feature_flags.json`:

| Env var | Flag key | Default | Values | What it does |
|---|---|---|---|---|
| `FEATURE_EXPERIMENT_BUCKET` | `experiment_bucket` | _(caller default)_ | string | Assigns the A/B experiment bucket (e.g. `control`, `beta`) for experiment routing. |
| `FEATURE_OPTIMIZER_CANARY_FRACTION` | `optimizer_canary_fraction` | `"0.1"` | string (float) | Fraction reserved for the optimizer canary when promoting tuned parameters. |
| `FEATURE_OPTIMIZER_AUTO_APPLY` | `optimizer_auto_apply` | `false` | boolean | Whether the optimizer auto-applies promoted parameters (vs. requiring manual promotion). |

> ⚠️ Flipping a live-engine flag is an outward-facing action: it restarts the bot, and for
> `active`/`true` money- or stability-affecting flags it changes live behaviour. After the restart,
> validate (position re-adoption, kline alive, the flag's intended effect) and keep a rollback (flip
> back) ready. See `.claude/LESSONS.md` §3.
>
> _Not flags:_ `nonexistent_flag` (test fixture) and `optimizer_bucket` (docstring example) are not
> real flags; `FEATURE_CACHE_TTL` / `FEATURE_ENGINEERING_CONFIG` / `FEATURE_ADDITION` belong to the
> ML **feature-engineering** config, a separate system despite the `FEATURE_` prefix.

## Constants and overrides

Runtime defaults such as `DEFAULT_INITIAL_BALANCE`, `DEFAULT_CHECK_INTERVAL`, and risk tuning values live in
`src/config/constants.py`. Treat them as sensible defaults – short-term adjustments should be supplied via configuration
or strategy parameters instead of editing the constant file directly. For example, the backtest CLI allows overriding
initial balance and risk per trade via command-line flags.

## Execution fill policy

`EXECUTION_FILL_POLICY` controls simulated fill behavior in backtests and paper/live
simulation. The default is `ohlc_conservative`.

- `ohlc_conservative`: limit orders fill at their limit price when a bar crosses the
  level, no price improvement without quotes, and stop orders behave like stop-market
  orders (taker liquidity).

Unknown values fall back to the default. Live trading still reconciles fills with the
exchange when available.

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
