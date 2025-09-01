# Logging Guide

This project uses centralized, structured logging with optional JSON output for production observability.

## Quick start

- Always initialize logging via `src.utils.logging_config.configure_logging()` in entrypoints.
- Default level is controlled by `LOG_LEVEL` (INFO by default).
- JSON output:
  - Defaults to ON in production-like environments (Railway detected or ENV/APP_ENV=production).
  - Force via `LOG_JSON=1` or disable via `LOG_JSON=0`.

## Context and structure

- Context fields are injected into every log record via contextvars.
- Core fields:
  - `component`, `strategy`, `symbol`, `timeframe`, `session_id`
  - `event_type` when using structured event helpers
- Use helpers in `src/utils/logging_events.py`:
  - `log_engine_event`, `log_decision_event`, `log_order_event`, `log_risk_event`, `log_data_event`, `log_db_event`

## Namespacing

- Logger names are auto-prefixed with `atb.` for consistent filtering.
- No need to rename existing loggers; the prefix filter handles it.

## Per-logger noise control

- Chatty libraries are set to WARN by default:
  - `sqlalchemy.engine`, `urllib3`, `binance`, `ccxt`
- Override via env:
  - `LOG_SQLALCHEMY_LEVEL`, `LOG_URLLIB3_LEVEL`, `LOG_BINANCE_LEVEL`, `LOG_CCXT_LEVEL`

## Redaction

- Sensitive keys (api_key, secret, token, password, auth, bearer, session) are redacted from messages and args.

## Environment controls

- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `LOG_JSON`: 0/1 to disable/enable JSON explicitly
- Production auto-detection for JSON default:
  - Railway env vars present, or `ENV=production` / `APP_ENV=production`

## Examples

```python
from src.utils.logging_config import configure_logging
from src.utils.logging_context import set_context
from src.utils.logging_events import log_engine_event

configure_logging()
set_context(component="live_engine", strategy="MlBasic", symbol="BTCUSDT", timeframe="1h")
log_engine_event("engine_start", initial_balance=10000, check_interval=60, mode="paper")
```

## Operations (observability)

- Prefer JSON logs in production; ship stdout to your log platform (e.g., Loki, ELK).
- Index useful labels: `event_type`, `component`, `strategy`, `symbol`, `session_id`.
- Suggested alerts:
  - Repeated order rejections over rolling window
  - Data staleness exceeding threshold
  - DB connection failures or retries spiking
  - Strategy hot-swap failures

## Tests

- Tests suppress logs; enable verbose logs ad-hoc if debugging.
- Avoid logging secrets in tests; redaction runs regardless.