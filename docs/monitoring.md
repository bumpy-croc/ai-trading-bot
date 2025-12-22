# Monitoring & observability

> **Last Updated**: 2025-12-14  
> **Related Documentation**: [Live trading](live_trading.md), [Database](database.md)

Instrumentation is delivered through structured logging, database events, and interactive dashboards. The goal is to provide the
same visibility during backtests, paper trading, and live execution.

## Logging configuration

- Call `src.infrastructure.logging.config.configure_logging()` in every entrypoint (already done in CLI modules).
- Set `LOG_LEVEL` to control verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- Enable JSON output with `LOG_JSON=1` (auto-enabled on Railway or when `ENV=production`). Disable with `LOG_JSON=0`.
- Per-library overrides: `LOG_SQLALCHEMY_LEVEL`, `LOG_URLLIB3_LEVEL`, `LOG_BINANCE_LEVEL`, `LOG_CCXT_LEVEL`.
- Context helpers in `src/infrastructure/logging/context.py` inject `component`, `strategy`, `symbol`, and `session_id` so that logs can be
  filtered downstream.
- Structured event emitters (`log_engine_event`, `log_order_event`, `log_risk_event`, `log_data_event`) live in
  `src/infrastructure/logging/events.py` and are used by both engines.

Sensitive fields (API keys, secrets, tokens) are redacted automatically before logs hit stdout.

## Dashboards

Dashboards are implemented under `src/dashboards` using Flask + Socket.IO and discovered dynamically:

```bash
atb dashboards list
atb dashboards run monitoring --port 8000
atb dashboards run backtesting --port 8001
```

Available dashboards:

- **monitoring** – live trading metrics, open positions, balance history, and recent events.
- **backtesting** – visualise strategy results saved by the backtester (requires DB logging or CSV exports).
- **market_prediction** – inspect ML prediction confidence and feature diagnostics.

Each dashboard accepts optional overrides (`--db-url`, `--update-interval`, `--logs-dir`) which are forwarded only if the
underlying dashboard supports them.

## Health endpoints

`PORT=9000 atb live-health -- ml_basic --paper-trading` runs the trading engine with an HTTP server exposing `/health` and
`/status`. Set either the `PORT` or `HEALTH_CHECK_PORT` environment variable (default: 8000) to pick the HTTP port. The status payload
checks configuration providers, database connectivity, and Binance API reachability so you can wire it into uptime monitors or
Kubernetes liveness probes.

## Operational tips

- Ship JSON logs to your preferred aggregator (Loki, ELK, Datadog) and index on `event_type`, `strategy`, `symbol`, and
  `session_id`.
- Set alerts for repeated order rejections, stale data events, and database connection failures.
- Use `atb tests heartbeat` to emit a `SystemEvent` on demand and confirm the database pipeline is healthy.
