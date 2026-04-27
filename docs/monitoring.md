# Monitoring & observability

> **Last Updated**: 2026-04-27
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

### Monitoring dashboard layout (V2)

The monitoring dashboard ships a chart-led V2 layout. Front-end stack: **React 18 (UMD) + Babel-standalone** loaded from
CDN with SRI hashes pinned, plus `socket.io-client` for live metric updates. Themes (dark / light) toggle from the topbar
and persist to `localStorage` (`tbm-theme`).

Tabs in the left rail:

| Tab    | Content                                                                        |
|--------|--------------------------------------------------------------------------------|
| Dash   | KPI strip + hero equity chart with overlay toggles + positions strip + inspector |
| Pos    | Full open-positions table (symbol, side, size, entry, current, P&L, trail SL, BE, MFE, MAE) |
| Strat  | Bot identity (strategy / symbols / timeframe / mode / status / uptime) + signal proxies |
| Trades | Filterable trade history with win/loss summary KPIs                            |
| Risk   | Dynamic risk factors, exposure caps (uses `max_open_positions` from session config), risk metrics |
| Logs   | Position open / trade exit history sourced from real timestamps only           |

The Dash tab's right inspector swaps based on selection: a position card → position inspector, a chart trade marker →
trade inspector. The chart's "+ benchmark" overlay is labelled "(approx)" — it is an indicative guide derived from the
bot's own slope, not real per-symbol buy-and-hold equity. The Logs tab is intentionally narrow in scope (open / exit
events only) so timestamps are always real; system / connection events are not yet logged here.

> **Babel-standalone caveat:** the dashboard transpiles its JSX in the browser. This is fine for an internal monitoring
> tool but adds ~2s of CPU on first paint and pulls ~3MB of Babel from the CDN. Pre-building the bundle with `esbuild`
> is a tracked follow-up; the current setup keeps the edit loop simple.

### `/api/dashboard/state` (bundled)

```http
GET /api/dashboard/state?trades_limit=50
```

Returns a single payload with `bot`, `metrics`, `positions`, `trades`, and `server_time`. Designed to make the V2 dashboard's
first paint a single round-trip. `trades_limit` is clamped to `1..500`. The endpoint reuses the `api_connection_status`
already computed by `_collect_metrics()` to avoid a second Binance round-trip in the same request.

The `bot` block is built by `_get_bot_meta()` which prefers the most recent **running** session (`end_time IS NULL`) and
exposes:

- `name`, `symbols[]`, `timeframe`, `mode`, `status`
- `initial_balance` — from the session row; the JS adapter renders `—` when missing rather than fabricating a baseline
- `max_open_positions` — parsed from `strategy_config` (top-level / `risk` / `risk_manager` / `position_management`)
- `connected` — derived from the metrics' API connection probe
- `uptime_seconds`, `last_update`

If `/api/dashboard/state` is unreachable the JS falls back to per-resource fetches: `/api/metrics`, `/api/positions`,
`/api/trades?limit=50`, `/api/system/status`.

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
