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

### Mobile layout

The dashboard reflows to a phone-friendly layout below the **768px** viewport breakpoint, driven by a `useIsMobile()`
hook backed by `window.matchMedia('(max-width: 768px)')` (re-evaluated on resize, so a desktop user dragging the
window narrow sees the mobile UI immediately). Specifically:

- The left rail is replaced by a fixed **bottom tab bar** with the same six sections.
- The topbar shrinks to a sticky compact strip with bot name, mode/status pills, live indicator, and theme toggle.
- The Dash view stacks: hero equity card → 3-column KPI grid → equity chart with overlay chips → positions list →
  inline inspector (replaces the desktop side panel).
- The Pos and Trades views render as full-width cards with stacked metadata instead of tables.
- The Strat and Risk views drop their two-column grids to single-column via the `.tbm-2col-grid` and
  `.tbm-metrics-grid` CSS classes (3-up on mobile for the risk-metrics row).
- iOS safe-area insets are respected via `viewport-fit=cover` plus
  `padding: env(safe-area-inset-top|bottom)` on the topbar and bottom tab bar.

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

## Write-gap watchdog (dead-man's-switch)

A dead engine cannot report its own death (the 2026-05-19 outage ran 12.8 days silently), so
liveness is judged from outside the process. `.github/workflows/heartbeat-monitor.yml` runs
`python -m cli.commands.live_watchdog --env all` every 30 minutes (also `atb live-control
watchdog --env <production|staging|development|all>`). It reads each environment's database and
treats either stream going quiet for more than `--threshold-minutes` (default
`$HEARTBEAT_THRESHOLD_MINUTES`, then 120) as a gap:

| Stream | Written by | Cadence |
| --- | --- | --- |
| `account_history` | `LiveAccountMonitor` snapshot | hourly |
| `system_events` loop-state row (`ENTRIES_ENABLED` or any `*_LATCHED` / `CLOSED_CANDLE_GATE_STALLED`) | `LatchedConditionMonitor` | hourly |

`reconciliation_audit_events` is deliberately **not** a liveness stream: it is event-driven, so a
quiet audit table is the healthy state.

A gap follows the same latched-condition pattern as the in-process monitor: a critical
`HEARTBEAT_GAP` row plus a webhook page (`$ALERT_WEBHOOK_URL`) immediately, re-paged at 1h, 4h
and 12h and then daily (`src/infrastructure/paging_schedule.py`, shared with
`LatchedConditionMonitor`), pages past the first day prefixed `STILL DOWN — DAY N`, and a
`HEARTBEAT_GAP_CLEARED` row once writes resume. The job is stateless, so the latch is rebuilt from
its own rows; an undelivered page is retried on the next run. The command exits non-zero while a
gap is open or a check cannot complete, so the CI failure notification is a second channel.
Judgement uses the newest rows globally rather than session state: a crash leaves the session
`is_active` and a clean stop clears it, but in both cases the writes stop. A deliberate stop
therefore also raises the gap until the engine restarts.

The only writes are the `HEARTBEAT_GAP` / `HEARTBEAT_GAP_CLEARED` rows in `system_events`.

## Failure observability

- `DatabaseManager.log_audit_event` never raises: the correction an audit row documents (a
  balance heal, a position close) must still complete. A failed write is logged at ERROR and
  reported to the engine, which records an `AUDIT_WRITE_FAILED` `system_events` row
  (component `reconciler`; one per 5 minutes with the folded-in count; a lost `CRITICAL` audit row
  also pages). The event is delivered on a daemon thread because the write usually fails while a
  reconciler lock is held.
- A database outage the live loop rides out pages `DB_UNREACHABLE` once after 5 minutes
  (webhook only: the row cannot be written while the database is down), enters close-only after
  30, and writes `DB_OUTAGE_RECOVERED` with the duration when the database returns, so every
  outage leaves a durable row.

## Operational tips

- Ship JSON logs to your preferred aggregator (Loki, ELK, Datadog) and index on `event_type`, `strategy`, `symbol`, and
  `session_id`.
- Set alerts for repeated order rejections, stale data events, and database connection failures.
- Use `atb tests heartbeat` to emit a `SystemEvent` on demand and confirm the database pipeline is healthy.
