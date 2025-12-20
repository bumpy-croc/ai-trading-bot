# Database

> **Last Updated**: 2025-12-20  
> **Related Documentation**: [Configuration](configuration.md), [Development workflow](development.md#railway-deployment-quick-start)

All persistent state flows through the PostgreSQL manager in `src/database/manager.py`. It enforces a PostgreSQL URL (SQLite is
only allowed for unit tests) and creates the full schema on startup using SQLAlchemy models from `src/database/models.py`.

## Responsibilities

- Connection pooling (`QueuePool`) with health checks and secure defaults (SSL, timeouts).
- Automatic table creation and Alembic-aware migrations.
- Logging trades, positions, orders, partial fills, account history, and optimisation cycles.
- Persisting strategy execution events and system diagnostics used by dashboards.
- Optional prediction cache storage for the ML subsystem.

Key tables include `trades`, `positions`, `orders`, `account_history`, `performance_metrics`, `system_events`,
`trading_sessions`, `optimization_cycles`, and `prediction_cache`. The enums (`TradeSource`, `OrderStatus`, etc.) live alongside
these models to keep the schema consistent between backtesting and live trading.

## Configuration and setup

1. Provide a PostgreSQL URL via `DATABASE_URL`. Example for Docker Compose:
   ```bash
   docker compose up -d postgres
   export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
   ```
2. Run migrations (the CLI wraps Alembic):
   ```bash
   atb db migrate
   ```
3. Verify connectivity and schema alignment:
   ```bash
   atb db verify
   ```

The manager falls back to creating tables automatically when connecting to an empty database. In CI the same code path allows
`sqlite:///:memory:` during fast unit tests while integration tests still use PostgreSQL.

## CLI tooling

`cli/commands/db.py` exposes operational helpers:

- `atb db verify` – checks connection health, expected tables, row counts, and type mismatches.
- `atb db migrate` – runs Alembic migrations using the repository’s migration scripts.
- `atb db backup` – dumps data using `pg_dump` with safe defaults (see Backups & recovery for scheduling tips).
- `atb db reset-railway` / `setup-railway` – remote administration helpers for Railway deployments.
- `atb db nuke` – drops all objects (including enums) and recreates the schema; intended for local resets only.

## Backups & recovery

- Create compressed dumps with retention management:

    ```bash
    atb db backup --env production --backup-dir ./backups --retention 7
    ```

  Dumps live under `backups/<db name>/<YYYY>/<MM>/<DD>/backup-<timestamp>.dump`. Schedule the command (Railway jobs or cron) hourly
  to meet the current 15 minute RPO when paired with heartbeat alerts.
- Restore by running `pg_restore -d <target_db> --clean --create backup-<ts>.dump` and applying `alembic upgrade head` to reach
  the latest revision before repointing the app.
- Test restores quarterly by loading a random dump into a sandbox database and running the integration test suite; this keeps
  recovery times well under the 60 minute RTO target.

## Railway deployments

- `atb db setup-railway --verify` checks that `DATABASE_URL` points to PostgreSQL, opens a session, and seeds a verification trade
  so monitoring dashboards see activity.
- `atb db reset-railway staging --yes` (or `production`) automates schema resets when you need a clean environment. Always take a
  backup first.
- Configure project variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `TRADING_MODE`, `INITIAL_BALANCE`) through the Railway CLI
  or dashboard; `ConfigManager` will pick them up automatically at runtime.

## Programmatic access

Most services interact with the database through the context manager API:

```python
from src.database.manager import DatabaseManager
from src.database.models import Trade, TradeSource

manager = DatabaseManager()
with manager.get_session() as session:
    latest_trade = session.query(Trade).filter(Trade.source == TradeSource.LIVE).order_by(Trade.id.desc()).first()
    print(latest_trade)
```

Use `DatabaseManager.create_trading_session` when logging new engine runs so dashboards can attribute metrics to a specific
session ID.
