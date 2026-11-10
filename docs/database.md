# Database

> **Last Updated**: 2025-11-10  
> **Related Documentation**: [Configuration](configuration.md), [Development workflow](development.md#railway-deployment-quick-start)

All persistent state flows through the PostgreSQL manager in `src/database/manager.py`. It enforces a PostgreSQL URL (SQLite is
only allowed for unit tests) and creates the full schema on startup using SQLAlchemy models from `src/database/models.py`.

## Responsibilities

- Connection pooling (`QueuePool`) with health checks and secure defaults (SSL, timeouts).
- Automatic table creation and Alembic-aware migrations.
- Logging trades, positions, orders, partial fills, and account history.
- Persisting strategy execution events and system diagnostics used by dashboards.
- Optional prediction cache storage for the ML subsystem.

Key tables include `trades`, `positions`, `orders`, `account_history`, `performance_metrics`, `system_events`,
`trading_sessions`, and `prediction_cache`. The enums (`TradeSource`, `OrderStatus`, etc.) live alongside
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

## Migrations

`migrations/versions/` is the only migration directory and holds one linear chain (`alembic heads` returns a single head).
Retired migrations from the pre-baseline schema were deleted rather than kept in the tree; recover them from git history if
needed (`git log --diff-filter=D --name-only -- migrations/`).

Workflow for a schema change:

1. Change `src/database/models.py`.
2. Point `DATABASE_URL` at a throwaway Postgres already upgraded to head, then run
   `alembic revision --autogenerate -m "<short description>"`.
3. Treat the generated file as a draft. Review it and hand-edit: autogenerate misses enum changes, views, server defaults,
   data backfills and `CREATE INDEX CONCURRENTLY`, and it can emit drops for objects it does not know about.
4. Hand-write the migration from scratch for data migrations and anything autogenerate cannot express.
5. Name the revision `NNNN_short_name`, keep the id at 32 characters or fewer (the `alembic_version` column is `VARCHAR(32)`),
   and set `down_revision` to the current head.
6. Verify on a fresh database: `alembic upgrade head`, then `atb db verify`.

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
