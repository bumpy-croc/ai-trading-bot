# Database

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
- `atb db backup --output backup.sql.gz` – dumps data using `pg_dump` with safe defaults.
- `atb db reset-railway` / `setup-railway` – remote administration helpers for Railway deployments.
- `atb db nuke` – drops all objects (including enums) and recreates the schema; intended for local resets only.

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
