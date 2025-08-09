# Database (PostgreSQL)

SQLAlchemy models and the PostgreSQL-only `DatabaseManager` used by backtesting, live engine, and monitoring.

## Modules
- `models.py`: ORM models (trading_sessions, trades, positions, account_history, performance_metrics, system_events, strategy_executions)
- `manager.py`: Connection pooling, table creation, query helpers, and domain methods

## Usage
```python
from database.manager import DatabaseManager

db = DatabaseManager()  # uses DATABASE_URL
print(db.get_database_info())
trades = db.get_recent_trades(limit=20)
```