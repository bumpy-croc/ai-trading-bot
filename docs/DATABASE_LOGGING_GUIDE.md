# Database Logging Guide

## Overview

The AI-Trading-Bot trading bot now includes comprehensive database logging for all trades, positions, and performance metrics. This enables better monitoring, analysis, and dashboard creation as outlined in the Week 1 go-live plan.

## Database Schema

The database includes the following tables:

### 1. **Trading Sessions** (`trading_sessions`)
- Tracks each trading session (live, paper, or backtest)
- Links all trades and metrics to specific sessions
- Records strategy configuration and performance summary

### 2. **Trades** (`trades`)
- Logs all completed trades with entry/exit prices, P&L, and reasons
- Includes strategy information and confidence scores
- Tracks both live and backtested trades

### 3. **Positions** (`positions`)
- Tracks currently active positions
- Updates with real-time unrealized P&L
- Includes risk management levels (stop loss, take profit)

### 4. **Account History** (`account_history`)
- Periodic snapshots of account state
- Tracks balance, equity, drawdown over time
- Useful for performance charts

### 5. **Performance Metrics** (`performance_metrics`)
- Aggregated metrics by period (daily, weekly, monthly)
- Includes win rate, Sharpe ratio, drawdown statistics
- Pre-calculated for efficient dashboard queries

### 6. **System Events** (`system_events`)
- Logs all system events, errors, and alerts
- Tracks strategy changes and model updates
- Useful for debugging and monitoring

### 7. **Strategy Executions** (`strategy_executions`)
- Detailed logs of each strategy decision
- Records indicator values, ML predictions, sentiment data
- Helps understand why trades were taken

## Configuration

### Database (PostgreSQL only)

The system connects to PostgreSQL via the `DATABASE_URL` environment variable. During tests, a PostgreSQL Testcontainers instance is started automatically.

### PostgreSQL Configuration

For production, use PostgreSQL by setting the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/ai-trading-bot"
```

Or pass it directly when initializing:

```python
from database.manager import DatabaseManager

db_manager = DatabaseManager("postgresql://username:password@localhost:5432/ai-trading-bot")
```

## Usage Examples

### Live Trading with Database Logging

```python
from live.trading_engine import LiveTradingEngine
from strategies.ml_basic import MlBasic
from core.data_providers.binance_provider import BinanceProvider

# Strategy and data provider setup
strategy = MlBasic()
data_provider = BinanceProvider()

# Create engine with database logging
engine = LiveTradingEngine(
    strategy=strategy,
    data_provider=data_provider,
    initial_balance=10000,
    enable_live_trading=False,  # Paper trading
    database_url=None  # Uses DATABASE_URL (PostgreSQL) if not provided
)

# Start trading (automatically logs to database)
engine.start("BTCUSDT", "1h")
```

### Backtesting with Database Logging

```python
from backtesting.engine import Backtester
from datetime import datetime, timedelta

# Create backtester with database logging
backtester = Backtester(
    strategy=strategy,
    data_provider=data_provider,
    initial_balance=10000,
    log_to_database=True  # Enable database logging
)

# Run backtest (automatically logs to database)
results = backtester.run(
    symbol="BTCUSDT",
    timeframe="1d",
    start=datetime.now() - timedelta(days=365),
    end=datetime.now()
)

print(f"Session ID: {results['session_id']}")
```

### Querying Performance Data

```python
from database.manager import DatabaseManager

db_manager = DatabaseManager()

# Get recent trades
trades = db_manager.get_recent_trades(limit=20)
for trade in trades:
    print(f"{trade['symbol']} - P&L: ${trade['pnl']:.2f} ({trade['pnl_percent']:.1f}%)")

# Get active positions
positions = db_manager.get_active_positions()
for pos in positions:
    print(f"{pos['symbol']} {pos['side']} - Unrealized P&L: ${pos['unrealized_pnl']:.2f}")

# Get performance metrics
metrics = db_manager.get_performance_metrics(period='daily')
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Total P&L: ${metrics['total_pnl']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
```

## Database Maintenance

### Cleanup Old Data

To prevent database bloat, periodically clean up old data:

```python
db_manager = DatabaseManager()
db_manager.cleanup_old_data(days_to_keep=90)  # Keep last 90 days
```

### Backup Strategy

For PostgreSQL:
```bash
pg_dump --dbname="$DATABASE_URL" > backup/ai-trading-bot_$(date +%Y%m%d).sql
```

## Viewing Data

### PostgreSQL Tools
- DBeaver (cross-platform)
- TablePlus (Mac/Windows/Linux)
- psql (CLI)

### Command Line
```bash
# Using psql with DATABASE_URL
psql "$DATABASE_URL"

# View tables
\dt

# Query trades
SELECT * FROM trades ORDER BY exit_time DESC LIMIT 10;

# View performance
SELECT * FROM performance_metrics WHERE period = 'daily' ORDER BY period_start DESC;
```

## Integration with Monitoring Dashboard

The database is designed to support the monitoring dashboard outlined in the go-live plan:

1. **Real-time Overview**: Query `positions` and `account_history` tables
2. **Performance Charts**: Use `account_history` for equity curves
3. **Trade History**: Query `trades` table with filters
4. **Risk Dashboard**: Calculate from `positions` and `account_history`
5. **System Status**: Monitor `system_events` table
6. **Strategy Analysis**: Use `strategy_executions` and `performance_metrics`

## Best Practices

1. **Session Management**: Always create a trading session at the start and end it properly
2. **Error Handling**: Database operations are wrapped in try-catch blocks
3. **Performance**: Indexes are created on commonly queried fields
4. **Data Integrity**: Use transactions for related operations
5. **Monitoring**: Check `system_events` table for errors and warnings

## Troubleshooting

### Common Issues

1. **"No module named 'sqlalchemy'"**
   ```bash
   pip install sqlalchemy psycopg2-binary
   ```

2. Ensure `DATABASE_URL` is correctly set and reachable from the process environment.

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Next Steps

With database logging implemented (Week 1 ✅), you can now:

1. **Week 2**: Build a monitoring dashboard using the logged data
2. **Week 3**: Set up alerts based on database triggers
3. **Week 4**: Implement automated reporting from historical data

The database provides a solid foundation for all monitoring and analysis needs as you move towards production deployment.
