# Database Manager Web UI

A lightweight Flask-Admin panel for viewing and managing the trading bot PostgreSQL database.

## Features
- **Auto-generated CRUD** for SQLAlchemy models (trades, positions, strategy executions, etc.)
- **Search and filters** - Query and filter database records
- **Detail/edit views** - View and modify individual records
- **Health endpoints** - `/health` for status checks, `/db_info` for connection info
- **Schema sync** - `/migrate` endpoint for running database migrations
- **User authentication** - Basic authentication with configurable credentials

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements-server.txt

# Set environment variables
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
export DB_MANAGER_SECRET_KEY=your-secret-key
export DB_MANAGER_ADMIN_USER=admin
export DB_MANAGER_ADMIN_PASS=your-password

# Run the admin UI
python src/database/admin_ui/app.py

# Open browser
# Navigate to http://localhost:8000/admin
```

### Schema Sync
Manually trigger database migrations:
```bash
curl http://localhost:8000/migrate
```

## Railway Deployment

Set these environment variables in Railway:
- `DATABASE_URL` - PostgreSQL connection string (automatically provided by Railway)
- `DB_MANAGER_SECRET_KEY` - Secret key for Flask session encryption
- `DB_MANAGER_ADMIN_USER` - Admin username (optional, defaults to 'admin')
- `DB_MANAGER_ADMIN_PASS` - Admin password (required for production)

Start command:
```bash
python src/database/admin_ui/app.py
```

The admin UI will be accessible at your Railway deployment URL under `/admin`.

## Available Models

The admin UI provides CRUD access to:
- **trades** - Completed trade records
- **positions** - Active and historical positions
- **orders** - Order history and status
- **account_history** - Account balance history
- **performance_metrics** - Performance snapshots
- **system_events** - System logs and events
- **trading_sessions** - Trading session metadata
- **optimization_cycles** - Optimization history

## Customization

### Custom Model Views
Extend `CustomModelView` in `app.py` to customize model display:

```python
class CustomTradeView(CustomModelView):
    column_list = ['id', 'symbol', 'side', 'entry_price', 'exit_price', 'pnl']
    column_filters = ['symbol', 'side']
    form_excluded_columns = ['id', 'created_at']
```

### Adding Routes
Add custom routes using the shared `db_manager` and `db_session`:

```python
@app.route('/custom-report')
def custom_report():
    with db_session() as session:
        trades = session.query(Trade).all()
        return render_template('report.html', trades=trades)
```

## Security

- Always set a strong `DB_MANAGER_SECRET_KEY` in production
- Use environment variables for all credentials
- Consider restricting access via firewall or VPN
- Enable HTTPS in production (Railway provides this automatically)

## See Also

- [Database documentation](../../docs/database.md) - Database schema and operations
- [Monitoring dashboard](../dashboards/monitoring/README.md) - Real-time trading metrics
