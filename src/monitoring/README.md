# Trading Bot Monitoring Dashboard

Real-time dashboard for performance, risk, system health, and trading activity.

## Key features
- WebSocket updates (default: hourly)
- Configurable metrics and priorities
- REST API: balance, history, status

Note: default refresh interval is 3600 seconds. Override with `--update-interval`.

## Quick start
```bash
# Install dependencies
make deps-server  # or make deps for full development setup

# Start monitoring dashboard
atb dashboards run monitoring --port 8000
# http://localhost:8000
```

Or run directly:
```bash
# Check available dashboards
atb dashboards list

# Run with custom settings
atb dashboards run monitoring --host 0.0.0.0 --port 8000
```

## Configuration
```bash
export DATABASE_URL="postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot"
export FLASK_SECRET_KEY="your-secret-key"
```

## API
- GET `/api/balance`
- GET `/api/balance/history?days=30`
- POST `/api/balance`
- GET `/api/status`
- WebSocket `metrics_update`

## Architecture
- Backend: Flask + Flask-SocketIO → PostgreSQL via `database.manager.DatabaseManager`
- Frontend: HTML/CSS/JS (Chart.js), responsive UI

Data flow:
```
PostgreSQL → Dashboard Backend → WebSocket → Frontend
```

## Customization
- Add metrics in `src/monitoring/dashboard.py`
- Edit templates in `src/monitoring/templates/`

## Security
- Set `FLASK_SECRET_KEY`
- Restrict access and use HTTPS

---
Disclaimer: Monitoring only. Validate decisions independently.
