# Trading Bot Monitoring Dashboard

## Overview
Real-time web dashboard for monitoring cryptocurrency trading bot performance, risk metrics, and system health.

## Key Metrics
- System Health (status, errors, last update)
- Risk (current/max drawdown, exposure, risk per trade)
- Performance (win rate, Sharpe ratio, total trades)
- Account (balance, active positions, unrealized P&L, available margin)

## Quick Start

```bash
pip install -r requirements-server.txt
python scripts/start_dashboard.py
# Open http://localhost:8000
```

- Default update interval: 3600 seconds (1 hour)
- Requires PostgreSQL (`DATABASE_URL`)

## Configuration
- Toggle metrics via the gear icon in the UI
- Add metrics in `src/monitoring/dashboard.py`
- Use `--update-interval` CLI flag to change refresh cadence

## File Structure
```
src/monitoring/
├── dashboard.py       # Main app
├── templates/         # HTML templates
├── static/            # CSS/JS assets
scripts/start_dashboard.py  # Launcher script
```

## Notes
- Database: PostgreSQL only
- Uses WebSocket (Flask-SocketIO) for live updates
- Responsive UI for desktop and mobile
