# Monitoring

Real-time web dashboard for tracking live trading performance, system health, and risk metrics.

## Overview

The monitoring dashboard provides real-time visibility into trading operations with WebSocket-based live updates. It displays critical metrics including account balance, active positions, performance statistics, risk levels, and system health.

## Features

- **Real-time updates** - WebSocket (Socket.IO) for live data streaming
- **System health** - Status indicators, error tracking, last update time
- **Risk monitoring** - Current drawdown, exposure, risk per trade
- **Performance metrics** - Win rate, Sharpe ratio, total trades
- **Account overview** - Balance, active positions, unrealized P&L, available margin
- **Trade history** - Recent trades with entry/exit prices and P&L
- **Charts and visualizations** - Balance curve, drawdown chart, performance graphs
- **Responsive design** - Works on desktop and mobile
- **Database integration** - PostgreSQL-backed metrics and history

## Quick Start

### CLI
```bash
# Start monitoring dashboard (default port 8000)
atb dashboards run monitoring --port 8000

# Custom update interval (default: 3600 seconds)
atb dashboards run monitoring --port 8000 --update-interval 60

# Open in browser
# Navigate to http://localhost:8000
```

### Configuration

The dashboard automatically detects configuration from environment:
- `DATABASE_URL` - PostgreSQL connection (required)
- `WEB_SERVER_USE_GEVENT` - Use gevent for production (Railway sets this automatically)
- `FLASK_SECRET_KEY` - Session encryption key

### Production Deployment

For production environments (Railway, etc.):
```bash
# Set environment variable for production-safe server
export WEB_SERVER_USE_GEVENT=1
atb dashboards run monitoring --port $PORT
```

## Dashboard Sections

### System Health
- **Status** - Online/Offline/Error
- **Last Update** - Timestamp of last data refresh
- **Error Count** - Number of recent errors
- **Uptime** - System uptime duration

### Risk Metrics
- **Current Drawdown** - Peak-to-current decline
- **Max Drawdown** - Historical worst drawdown
- **Current Exposure** - Total position size as % of balance
- **Risk Per Trade** - Average risk per trade
- **Dynamic Risk Factor** - Current risk adjustment multiplier
- **Risk Reason** - Why risk is adjusted (if applicable)

### Performance
- **Win Rate** - Percentage of profitable trades
- **Sharpe Ratio** - Risk-adjusted return metric
- **Total Trades** - Number of completed trades
- **Profit Factor** - Gross profit / gross loss
- **Total Return** - Cumulative percentage gain/loss

### Account
- **Balance** - Current account balance
- **Active Positions** - Number of open positions
- **Unrealized P&L** - Floating profit/loss on open positions
- **Available Margin** - Remaining buying power
- **Position Details** - List of open positions with size, entry, current price

### Trade History
- Recent trades with:
  - Symbol and side (long/short)
  - Entry and exit prices
  - Quantity and P&L
  - Exit reason and timestamps

## Customization

### Adding Custom Metrics

Edit `src/dashboards/monitoring/dashboard.py` to add new metrics:

```python
def get_dashboard_metrics():
    metrics = {
        # Existing metrics...
        
        # Add custom metric
        'custom_metric': calculate_custom_metric(),
    }
    return metrics
```

Update template in `src/dashboards/monitoring/templates/dashboard.html` to display it.

### Styling

CSS is located in `src/dashboards/monitoring/static/css/dashboard.css`. Customize colors, layout, and responsive breakpoints.

### Update Frequency

Control how often data refreshes:
```bash
atb dashboards run monitoring --update-interval 30  # 30 seconds
```

## File Structure

```
src/dashboards/monitoring/
├── dashboard.py        # Main Flask application
├── static/
│   ├── css/
│   │   └── dashboard.css    # Dashboard styling
│   └── js/
│       └── dashboard.js     # Client-side JavaScript
├── templates/
│   └── dashboard.html       # Main dashboard template
└── README.md                # This file
```

## Dependencies

- Flask - Web framework
- Flask-SocketIO - WebSocket support
- PostgreSQL - Data persistence
- Chart.js (via CDN) - Charting library
- Bootstrap (via CDN) - UI framework

## API Endpoints

### REST API
- `GET /api/balance` - Current account balance
- `GET /api/balance/history?days=30` - Balance history
- `POST /api/balance` - Update balance (internal use)
- `GET /api/status` - System status and health
- `GET /api/metrics` - All dashboard metrics

### WebSocket Events
- `metrics_update` - Real-time metric updates
- `trade_update` - New trade notifications
- `position_update` - Position changes

## Security

- Set `FLASK_SECRET_KEY` in production
- Use HTTPS in production (Railway provides this)
- Restrict access via firewall or authentication middleware
- Database credentials via environment variables only

## Troubleshooting

### Dashboard won't start
- Verify `DATABASE_URL` is set: `echo $DATABASE_URL`
- Check PostgreSQL is accessible: `python scripts/verify_database_connection.py`
- Ensure port is not already in use: `lsof -i :8000`

### No data displayed
- Verify trading engine is running and logging to database
- Check database has recent trades: `atb db verify`
- Look for errors in console output

### WebSocket connection fails
- Check browser console for errors
- Verify firewall allows WebSocket connections
- Try with `WEB_SERVER_USE_GEVENT=0` for debugging

## Documentation

See [docs/MONITORING_SUMMARY.md](../../docs/MONITORING_SUMMARY.md) for additional information on monitoring architecture and deployment.
