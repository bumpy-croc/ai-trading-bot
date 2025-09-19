# Monitoring Dashboard

Real-time monitoring interface for live trading performance and system health.

## Features

- Live trading performance metrics
- Real-time balance and position tracking  
- Trade execution history
- Risk management status
- System health indicators
- WebSocket-based live updates

## Usage

```bash
# Start monitoring dashboard
atb dashboards run monitoring --port 8000

# Access at http://localhost:8000
```

## Technology

- Flask application with Socket.IO for real-time updates
- Bootstrap UI components
- Chart.js for data visualization
- WebSocket connection to trading engine