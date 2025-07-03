# Trading Bot Monitoring Dashboard

## Overview
Real-time web dashboard for monitoring cryptocurrency trading bot performance, risk metrics, and system health.

## Key Metrics Monitored

### System Health
- **API Connection Status**: Real-time connectivity to exchanges
- **Data Feed Status**: Active/Delayed/Stale data monitoring
- **Error Rate**: Hourly error percentage
- **System Uptime**: Bot uptime duration
- **Last Update**: Most recent data timestamp

### Risk Management
- **Current Drawdown**: Real-time drawdown from peak
- **Max Drawdown**: Largest historical drawdown
- **Daily/Weekly P&L**: Profit/loss tracking
- **Position Sizes**: Total value of active positions
- **Risk Per Trade**: Configured risk percentage

### Trading Performance
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Fill Rate**: Successfully executed orders percentage
- **Average Slippage**: Execution slippage tracking
- **Total Trades**: Completed trades count

### Account Status
- **Current Balance**: Real-time account balance
- **Active Positions**: Number of open positions
- **Unrealized P&L**: Total unrealized profit/loss
- **Available Margin**: Available margin for new positions

## Quick Start

### Installation
```bash
pip install -r monitoring/requirements.txt
python start_dashboard.py
```

### Access
- **URL**: http://localhost:8080
- **Updates**: Via WebSocket (1-hour default)
- **Mobile**: Responsive design works on all devices

## Configuration
- **Gear Icon**: Toggle metrics on/off
- **Priorities**: High (red), Medium (yellow), Low (green)
- **Custom Metrics**: Easy to add via `dashboard.py`

## File Structure
```
monitoring/
├── dashboard.py          # Main application
├── templates/
│   └── dashboard.html    # Frontend interface
├── requirements.txt      # Dependencies
└── demo_data.py         # Sample data generator

start_dashboard.py       # Launcher script
```

## Key Features
- **Real-time Updates**: WebSocket-based live data
- **Mobile Optimized**: Works on all screen sizes
- **Configurable**: Enable/disable any metric
- **Secure**: Read-only access, no trading data modification
- **Production Ready**: HTTPS support, error handling

## Adding Custom Metrics
```python
# In dashboard.py
def _get_my_metric(self) -> float:
    # Your calculation logic
    return calculated_value

# Add to monitoring config
'my_metric': {
    'enabled': True,
    'priority': 'high',
    'format': 'currency'
}
```

## Critical Alerts
- API disconnection
- High error rates
- Excessive drawdown
- Failed order execution

The dashboard provides complete visibility into trading bot operations with a clean, professional interface that updates in real-time. 