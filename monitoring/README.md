# Trading Bot Monitoring Dashboard

A comprehensive real-time monitoring dashboard for the cryptocurrency trading bot, providing insights into performance, risk metrics, system health, and trading activity.

## 🎯 Key Features

### Real-Time Monitoring
- **Live Updates**: WebSocket-based real-time data updates every 5 seconds
- **Configurable Metrics**: Enable/disable specific metrics based on your needs
- **Priority-Based Display**: High, medium, and low priority metrics with visual indicators

### Core Metrics Tracked

#### Performance Metrics
- **Total P&L**: Cumulative profit/loss across all trades
- **Current Balance**: Real-time account balance
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of completed trades
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline

#### Risk Management
- **Active Positions**: Number of open positions
- **Current Exposure**: Portfolio exposure as percentage of balance
- **Risk Per Trade**: Average risk per trade
- **Volatility**: Portfolio volatility measurement

#### Market Data
- **Current Price**: Real-time BTC/USDT price
- **24h Price Change**: Daily price movement percentage
- **RSI**: Relative Strength Index indicator
- **EMA Trend**: Moving average trend direction
- **24h Volume**: Trading volume

#### System Health
- **System Status**: Overall system health (Healthy/Warning/Error)
- **API Status**: Exchange API connectivity
- **Error Count**: Recent error occurrences
- **Last Update**: Timestamp of last data refresh

#### Strategy Information
- **Current Strategy**: Active trading strategy
- **Strategy Confidence**: Confidence level (if available)
- **Signals Today**: Number of trading signals generated

## 🚀 Quick Start

### Prerequisites
```bash
# Install dashboard dependencies
pip install -r monitoring/requirements.txt
```

### Launch Dashboard
```bash
# Simple launch (recommended)
python start_dashboard.py

# Or direct launch with options
python monitoring/dashboard.py --host 0.0.0.0 --port 5000
```

### Access Dashboard
Open your browser and navigate to: `http://localhost:5000`

## 📊 Dashboard Interface

### Main Dashboard
- **Key Metrics Grid**: Configurable grid of important metrics
- **Performance Chart**: Real-time portfolio value chart
- **System Health Panel**: System status and health indicators
- **Active Positions Table**: Current open positions with P&L
- **Recent Trades Table**: Latest completed trades

### Configuration Panel
- **Metrics Toggle**: Enable/disable specific metrics
- **Update Interval**: Adjust refresh rate (1-60 seconds)
- **Priority Filtering**: Focus on high-priority metrics
- **Save/Reset Options**: Persist configuration changes

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Custom database URL
export DATABASE_URL="sqlite:///trading_bot.db"

# Optional: Flask secret key for production
export FLASK_SECRET_KEY="your-secret-key-here"
```

### Command Line Options
```bash
python monitoring/dashboard.py \
    --host 0.0.0.0 \           # Bind address (default: 0.0.0.0)
    --port 5000 \              # Port number (default: 5000)
    --debug \                  # Enable debug mode
    --db-url "sqlite:///..." \ # Database URL
    --update-interval 5        # Update interval in seconds
```

### Metric Configuration
The dashboard allows you to customize which metrics are displayed:

1. Click the **gear icon** in the top-right corner
2. Toggle metrics on/off in the configuration panel
3. Adjust update interval if needed
4. Click **Save Configuration** to persist changes

## 🔧 Architecture

### Backend (Python/Flask)
- **Flask**: Web framework for API endpoints
- **Flask-SocketIO**: WebSocket support for real-time updates
- **Database Integration**: Connects to trading bot database
- **Data Providers**: Integrates with exchange APIs for live data

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: WebSocket client for live data
- **Chart.js**: Interactive performance charts
- **Bootstrap**: Modern UI components

### Data Flow
```
Trading Bot Database → Dashboard Backend → WebSocket → Frontend
Exchange APIs → Dashboard Backend → WebSocket → Frontend
```

## 📈 Metrics Deep Dive

### Performance Metrics
- **Total P&L**: Calculated from completed trades in database
- **Win Rate**: (Winning trades / Total trades) × 100
- **Sharpe Ratio**: Annualized risk-adjusted return
- **Max Drawdown**: Largest portfolio decline from peak

### Risk Metrics
- **Current Exposure**: Sum of position values / account balance
- **Risk Per Trade**: Configured risk percentage per trade
- **Volatility**: Annualized standard deviation of returns

### System Health
- **Healthy**: Last update within 5 minutes
- **Warning**: Last update 5-15 minutes ago
- **Error**: No updates for 15+ minutes

## 🛠️ Customization

### Adding New Metrics
1. **Backend**: Add metric calculation method in `dashboard.py`
2. **Configuration**: Add metric to `monitoring_config` dictionary
3. **Frontend**: Metric will automatically appear in configuration panel

Example:
```python
# In dashboard.py
def _get_my_custom_metric(self) -> float:
    # Your calculation logic here
    return calculated_value

# Add to monitoring_config
'my_custom_metric': {
    'enabled': True, 
    'priority': 'medium', 
    'format': 'number'
}
```

### Styling Customization
- Modify CSS variables in `templates/dashboard.html`
- Adjust color scheme, fonts, and layout
- Add custom chart types or visualizations

### Alert Integration
The dashboard supports webhook alerts:
```python
dashboard = MonitoringDashboard(
    alert_webhook_url="https://your-webhook-url.com"
)
```

## 🔒 Security Considerations

### Production Deployment
- Set strong `FLASK_SECRET_KEY`
- Use HTTPS in production
- Implement authentication if needed
- Restrict network access to dashboard

### API Keys
- Never expose exchange API keys in dashboard
- Use environment variables or secure key management
- Implement read-only API access where possible

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablet devices (iPad, Android tablets)
- Mobile phones (iOS, Android)

Mobile-specific features:
- Touch-friendly interface
- Responsive metric cards
- Collapsible navigation
- Optimized chart sizing

## 🐛 Troubleshooting

### Common Issues

**Dashboard won't start**
```bash
# Check dependencies
pip install -r monitoring/requirements.txt

# Verify database access
python -c "from core.database.manager import DatabaseManager; print('DB OK')"
```

**No data showing**
- Verify trading bot database has data
- Check database connection URL
- Ensure exchange API credentials are configured

**WebSocket connection fails**
- Check firewall settings
- Verify port 5000 is available
- Try different browser or clear cache

**Performance issues**
- Reduce update interval
- Disable unnecessary metrics
- Check system resources

### Debug Mode
```bash
python monitoring/dashboard.py --debug
```

### Logs
Dashboard logs are written to console. For production, redirect to file:
```bash
python monitoring/dashboard.py > dashboard.log 2>&1
```

## 🤝 Contributing

### Adding Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

### Reporting Issues
- Use GitHub issues for bug reports
- Include system information and error logs
- Provide steps to reproduce the issue

## 📄 License

This monitoring dashboard is part of the trading bot project and follows the same license terms.

---

**⚠️ Disclaimer**: This dashboard is for monitoring purposes only. Always verify trading decisions independently and understand the risks involved in cryptocurrency trading.