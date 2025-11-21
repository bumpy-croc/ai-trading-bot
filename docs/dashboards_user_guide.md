# Trading Bot Dashboards - User Guide

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Dashboard Sections](#dashboard-sections)
4. [Features Guide](#features-guide)
5. [Keyboard Shortcuts](#keyboard-shortcuts)
6. [Export Data](#export-data)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Overview

The Trading Bot Monitoring Dashboard provides real-time and historical insights into your trading bot's performance, risk metrics, ML model accuracy, and system health. The enhanced dashboard features a tabbed interface with six specialized sections, dark/light theme support, and comprehensive export functionality.

### Dashboard Types

The system includes three dashboards:

1. **Monitoring Dashboard** (Port 8080) - Main real-time trading monitor **(This Guide)**
2. **Backtesting Dashboard** (Port 8001) - Historical backtest results viewer
3. **Market Prediction Dashboard** (Port 8002) - Price forecasting and sentiment analysis

---

## Getting Started

### Launching the Dashboard

```bash
# Start the monitoring dashboard
atb dashboards run monitoring --port 8080

# The dashboard will be available at:
# http://localhost:8080
```

### First-Time Setup

1. **Ensure Database is Running**
   ```bash
   docker compose up -d postgres
   export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
   ```

2. **Verify Connection**
   ```bash
   atb db verify
   ```

3. **Generate Sample Data** (Optional)
   ```bash
   # Run a quick backtest to populate the database
   atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

   # Or start paper trading to generate live data
   atb live ml_basic --symbol BTCUSDT --paper-trading
   ```

4. **Access Dashboard**
   - Open browser to `http://localhost:8080`
   - You should see the Overview tab with key metrics

---

## Dashboard Sections

### 1. Overview Tab

The default landing page providing a high-level summary of your trading bot.

**Key Components:**

- **Key Metrics Cards**
  - Current Balance, Daily P&L, Win Rate, Active Positions
  - Dynamic Risk Factor (if adaptive risk management is enabled)
  - Total Trades, Total P&L
  - Color-coded for quick status assessment (green=positive, red=negative)

- **Portfolio Performance Chart**
  - 7-day equity curve by default
  - Toggle between 7D, 30D, 90D views
  - Smooth line chart showing balance over time

- **Active Positions Table**
  - Real-time position monitoring
  - Columns: Symbol, Side, Size, Entry, Current Price, P&L, Trailing SL, Breakeven, MFE, MAE, Close Target
  - **Close Target**: Shows how much more profit/loss until TP or SL is hit
  - Toggle between currency ($) and percentage (%) display
  - Export button for CSV download

- **Recent Trades Table**
  - Last 10 completed trades
  - Columns: Symbol, Side, Quantity, Entry, Exit, P&L
  - Color-coded P&L (green for profits, red for losses)
  - Export button for CSV download

**Usage Tips:**
- Monitor the Dynamic Risk Factor indicator for risk management status
- Check Close Target to see which positions are near their exit points
- Use export buttons to analyze data in Excel or other tools

---

### 2. Performance Tab

Advanced performance analytics with rolling metrics and trend analysis.

**Key Components:**

- **Time Range Selector**
  - Choose from: 7, 30, 90, or 365 days
  - Affects all charts on this tab

- **Rolling Sharpe Ratio Chart**
  - 7-day rolling Sharpe ratio (annualized)
  - Measures risk-adjusted returns over time
  - Values > 1.0 indicate good performance
  - Smooth line chart with trend visualization

- **Drawdown Over Time Chart**
  - Shows percentage drawdown from peak balance
  - Helps identify losing streaks and recovery periods
  - 0% means at all-time high
  - Negative values show how far below peak

- **Win Rate Trending Chart**
  - Daily win rate percentage
  - Helps identify if strategy effectiveness is changing
  - 50%+ generally considered good

- **Equity Curve with Drawdown**
  - Combined view of balance and drawdown
  - Visualize relationship between equity growth and risk

**Export:**
- Click "Export Data" to download all performance metrics as CSV
- Includes timestamps, balance, equity, P&L, and drawdown data

**Interpreting the Charts:**

- **Rising Sharpe**: Strategy is improving risk-adjusted returns
- **Flat Drawdown**: Consistent performance without major losing streaks
- **Increasing Win Rate**: Strategy is adapting well to current market
- **Declining Trends**: May indicate strategy degradation or changing market conditions

---

### 3. Trade Analysis Tab

Detailed analysis of trade patterns, timing, and distributions.

**Key Components:**

- **Time Range Selector**
  - 7, 30, or 90 days
  - Filters all data on this tab

- **Trade Statistics Cards**
  - Total Trades
  - Average Duration (hours)
  - Median Duration (hours)
  - Best Trade (largest profit)

- **P&L Distribution Histogram**
  - Shows distribution of trade P&L across bins
  - Helps identify if profits are consistent or sporadic
  - Mean, median, and std dev displayed in tooltip

- **Profit by Hour of Day Chart**
  - Bar chart showing total profit for each hour (0-23)
  - Green bars = profit, red bars = loss
  - Identify best/worst trading hours

- **Profit by Day of Week Chart**
  - Bar chart for Monday through Sunday
  - Identify which days are most profitable
  - Useful for avoiding unprofitable trading periods

- **Best & Worst Trades Table**
  - Top 3 best trades and top 3 worst trades
  - Shows symbol and P&L
  - Helpful for understanding what works and what doesn't

**Usage Tips:**

- **Look for patterns**: If certain hours consistently lose money, consider avoiding them
- **Distribution analysis**: A normal distribution suggests consistent strategy performance
- **Duration**: Compare avg vs median to identify outlier trades
- **Day of week**: Some strategies perform better on certain days due to market volume patterns

---

### 4. ML Models Tab

Machine learning model performance tracking and accuracy monitoring.

**Key Components:**

- **Model Selector**
  - Dropdown to filter by specific model
  - "All Models" shows aggregate data
  - Auto-populated from `prediction_performance` table

- **Time Range Selector**
  - 7, 30, or 90 days

- **Model Performance Summary Cards**
  - Average MAE (Mean Absolute Error)
  - Average RMSE (Root Mean Squared Error)
  - Average MAPE (Mean Absolute Percentage Error)
  - Information Coefficient (IC)

- **MAE Over Time Chart**
  - Lower is better
  - Shows model prediction accuracy trending
  - Rising MAE may indicate model degradation

- **RMSE Over Time Chart**
  - Similar to MAE but penalizes larger errors more
  - Useful for detecting outlier predictions

**Understanding the Metrics:**

- **MAE**: Average absolute difference between predicted and actual prices
  - Example: MAE of 150 means predictions are off by $150 on average

- **RMSE**: Similar to MAE but gives more weight to large errors
  - Higher than MAE if there are occasional large mispredictions

- **MAPE**: Percentage error, easier to interpret across different price levels
  - Example: MAPE of 2.5% means predictions are off by 2.5% on average

- **IC (Information Coefficient)**: Correlation between predictions and actual outcomes
  - Range: -1 to 1
  - Values > 0.05 are generally considered useful
  - Values > 0.1 are considered very good

**No Data Available:**
If you see "No model performance data available", it means:
- No ML predictions have been made yet
- Run a backtest or live trading session with ML strategies to populate this data
- The `prediction_performance` table is empty

---

### 5. Risk Tab

Comprehensive risk metrics, exposure analysis, and correlation visualization.

**Key Components:**

- **Risk Metrics Cards**
  - VaR (95%): Value at Risk at 95% confidence level
  - Current Drawdown: Current percentage below peak
  - Max Drawdown: Worst historical drawdown
  - Total Exposure: Total value of open positions

- **Position Concentration Pie Chart**
  - Shows percentage allocation across symbols
  - Helps identify over-concentration in single assets
  - Diversification visualization

- **Risk Adjustments History Table**
  - Recent risk parameter changes
  - Shows parameter name, adjustment factor, reason, and timestamp
  - Useful for understanding why position sizes changed

- **Correlation Heatmap**
  - Shows correlation between traded symbols
  - Values range from -1 (inverse) to 1 (perfect correlation)
  - Color-coded: Green for positive, Red for negative
  - Helps assess portfolio diversification

**Interpreting Risk Metrics:**

- **VaR (95%)**: "There is a 95% chance losses won't exceed this amount"
  - Example: VaR of -$500 means 95% of the time, daily loss < $500

- **Drawdown**: Distance from peak balance
  - 0% = at all-time high
  - -10% = currently 10% below peak

- **Concentration**: Ideal is balanced across multiple assets
  - >50% in one asset = high concentration risk

- **Correlation**:
  - >0.7 = highly correlated (not diversified)
  - <0.3 = low correlation (well diversified)
  - <0 = inverse correlation (natural hedge)

---

### 6. System Tab

System health monitoring, error tracking, and operational metrics.

**Key Components:**

- **System Health Cards**
  - DB Latency: Database query response time (ms)
  - API Status: Connection status to exchange APIs
  - Memory Usage: System memory utilization (%)
  - Uptime: How long the dashboard has been running

- **Error Rate Display**
  - Large percentage showing error rate in last hour
  - Calculated as: (Errors / Total Events) × 100%
  - Should be close to 0% during normal operation

- **Recent Errors Table**
  - Last 10 errors or warnings
  - Columns: Message, Severity (ERROR/WARNING), Timestamp
  - Color-coded severity badges

**Health Indicators:**

- **DB Latency**:
  - <50ms: Excellent
  - 50-200ms: Good
  - 200-500ms: Acceptable
  - >500ms: Slow (investigate database performance)

- **API Status**:
  - "Connected": Normal
  - "Disconnected": Check network/API keys
  - "Rate Limited": Reduce API call frequency

- **Error Rate**:
  - 0-1%: Normal (occasional errors expected)
  - 1-5%: Monitor closely
  - >5%: Investigate immediately

**Troubleshooting:**
- High error rate? Check Recent Errors table for patterns
- High latency? Database might be overloaded or network issues
- Check logs: `railway logs --environment production`

---

## Features Guide

### Dark/Light Theme Toggle

**Location:** Top-right corner of navbar (moon/sun icon)

**Usage:**
1. Click the moon icon to switch to light mode
2. Click the sun icon to switch back to dark mode
3. Preference is saved to browser localStorage
4. Persists across sessions

**Customization:**
- Modify CSS variables in `dashboard.css` for custom themes
- Variables: `--dark-bg`, `--text-primary`, `--border-color`, etc.

---

### Keyboard Shortcuts

Press `?` to see keyboard shortcuts help modal.

**Available Shortcuts:**

| Key | Action |
|-----|--------|
| `1` | Switch to Overview tab |
| `2` | Switch to Performance tab |
| `3` | Switch to Trade Analysis tab |
| `4` | Switch to ML Models tab |
| `5` | Switch to Risk tab |
| `6` | Switch to System tab |
| `R` | Refresh current tab data |
| `E` | Export current tab data (CSV) |
| `?` | Show keyboard shortcuts help |

**Tips:**
- Shortcuts don't work when typing in input fields
- Use `E` to quickly export data without clicking
- Use number keys for rapid navigation between tabs

---

### Configuration Panel

**Access:** Click "Settings" button in navbar

**Options:**

1. **Update Interval**
   - How often to refresh data (in seconds)
   - Default: 5 seconds
   - Range: 1-3600 seconds
   - Lower = more real-time, but higher load

2. **Visible Metrics**
   - Check/uncheck metrics to show/hide in Overview tab
   - Reduces clutter if you only care about specific metrics
   - Saves to server configuration

3. **Close Target Display Options**
   - Click gear icon next to positions table
   - Configure what to show in Close Target column
   - Options: Take Profit, Stop Loss, Trailing Stop, Partial Exits
   - Tooltip customization: Risk:Reward, Time in Position
   - Adjust tooltip length and refresh interval

---

## Export Data

### Supported Formats

Currently supports CSV export for:
- Active Positions
- Recent Trades
- Performance Metrics

**Future Support (Planned):**
- PDF reports
- JSON exports
- Scheduled email reports

### How to Export

**Method 1: Export Buttons**
1. Navigate to the relevant tab
2. Click the "Export" or "Export Data" button
3. CSV file downloads automatically
4. Filename includes current date

**Method 2: Keyboard Shortcut**
1. Navigate to the relevant tab
2. Press `E` key
3. CSV file downloads automatically

**Method 3: Direct API Access**
```bash
# Export trades (last 30 days)
curl http://localhost:8080/api/export/trades?days=30 > trades.csv

# Export performance metrics
curl http://localhost:8080/api/export/performance?days=30 > performance.csv

# Export current positions
curl http://localhost:8080/api/export/positions > positions.csv
```

### CSV File Structure

**Trades Export:**
```csv
symbol,side,entry_price,exit_price,quantity,entry_time,exit_time,pnl,pnl_percent,strategy_name,exit_reason
BTCUSDT,long,50000.00,51000.00,0.1,2025-11-01 10:00:00+00:00,2025-11-01 14:00:00+00:00,100.00,2.00,ml_basic,take_profit
```

**Performance Export:**
```csv
timestamp,balance,equity,total_pnl,daily_pnl,drawdown,open_positions
2025-11-01 00:00:00+00:00,10000.00,10100.00,100.00,50.00,-2.50,2
```

**Positions Export:**
```csv
symbol,side,entry_price,current_price,quantity,unrealized_pnl,entry_time
BTCUSDT,long,50000.00,51000.00,0.1,100.00,2025-11-01 10:00:00+00:00
```

---

## Troubleshooting

### Dashboard Won't Load

**Symptom:** Blank page or "Connection Refused" error

**Solutions:**
1. Check if dashboard is running:
   ```bash
   ps aux | grep dashboard
   ```

2. Verify port is correct:
   ```bash
   atb dashboards run monitoring --port 8080
   ```

3. Check database connection:
   ```bash
   atb db verify
   ```

4. Check logs:
   ```bash
   tail -f logs/dashboard.log
   ```

---

### No Data Showing

**Symptom:** Tables show "No data available" or charts are empty

**Solutions:**

1. **No Active Positions:**
   - This is normal if bot isn't currently trading
   - Start paper trading or run a backtest to generate data

2. **No Recent Trades:**
   - Bot hasn't made any trades yet
   - Check bot is running: `atb live status`

3. **No Performance Data:**
   - Requires at least 2 account balance snapshots
   - Run bot for a few minutes to generate data

4. **No Model Performance:**
   - Only populated when ML predictions are made
   - Run ML strategy: `atb live ml_basic --paper-trading`

---

### Charts Not Updating

**Symptom:** Dashboard loads but data doesn't refresh

**Solutions:**

1. Check WebSocket connection:
   - Open browser developer console (F12)
   - Look for WebSocket connection status
   - Should see "Connected to monitoring dashboard"

2. Check update interval:
   - Click Settings
   - Verify Update Interval is reasonable (5-10 seconds)

3. Hard refresh:
   - Press `Ctrl+Shift+R` (Windows/Linux)
   - Press `Cmd+Shift+R` (Mac)

---

### Export Not Working

**Symptom:** Export button doesn't download file

**Solutions:**

1. Check browser pop-up blocker:
   - Allow pop-ups from localhost
   - Try export again

2. Check browser download settings:
   - Ensure downloads aren't blocked
   - Check download folder permissions

3. Try direct API access:
   ```bash
   curl http://localhost:8080/api/export/trades?days=30 > trades.csv
   ```

4. Check server logs for errors:
   ```bash
   tail -f logs/dashboard.log
   ```

---

### Performance Issues

**Symptom:** Dashboard is slow or laggy

**Solutions:**

1. Reduce update frequency:
   - Settings → Update Interval → Increase to 30 seconds

2. Clear browser cache:
   - Press `Ctrl+Shift+Delete`
   - Clear cached data

3. Reduce chart history:
   - Use shorter time ranges (7D instead of 90D)
   - Fewer data points = faster rendering

4. Check database performance:
   ```bash
   atb db verify
   ```

5. Optimize database queries (advanced):
   - Add indexes if missing
   - Run `VACUUM ANALYZE` on PostgreSQL

---

## Advanced Usage

### Custom Themes

Modify `/static/css/dashboard.css`:

```css
/* Add custom theme */
[data-theme="custom"] {
    --dark-bg: #your-color;
    --text-primary: #your-color;
    /* ... other variables */
}
```

Then update JavaScript in `dashboard-enhanced.js` to add theme option.

---

### API Integration

All dashboard data is available via REST API:

**Endpoints:**

```bash
# Metrics
GET /api/metrics

# Positions
GET /api/positions
GET /api/positions/<id>/orders

# Trades
GET /api/trades?limit=50

# Performance
GET /api/performance?days=30
GET /api/performance/advanced?days=30&window=7

# Trade Analysis
GET /api/trades/analysis?days=30
GET /api/trades/distribution?days=30&bins=20

# Models
GET /api/models/list
GET /api/models/performance?days=30&model=model_name

# Risk
GET /api/risk/detailed
GET /api/correlation/matrix-formatted

# System
GET /api/system/health-detailed

# Export
GET /api/export/trades?days=30
GET /api/export/performance?days=30
GET /api/export/positions
```

**Example: Python Script**

```python
import requests

# Get current positions
response = requests.get('http://localhost:8080/api/positions')
positions = response.json()

for pos in positions:
    print(f"{pos['symbol']}: {pos['unrealized_pnl']}")
```

---

### Embedding in Other Tools

Dashboard can be embedded in external tools:

**Grafana Integration:**
1. Use Infinity data source plugin
2. Point to dashboard API endpoints
3. Create custom Grafana dashboard

**Jupyter Notebook:**
```python
import pandas as pd
import requests

# Fetch trade data
response = requests.get('http://localhost:8080/api/export/trades?days=90')
trades_csv = response.text

# Load into pandas
df = pd.read_csv(io.StringIO(trades_csv))
df['pnl'].plot()
```

---

### Production Deployment

When deploying to production (Railway, AWS, etc.):

1. **Set Environment Variables:**
   ```bash
   DATABASE_URL=postgresql://...
   LOG_LEVEL=INFO
   LOG_JSON=true
   WEB_SERVER_USE_GEVENT=1  # Use gevent for production
   ```

2. **Configure Port:**
   ```bash
   PORT=8080  # Railway auto-detects this
   ```

3. **Enable HTTPS:**
   - Railway provides this automatically
   - For custom deployment, use nginx reverse proxy

4. **Monitor Logs:**
   ```bash
   railway logs --environment production
   ```

5. **Set Up Alerts:**
   - Configure monitoring for high error rates
   - Set up health check endpoints
   - Use uptime monitoring services (UptimeRobot, Pingdom)

---

## Best Practices

### Monitoring Your Bot

1. **Check Dashboard Daily**
   - Review overnight performance in Overview tab
   - Check for any errors in System tab
   - Verify positions are behaving as expected

2. **Weekly Analysis**
   - Review Performance tab for trends
   - Analyze Trade Analysis tab for patterns
   - Check Model Performance if using ML strategies

3. **Monthly Review**
   - Export all data for month
   - Calculate custom metrics in Excel
   - Review risk management effectiveness

### Data Retention

- Dashboard shows recent data based on selected time ranges
- Database retains all historical data
- Consider archiving old data after 1+ year
- Regular backups: `atb db backup`

### Security

- Don't expose dashboard publicly without authentication
- Use VPN or SSH tunnel for remote access
- Keep API keys secure (never commit to git)
- Regularly update dependencies

---

## FAQ

**Q: How often does the dashboard update?**
A: Default is 5 seconds. Configurable in Settings panel.

**Q: Can I run multiple dashboards simultaneously?**
A: Yes, each dashboard type runs on a different port.

**Q: Does the dashboard work with paper trading?**
A: Yes! All features work with both paper and live trading.

**Q: Can I customize which metrics are shown?**
A: Yes, use the Settings panel to enable/disable metrics.

**Q: How do I add custom charts?**
A: Modify `dashboard-enhanced.js` and add new Chart.js visualizations.

**Q: Can I access the dashboard remotely?**
A: Yes, but secure it properly (VPN, SSH tunnel, or authentication layer).

**Q: What browsers are supported?**
A: Modern browsers: Chrome, Firefox, Safari, Edge (latest versions).

**Q: Can I change the default port?**
A: Yes: `atb dashboards run monitoring --port 9000`

**Q: How do I report bugs or request features?**
A: https://github.com/anthropics/claude-code/issues

---

## Appendix

### Keyboard Shortcuts Reference Card

Print or bookmark this reference:

```
═══════════════════════════════════════════
  TRADING BOT DASHBOARD - KEYBOARD SHORTCUTS
═══════════════════════════════════════════

NAVIGATION:
  1-6        Switch tabs (Overview, Performance, Trades, Models, Risk, System)

ACTIONS:
  R          Refresh current tab
  E          Export current tab data (CSV)
  ?          Show this help
  Esc        Close modals/panels

TIPS:
  - Shortcuts don't work when typing in inputs
  - Use number keys for fast tab switching
  - Press E to export without clicking

═══════════════════════════════════════════
```

### Metric Definitions

**Technical Metrics:**
- **Sharpe Ratio**: Risk-adjusted return measure (higher is better, >1 is good)
- **Drawdown**: Percentage decline from peak (0% = at peak, negative = below peak)
- **Win Rate**: Percentage of profitable trades (>50% generally good)
- **P&L**: Profit and Loss in currency units
- **MFE (Maximum Favorable Excursion)**: Best profit point during trade
- **MAE (Maximum Adverse Excursion)**: Worst drawdown during trade

**ML Metrics:**
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (penalizes large errors more)
- **MAPE**: Mean Absolute Percentage Error (percentage-based error)
- **IC**: Information Coefficient (correlation between prediction and reality)

**Risk Metrics:**
- **VaR**: Value at Risk (expected worst-case loss at confidence level)
- **Correlation**: Statistical relationship between assets (-1 to 1)
- **Concentration**: Percentage allocation to single asset
- **Exposure**: Total value of open positions

---

## Version History

**v2.0** (November 2025)
- Added tabbed interface with 6 sections
- Implemented dark/light theme toggle
- Added keyboard shortcuts
- Added CSV export functionality
- Comprehensive chart visualizations
- Mobile-responsive design

**v1.0** (October 2025)
- Initial dashboard release
- Basic metrics and positions table
- Simple equity curve chart
- WebSocket real-time updates

---

## Support & Resources

- **Documentation**: `docs/` directory
- **Code**: `src/dashboards/monitoring/`
- **Tests**: `tests/unit/test_dashboard_analytics.py`
- **Issues**: GitHub Issues
- **Community**: Trading Bot Discord/Forum

---

**Last Updated**: November 21, 2025
**Author**: Claude (AI Assistant)
**License**: See project LICENSE file
