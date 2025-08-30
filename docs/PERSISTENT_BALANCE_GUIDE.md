# Persistent Balance & Position Management Guide

## Overview

The AI Trading Bot now includes a comprehensive persistent balance and position management system that ensures your trading progress is never lost when the service restarts. This addresses the critical issue of losing balance and position data during Railway deployments or service restarts.

## 🎯 Key Features

### ✅ Persistent Balance Tracking
- **Automatic Recovery**: Resume from last known balance on restart
- **Real-time Updates**: Balance updated after every trade
- **Manual Adjustments**: Modify balance through the dashboard
- **Complete History**: Track all balance changes with timestamps and reasons

### ✅ Position Recovery
- **State Preservation**: Active positions are recovered on restart
- **Complete Context**: Entry price, stop loss, take profit all restored
- **Seamless Continuation**: Trading resumes exactly where it left off

### ✅ Multi-Currency Support
- **USD Base**: Primary balance tracking in USD
- **Asset Breakdown**: Track individual crypto holdings
- **Exchange Integration**: Ready for multi-asset trading

## 🚀 How It Works

### 1. **First Time Setup**
```bash
# Ensure PostgreSQL is running and DATABASE_URL is set
python scripts/verify_database_connection.py

# Start trading with initial balance
atb live ml_basic --symbol BTCUSDT --paper-trading
```

### 2. **Balance Recovery Process**
When the bot starts:
1. **Check for Active Session**: Looks for existing trading session
2. **Recover Balance**: Gets last known balance from database
3. **Restore Positions**: Recovers all active positions
4. **Resume Trading**: Continues exactly where it left off

```
🔍 Found active session #123
💾 Recovered balance from previous session: $1,250.00
🔄 Recovering 2 active positions...
✅ Recovered position: BTCUSDT long @ $45,000.00
✅ Recovered position: BTCUSDT long @ $46,500.00
🎯 Successfully recovered 2 positions
```

### 3. **Balance Updates**
The system updates balance in real-time:
- **Trade P&L**: Automatically updated after each trade
- **Manual Adjustments**: Via dashboard or API
- **Deposits/Withdrawals**: Track external balance changes

## 📊 Dashboard Balance Management

### View Current Balance
- **Real-time Display**: Current balance prominently shown
- **24h Change**: See balance change over last 24 hours
- **Update History**: Track recent balance modifications

### Manual Balance Adjustment
```javascript
// Via API
POST /api/balance
{
  "balance": 5000,
  "reason": "Added funds from external account",
  "updated_by": "user"
}
```

### Balance History
- **Complete Timeline**: See all balance changes
- **Reason Tracking**: Why each change was made
- **User Attribution**: Who made manual adjustments

## 🔧 Configuration Options

### LiveTradingEngine Parameters
```python
engine = LiveTradingEngine(
    strategy=strategy,
    data_provider=data_provider,
    initial_balance=1000,              # Used only for new sessions
    resume_from_last_balance=True,     # Enable balance recovery (default)
    enable_live_trading=False,         # Paper trading mode
)
```

### Command Line Options
```bash
# Resume from last balance (default)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Start fresh (force reset - requires explicit confirmation)
# Note: Check CLI help for specific balance reset options
atb live ml_basic --symbol BTCUSDT --paper-trading --help
```

## 💡 Usage Scenarios

### Scenario 1: Gradual Capital Increase
1. **Start Small**: Begin with $1,000
2. **Monitor Performance**: Let bot trade for a week
3. **Increase Capital**: Use dashboard to adjust balance to $5,000
4. **Continue Trading**: Bot automatically uses new balance for position sizing

### Scenario 2: Service Restart
1. **Active Trading**: Bot has $1,250 balance and 2 open positions
2. **Railway Restart**: Service redeploys automatically
3. **Automatic Recovery**: Bot recovers $1,250 balance and 2 positions
4. **Seamless Continuation**: Trading continues without interruption

### Scenario 3: Strategy Change
1. **Switch Strategies**: Change configuration of `ml_basic` (e.g., model update)
2. **Keep Positions**: Existing positions remain open
3. **Preserve Balance**: Current balance maintained
4. **Updated Strategy**: Uses recovered balance for new trades

## 🛠️ Database Schema

### AccountBalance Table
```sql
CREATE TABLE account_balances (
    id INTEGER PRIMARY KEY,
    base_currency VARCHAR(10) DEFAULT 'USD',
    total_balance FLOAT NOT NULL,
    available_balance FLOAT NOT NULL,
    reserved_balance FLOAT DEFAULT 0.0,
    asset_balances JSON,
    last_updated DATETIME NOT NULL,
    updated_by VARCHAR(50) DEFAULT 'system',
    update_reason VARCHAR(200),
    session_id INTEGER REFERENCES trading_sessions(id)
);
```

## 🔄 Migration Process

### For Existing Users
```bash
# 1. Stop any running trading bot
# 2. Run database migration
atb db migrate

# 3. Validate database connectivity
python scripts/verify_database_connection.py

# 4. Restart trading bot
atb live ml_basic --symbol BTCUSDT --paper-trading
```

### Migration Details
The migration script:
- **Adds AccountBalance table** for persistent tracking
- **Migrates existing sessions** to have balance records
- **Calculates current balance** from trade history
- **Preserves all data** - no information lost

## 📱 API Endpoints

### Get Balance Information
```
GET /api/balance
```
Response:
```json
{
  "current_balance": 1250.00,
  "balance_change_24h": 5.2,
  "last_updated": "2024-01-15T10:30:00Z",
  "last_update_reason": "trade_pnl_take_profit",
  "recent_history": [...]
}
```

### Update Balance
```
POST /api/balance
{
  "balance": 5000,
  "reason": "Capital increase",
  "updated_by": "trader"
}
```

### Get Balance History
```
GET /api/balance/history?limit=50
```

## ⚠️ Important Notes

### USD vs Crypto Balance
- **Primary Tracking**: USD is the primary balance currency
- **Position Sizing**: Calculated in USD for consistency
- **Multi-Asset Future**: Ready for BTC/ETH base currencies
- **Exchange Integration**: Works with any USD-based trading

### Binance Integration
- **Virtual Balance**: Bot tracks its own virtual USD balance
- **Trade Execution**: Positions executed with actual crypto
- **P&L Calculation**: Converted back to USD for tracking
- **Real Balance**: Monitor actual exchange balance separately

### Risk Management
- **Max Position Size**: Still enforced (e.g., 10% of balance)
- **Risk Per Trade**: Calculated from current balance
- **Drawdown Limits**: Based on peak balance tracking

## 🔍 Troubleshooting

### Balance Not Recovering
```bash
# Check DB connectivity
atb db verify

# View current balance via database admin UI
# Start admin UI: python src/database_manager/app.py
# Or check via CLI: atb db --help  # Check available database commands

# Ensure database connection is healthy
python scripts/verify_database_connection.py
```

### Positions Not Restored
- Check database for position records
- Verify session_id matches between positions and sessions
- Review logs for recovery errors

### Dashboard Balance Management
- Ensure database migration completed
- Check API endpoints are responding
- Verify balance update permissions

## 🎉 Benefits

### For Traders
- **Never Lose Progress**: Balance and positions survive restarts
- **Flexible Capital Management**: Easily adjust trading capital
- **Complete Transparency**: Full history of all changes
- **Peace of Mind**: Trading continues seamlessly

### For Developers
- **Robust Architecture**: Clean separation of concerns
- **Extensible Design**: Ready for multi-asset trading
- **Comprehensive Logging**: Full audit trail
- **Database Integrity**: Consistent data model

## 🚀 Future Enhancements

### Planned Features
- **Multi-Asset Balances**: Trade with BTC/ETH base currencies
- **Portfolio Rebalancing**: Automatic asset allocation
- **External Balance Sync**: Integration with exchange APIs
- **Advanced Risk Management**: Position-level balance allocation

### Coming Soon
- **Mobile Dashboard**: iOS/Android balance management
- **Automated Deposits**: Scheduled capital increases
- **Performance Attribution**: Track returns by strategy
- **Risk Metrics**: Real-time exposure monitoring

---

This persistent balance system ensures your trading bot investment grows safely and continuously, regardless of technical restarts or deployments. Your progress is now truly persistent and recoverable.
