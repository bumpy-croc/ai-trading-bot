# Account Synchronization Guide

## Overview

The Account Synchronization system ensures data integrity between your Binance exchange account and the trading bot's database. This addresses the critical scenario where the bot loses track of positions or trades due to shutdowns, errors, or network issues.

## 🎯 Key Features

### ✅ **Automatic Data Integrity**
- **Balance Synchronization**: Ensures bot's balance matches Binance
- **Position Recovery**: Detects and recovers missing positions
- **Trade Recovery**: Finds and logs missing trades
- **Order Status Updates**: Keeps order status in sync

### ✅ **Robust Error Handling**
- **Graceful Degradation**: Works even when Binance API is unavailable
- **Retry Logic**: Automatic retries for failed operations
- **Fallback Mechanisms**: Uses cached data when exchange is down

### ✅ **Multi-Exchange Ready**
- **Abstract Interface**: Easy to add Coinbase, Kraken, etc.
- **Exchange Agnostic**: Same API regardless of exchange
- **Future-Proof**: Designed for expansion

## 🏗️ Architecture

### **Core Components**

1. **ExchangeInterface** (`src/data_providers/exchange_interface.py`)
   - Abstract base class for exchange operations
   - Defines common API for all exchanges
   - Handles order execution, balance queries, position management

2. **BinanceExchange** (`src/data_providers/binance_exchange.py`)
   - Binance-specific implementation
   - Real order execution and account queries
   - Handles Binance API rate limits and errors

3. **AccountSynchronizer** (`src/live/account_sync.py`)
   - Orchestrates synchronization between exchange and database
   - Detects discrepancies and corrects them
   - Provides emergency recovery capabilities

4. **DatabaseManager** (Enhanced)
   - Added methods for order and trade management
   - Supports position updates and status changes
   - Handles trade recovery and logging

### **Data Flow**

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Binance   │◄──►│ AccountSync  │◄──►│  Database   │
│   Exchange  │    │              │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       │                   │                   │
   Real-time           Comparison          Persistent
   Account Data        & Correction         Storage
```

## 🚀 How It Works

### **1. Initial Synchronization**

When the trading bot starts:

```python
# Automatic sync on startup
if self.account_synchronizer and self.enable_live_trading:
    sync_result = self.account_synchronizer.sync_account_data(force=True)
    
    if sync_result.success:
        # Check if balance was corrected
        balance_sync = sync_result.data.get('balance_sync', {})
        if balance_sync.get('corrected', False):
            corrected_balance = balance_sync.get('new_balance')
            self.current_balance = corrected_balance
```

**What happens:**
1. **Fetch Exchange Data**: Get current balances, positions, orders from Binance
2. **Compare with Database**: Check for discrepancies
3. **Correct Discrepancies**: Update database with exchange data
4. **Log Changes**: Record all corrections for audit trail

### **2. Periodic Synchronization**

Every 30 minutes (configurable):

```python
# Periodic sync during trading
if self.account_synchronizer and self.enable_live_trading:
    sync_result = self.account_synchronizer.sync_account_data()
    if sync_result.success:
        logger.debug("Periodic account sync completed")
```

**What happens:**
1. **Check Last Sync**: Avoid too frequent API calls
2. **Incremental Updates**: Only sync what's changed
3. **Status Updates**: Update order/position status
4. **Balance Verification**: Ensure balance accuracy

### **3. Emergency Recovery**

When data integrity is suspected:

```python
# Emergency sync for data recovery
sync_result = self.account_synchronizer.emergency_sync()
```

**What happens:**
1. **Force Full Sync**: Ignore rate limits, get all data
2. **Trade Recovery**: Look back 30 days for missing trades
3. **Position Reconciliation**: Match all positions exactly
4. **Comprehensive Logging**: Detailed audit trail

## 📊 Synchronization Types

### **Balance Synchronization**

**Purpose**: Ensure bot's balance matches Binance exactly

**Process**:
1. Get USDT balance from Binance
2. Compare with database balance
3. If difference > 1%, correct database
4. Log correction with reason

**Example**:
```
Balance discrepancy detected: DB=$1,250.00 vs Exchange=$1,275.50 (diff: 2.04%)
💰 Balance corrected from exchange: $1,275.50
```

### **Position Synchronization**

**Purpose**: Recover positions that exist on Binance but not in database

**Process**:
1. Get all positions from Binance
2. Compare with database positions
3. Add missing positions to database
4. Close positions that no longer exist on exchange

**Example**:
```
New position found on exchange: BTCUSDT long 0.001
Position closed on exchange: ETHUSDT short 0.5
```

### **Order Synchronization**

**Purpose**: Keep order status in sync between exchange and database

**Process**:
1. Get open orders from Binance
2. Update order status in database
3. Mark cancelled orders as cancelled
4. Log new orders found on exchange

### **Trade Recovery**

**Purpose**: Find and log trades that happened but weren't recorded

**Process**:
1. Get recent trades from Binance (last 7-30 days)
2. Compare with database trades
3. Add missing trades to database
4. Calculate P&L for recovered trades

## 🔧 Configuration

### **Environment Variables**

```bash
# Required for live trading
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional: Use testnet for testing
BINANCE_TESTNET=true
```

### **Trading Engine Configuration**

```python
# Enable account synchronization
engine = LiveTradingEngine(
    strategy=strategy,
    data_provider=data_provider,
    enable_live_trading=True,  # Required for sync
    account_snapshot_interval=1800,  # Sync every 30 minutes
    # ... other parameters
)
```

### **API Permissions**

For account synchronization, your Binance API key needs:

- ✅ **Enable Reading** (required)
- ✅ **Enable Spot & Margin Trading** (for live trading)
- ❌ **Enable Withdrawals** (not needed, security risk)

## 🛠️ Usage Examples

### **Manual Synchronization**

```python
from data_providers.binance_exchange import BinanceExchange
from database.manager import DatabaseManager
from src.live.account_sync import AccountSynchronizer

# Initialize components
exchange = BinanceExchange(api_key, api_secret)
db_manager = DatabaseManager()
synchronizer = AccountSynchronizer(exchange, db_manager, session_id)

# Basic sync
sync_result = synchronizer.sync_account_data()
if sync_result.success:
    print("✅ Sync successful")
else:
    print(f"❌ Sync failed: {sync_result.message}")

# Emergency sync
emergency_result = synchronizer.emergency_sync()
```

### **Testing the System**

#### **Automated Tests**

```bash
# Run all account sync tests
python scripts/run_account_sync_tests.py

# Run unit tests only
python scripts/run_account_sync_tests.py --type unit

# Run integration tests only
python scripts/run_account_sync_tests.py --type integration

# Run with verbose output
python scripts/run_account_sync_tests.py --verbose

# Run with coverage report
python scripts/run_account_sync_tests.py --coverage
```

#### **Manual Testing**

```bash
# Test account synchronization
python scripts/test_account_sync.py

# Test specific components
python -c "
from src.data_providers.binance_exchange import BinanceExchange
exchange = BinanceExchange('your_key', 'your_secret')
print('Connection:', exchange.test_connection())
print('Balances:', len(exchange.get_balances()))
"
```

#### **Test Coverage**

The test suite covers:

- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: End-to-end testing with realistic data flows
- **Error Handling**: Exception scenarios and edge cases
- **Data Validation**: Balance, position, and order synchronization
- **Trade Recovery**: Missing trade detection and recovery
- **Emergency Sync**: Forced synchronization scenarios

#### **Test Structure**

```
tests/
├── test_account_sync.py          # Main test file
├── conftest.py                   # Test configuration and fixtures
└── data/                         # Test data files (if needed)

scripts/
├── run_account_sync_tests.py     # Test runner script
└── test_account_sync.py          # Manual testing script
```

### **Monitoring Synchronization**

```python
# Check sync status in dashboard
from src.monitoring.dashboard import MonitoringDashboard

dashboard = MonitoringDashboard()
sync_status = dashboard.get_sync_status()
print(f"Last sync: {sync_status['last_sync']}")
print(f"Sync health: {sync_status['health']}")
```

## 🚨 Troubleshooting

### **Common Issues**

#### **1. API Connection Failed**
```
❌ Exchange connection failed: Invalid API key
```

**Solution**:
- Verify API key and secret are correct
- Check API key permissions
- Ensure IP whitelist includes your server

#### **2. Balance Discrepancy**
```
Balance discrepancy detected: DB=$1,000.00 vs Exchange=$950.00
```

**Possible Causes**:
- Manual trades on Binance
- Fees or commissions
- Pending orders
- Network delays

**Solution**:
- Check recent trades on Binance
- Verify no manual trading
- Run emergency sync

#### **3. Missing Positions**
```
New position found on exchange: BTCUSDT long 0.001
```

**Possible Causes**:
- Bot shutdown during trade
- Network interruption
- Database corruption

**Solution**:
- Position will be automatically recovered
- Check logs for root cause
- Verify bot stability

### **Debug Commands**

```bash
# Test exchange connection
python scripts/test_account_sync.py

# Check database integrity
python -c "
from src.database.manager import DatabaseManager
db = DatabaseManager()
print('DB connection:', db.test_connection())
print('Active positions:', len(db.get_active_positions()))
"

# Force emergency sync
python -c "
from src.live.account_sync import AccountSynchronizer
# ... initialize and run emergency_sync()
"
```

## 🔮 Future Enhancements

### **Planned Features**

1. **Multi-Exchange Support**
   - Coinbase integration
   - Kraken integration
   - Unified portfolio view

2. **Advanced Recovery**
   - Historical trade reconstruction
   - P&L recalculation
   - Risk analysis

3. **Real-time Sync**
   - WebSocket integration
   - Instant position updates
   - Live balance tracking

4. **Enhanced Monitoring**
   - Sync health dashboard
   - Discrepancy alerts
   - Performance metrics

### **Exchange Migration**

To add a new exchange (e.g., Coinbase):

1. **Create Exchange Implementation**:
```python
class CoinbaseExchange(ExchangeInterface):
    def _initialize_client(self):
        # Initialize Coinbase client
        
    def get_balances(self) -> List[AccountBalance]:
        # Implement Coinbase balance retrieval
        
    def place_order(self, ...) -> Optional[str]:
        # Implement Coinbase order placement
```

2. **Update Trading Engine**:
```python
# In trading engine initialization
if exchange_type == 'coinbase':
    self.exchange_interface = CoinbaseExchange(api_key, api_secret)
elif exchange_type == 'binance':
    self.exchange_interface = BinanceExchange(api_key, api_secret)
```

3. **Test Integration**:
```bash
python scripts/test_account_sync.py --exchange coinbase
```

## 📈 Benefits

### **For Traders**
- **Data Integrity**: Always know your true position
- **Risk Management**: Accurate balance for position sizing
- **Peace of Mind**: No lost trades or positions
- **Audit Trail**: Complete history of all corrections

### **For Developers**
- **Robust Architecture**: Handles failures gracefully
- **Extensible Design**: Easy to add new exchanges
- **Comprehensive Logging**: Full audit trail
- **Test Coverage**: Automated testing suite

## ⚠️ Important Notes

### **Security Considerations**
- Never commit API keys to version control
- Use environment variables for credentials
- Restrict API key permissions
- Monitor API usage for anomalies

### **Rate Limits**
- Binance has rate limits (1200 requests/minute)
- Sync operations respect these limits
- Emergency sync may temporarily exceed limits
- Monitor rate limit usage

### **Data Consistency**
- Sync operations are atomic
- Database transactions ensure consistency
- Rollback on failure
- Audit trail for all changes

### **Performance Impact**
- Sync operations are lightweight
- Minimal impact on trading performance
- Background processing where possible
- Configurable sync intervals

---

**Remember**: The account synchronization system is designed to be robust and self-healing. It will automatically detect and correct discrepancies, ensuring your trading bot always has accurate data about your exchange account.