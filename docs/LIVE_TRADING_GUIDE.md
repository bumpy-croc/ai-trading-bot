# 🚀 Live Trading System Guide

## 🎯 **Overview**

This live trading system transforms your backtested strategies into real-time trading bots that can execute actual trades on the Binance exchange. The system is designed with safety, reliability, and professional-grade features.

---

## 🔄 **Backtesting vs Live Trading: Key Differences**

| **Aspect** | **Backtesting** | **Live Trading** |
|------------|----------------|-----------------|
| **Data Source** | Historical CSV files | Real-time API calls |
| **Execution Speed** | Process entire dataset | Continuous monitoring |
| **Market Orders** | Simulated (no real money) | Actual exchange orders |
| **Risk** | Zero financial risk | Real capital at stake |
| **Latency** | Not important | Critical for execution |
| **Error Handling** | Simple logging | Robust recovery systems |
| **Monitoring** | Results at end | 24/7 performance tracking |
| **Sentiment Data** | Historical sentiment | Fresh API calls every 15 minutes |

---

## 🏗️ **System Architecture**

### **Core Components:**

1. **LiveTradingEngine** - Main orchestrator
2. **Real-time Data Provider** - Binance API integration
3. **Strategy Execution** - Your trading strategies
4. **Risk Management** - Position sizing & stop losses
5. **Order Management** - Trade execution & tracking
6. **Monitoring & Alerts** - Performance tracking
7. **Sentiment Integration** - [Live sentiment data](LIVE_SENTIMENT_ANALYSIS.md) as detected by the SentiCrypt API (unfortunately with stale data, so may need to find alternative sources)[[memory:4029677580658624610]]

### **Data Flow:**
```
Real-time Market Data → Strategy Analysis → Risk Checks → Order Execution → Position Tracking
```

---

## 🛡️ **Safety Features**

### **Multiple Safety Layers:**

1. **Paper Trading Mode** (Default) - Test without real money
2. **Explicit Risk Acknowledgment** - Must confirm for live trading
3. **Position Size Limits** - Maximum 50% of balance per position
4. **Stop Loss Protection** - Automatic loss limiting
5. **Time-based Exits** - Close positions after 24 hours
6. **Error Recovery** - Graceful handling of API failures
7. **Emergency Shutdown** - Close all positions on stop

### **Risk Management:**
- **Maximum Drawdown Protection** - Stop if losses exceed threshold
- **Consecutive Error Handling** - Pause trading after repeated failures
- **Rate Limit Compliance** - Prevent API blocking
- **Balance Monitoring** - Track real-time P&L

---

## 🚀 **Getting Started**

### **1. Prerequisites**

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Binance API credentials
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

### **2. Paper Trading (Recommended First)**

```bash
# Safe paper trading - no real money
python scripts/run_live_trading.py ml_basic --symbol BTC-USD --paper-trading

# With sentiment analysis
python scripts/run_live_trading.py ml_with_sentiment --symbol BTC-USD --paper-trading --use-sentiment

# Custom configuration
python scripts/run_live_trading.py ml_basic --balance 5000 --max-position 0.05 --check-interval 30
```

### **3. Live Trading (Advanced)**

```bash
# DANGER: Real money trading
python scripts/run_live_trading.py ml_basic --symbol BTC-USD --live-trading --i-understand-the-risks

# The system will ask for additional confirmation
```

---

## ⚙️ **Configuration Options**

### **Basic Parameters:**
```bash
--symbol BTC-USD              # Trading pair
--timeframe 1h                # Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
--balance 10000              # Initial balance
--max-position 0.1           # Max 10% of balance per position
--check-interval 60          # Check every 60 seconds
```

### **Risk Management:**
```bash
--risk-per-trade 0.01        # Risk 1% per trade
--max-risk-per-trade 0.02    # Maximum 2% risk
--max-drawdown 0.2           # Stop if 20% drawdown
```

### **Advanced Features:**
```bash
--use-sentiment              # Enable sentiment analysis
--webhook-url "https://..."  # Slack/Discord alerts
--log-trades                 # Save trade history
--no-cache                   # Disable data caching
```

---

## 📊 **Live Trading Features**

### **Real-time Monitoring:**
- **Current Balance** - Live P&L tracking
- **Active Positions** - Open trades with unrealized P&L
- **Performance Metrics** - Win rate, drawdown, Sharpe ratio
- **Market Status** - Price updates and sentiment scores

### **Position Management:**
- **Automatic Stop Loss** - Configurable risk protection
- **Take Profit Targets** - Lock in gains
- **Position Sizing** - Kelly criterion or fixed percentage
- **Multi-asset Support** - Trade multiple pairs simultaneously

### **Sentiment Integration:**
- **Live Sentiment Data** - Fresh API calls every 15 minutes
- **Confidence Boosting** - Higher confidence with fresh sentiment
- **Fallback Mechanisms** - Graceful degradation if sentiment fails
- **Historical Comparison** - See sentiment changes over time

---

## 🔍 **Monitoring & Alerts**

### **Console Output:**
```
2025-01-27 15:30:00 - Status: BTCUSDT @ $95,234.56 | Balance: $10,245.67 | Positions: 1 | Unrealized: +$145.67 | Trades: 15 (67% win)
```

### **Log Files:**
- **`live_trading_YYYYMMDD.log`** - Detailed trading logs
- **`logs/trades/trades_YYYYMM.json`** - Trade history in JSON format

### **Webhook Alerts:**
```json
{
  "text": "🤖 Trading Bot: Position Opened: BTCUSDT long @ $95,234.56",
  "timestamp": "2025-01-27T15:30:00Z"
}
```

---

## 📈 **Strategy Integration**

### **Supported Strategies:**
- **`