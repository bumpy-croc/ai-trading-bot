---
description: Trading Bot Core Information & Essential Guidelines
globs: 
alwaysApply: true
---

# 🤖 Trading Bot Core Information

## 🎯 System Overview

Sophisticated cryptocurrency trading system with **trend-following risk containment**. Supports backtesting, live trading, ML models, and multiple data sources.

**Core Philosophy**: Trade with the trend, not against it. Protect capital above all else.

---

## 🏗️ Core Architecture

### **System Components**
```
Data Providers → Indicators → Strategies → Risk Manager → Execution Layer
```

### **Key Directories**
- `src/strategies/` - Trading strategy implementations
- `src/live/` - Live trading engine (most important component)
- `src/backtesting/` - Historical simulation engine
- `src/risk/` - Risk management system
- `src/data_providers/` - Market & sentiment data adapters
- `ml/` - Trained ML models (.h5/.keras/.onnx)
- `scripts/` - CLI utilities & automation
- `tests/` - Comprehensive test suite

---

## 🚀 Live Trading Engine (Core Component)

### **Key Features**
- **Real-time data streaming** from Binance API
- **Strategy execution** with ML model integration
- **Risk management** with position sizing & stop-losses
- **Sentiment data integration** (SentiCrypt API)
- **Database logging** for all trades & positions
- **Graceful error handling** & recovery
- **Hot-swapping strategies** without stopping

### **Safety Features**
- **Paper Trading Mode** (default) - No real money at risk
- **Explicit Risk Confirmation** - Must confirm for live trading
- **Position Size Limits** - Maximum 10% of balance per position
- **Stop Loss Protection** - Automatic loss limiting

---

## 📊 Available Strategies

- **Adaptive Strategy**: Adaptive EMA crossover with market regime detection
- **Enhanced Strategy**: Multi-indicator confluence (RSI + EMA + MACD)
- **ML Basic Strategy**: Uses ML price predictions for entry/exit decisions
- **ML with Sentiment Strategy**: Combines ML predictions with sentiment analysis
- **High Risk High Reward Strategy**: Aggressive trading with higher risk tolerance

---

## 🧠 ML Model Integration

### **Model Types**
1. **Price Prediction Models** (`btcusdt_price.*`) - 5 features (OHLCV)
2. **Sentiment-Enhanced Models** (`btcusdt_sentiment.*`) - 13 features (5 price + 8 sentiment)

### **Live Trading Integration**
- **Real-time ONNX inference**
- **Confidence-based position sizing**
- **Graceful fallback** when sentiment data unavailable

---

## 🛡️ Risk Management

### **Default Risk Parameters**
```python
base_risk_per_trade: float = 0.02      # 2% risk per trade
max_risk_per_trade: float = 0.03       # 3% maximum risk per trade
max_position_size: float = 0.25        # 25% maximum position size
max_daily_risk: float = 0.06           # 6% maximum daily risk
max_drawdown: float = 0.20             # 20% maximum drawdown
```

---

## 🔧 Essential Commands

### **Quick Development**
```bash
# Quick backtest (development)
python scripts/run_backtest.py adaptive --days 30 --no-db

# Paper trading (safe)
python scripts/run_live_trading.py adaptive --paper-trading

# Quick tests
python tests/run_tests.py smoke
```

### **Production**
```bash
# Production backtest
python scripts/run_backtest.py ml_with_sentiment --days 365

# Live trading (requires confirmation)
python scripts/run_live_trading.py ml_with_sentiment --live-trading --i-understand-the-risks

# Monitor dashboard
python scripts/start_dashboard.py
```

### **Safety**
```bash
# Health check
python scripts/health_check.py

# Critical tests
python tests/run_tests.py critical

# Emergency stop
python scripts/run_live_trading.py --stop
```

---

## 🎯 Natural Language Commands

When you say these phrases, I'll run the corresponding commands:

### **Testing**
- "run smoke tests" → `python tests/run_tests.py smoke`
- "run unit tests" → `python tests/run_tests.py unit`
- "run critical tests" → `python tests/run_tests.py critical`
- "run all tests" → `python tests/run_tests.py all`

### **Backtesting**
- "run backtest" → `python scripts/run_backtest.py adaptive --days 30 --no-db`
- "run backtest for [strategy]" → `python scripts/run_backtest.py [strategy] --days 30 --no-db`
- "run production backtest" → `python scripts/run_backtest.py adaptive --days 30`

### **Live Trading**
- "start paper trading" → `python scripts/run_live_trading.py adaptive --paper-trading`
- "start live trading" → `python scripts/run_live_trading.py adaptive --live-trading --i-understand-the-risks`
- "start dashboard" → `python scripts/start_dashboard.py`

### **Health & Monitoring**
- "check health" → `python scripts/health_check.py`
- "check positions" → `python scripts/health_check.py --positions`
- "check cache" → `python scripts/cache_manager.py --check`

---

## 📚 Detailed Guides (Fetch as Needed)

For detailed implementation guides, use:
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["ml-models"])` - ML model training & integration
- `fetch_rules(["commands"])` - Complete command reference

---

**Remember**: This is real money. Always validate changes thoroughly. When in doubt, backtest more. 🛡️