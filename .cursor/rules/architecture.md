---
description: Trading Bot Core Architecture & System Overview
globs: 
alwaysApply: false
---

# 🏗️ Trading Bot Architecture

## System Overview
Cryptocurrency trading system with trend-following risk containment. Supports backtesting, live trading, ML models, and multiple data sources.

**Philosophy**: Trade with the trend, not against it. Protect capital above all else.

---

## System Architecture

### High-Level Data Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│ Indicator Layer │───▶│ Strategy Layer  │
│                 │    │                 │    │                 │
│ • Binance API   │    │ • RSI, EMA      │    │ • Signal Gen    │
│ • Sentiment API │    │ • Bollinger     │    │ • ML Models     │
│ • CSV Cache     │    │ • MACD, ATR     │    │ • Risk Logic    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Risk Manager   │    │ Execution Layer │
                       │                 │    │                 │
                       │ • Position Size │    │ • Live Trading  │
                       │ • Stop Loss     │    │ • Backtesting   │
                       │ • Exposure      │    │ • Order Mgmt    │
                       └─────────────────┘    └─────────────────┘
```

### Component Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Trading Bot                           │
├─────────────────────────────────────────────────────────────────┤
│  Data Providers  │  Indicators  │  Strategies   │  Risk Mgmt    │
│  ┌─────────────┐ │ ┌──────────┐ │ ┌──────────┐  │ ┌──────────┐  │
│  │ Binance     │ │ │ RSI      │ │ │ Adaptive │  │ │ Position │  │
│  │ SentiCrypt  │ │ │ EMA      │ │ │ Enhanced │  │ │ Sizing   │  │
│  │ CryptoComp  │ │ │ MACD     │ │ │ ML Basic │  │ │ Stop Loss│  │
│  │ Cached      │ │ │ ATR      │ │ │ ML+Sent  │  │ │ Exposure │  │
│  └─────────────┘ │ └──────────┘ │ └──────────┘  │ └──────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Execution Layer  │  Database   │  Monitoring   │  ML Models    │
│  ┌─────────────┐  │ ┌─────────┐ │ ┌──────────┐  │ ┌──────────┐  │
│  │ Live Trading│  │ │ Trades  │ │ │ Dashboard│  │ │ ONNX     │  │
│  │ Backtesting │  │ │ Positions│ │ │ Metrics  │  │ │ Keras    │  │
│  │ Paper Trade │  │ │ Sessions│ │ │ Alerts   │  │ │ Metadata │  │
│  └─────────────┘  │ └─────────┘ │ └──────────┘  │ └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Live Trading Engine

### Key Features
- Real-time data streaming from Binance API
- Strategy execution with ML model integration
- Risk management with position sizing & stop-losses
- Sentiment data integration (SentiCrypt API)
- Database logging for all trades & positions
- Graceful error handling & recovery
- Hot-swapping strategies without stopping
- Performance monitoring & alerts

### Safety Features
- **Paper Trading Mode** (default) - No real money at risk
- **Explicit Risk Acknowledgment** - Must confirm for live trading
- **Position Size Limits** - Maximum 10% of balance per position
- **Stop Loss Protection** - Automatic loss limiting
- **Error Recovery** - Graceful handling of API failures

---

## Machine Learning Integration

### Model Types
1. **Price Prediction Models** (`btcusdt_price.*`)
   - Input: 120 time steps × 5 features (OHLCV)
   - Architecture: CNN + LSTM + Dense layers

2. **Sentiment-Enhanced Models** (`btcusdt_sentiment.*`)
   - Input: 120 time steps × 13 features (5 price + 8 sentiment)
   - Architecture: CNN + LSTM + Dense layers

### Live Trading Integration
- Real-time ONNX inference
- Confidence-based position sizing
- Graceful fallback when sentiment data unavailable

---

## Strategy System

### Available Strategies
- `MlBasic`, `MlAdaptive`, `MlWithSentiment` (see `src/strategies/`)

### Strategy Base Class
All strategies implement:
- `calculate_indicators()` - Strategy-specific indicators
- `check_entry_conditions()` - Entry signal logic
- `check_exit_conditions()` - Exit signal logic
- `calculate_position_size()` - Position sizing logic

---

## Database Architecture

### Core Tables
- **trading_sessions**: Track trading sessions with strategy configuration
- **trades**: Complete trade history with entry/exit prices and P&L
- **positions**: Active positions with real-time unrealized P&L
- **account_history**: Balance snapshots for performance tracking
- **performance_metrics**: Aggregated metrics (win rate, Sharpe ratio, drawdown)
- **strategy_executions**: Detailed strategy decision logs

### Database Features
- **ACID Transactions**: Critical for financial data integrity
- **Connection Pooling**: Efficient resource management
- **Indexed Queries**: Fast performance for time-series data
- **JSONB Support**: Flexible storage for strategy configurations

---

## Data Flow

### Live Trading Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Binance API │───▶│ Data Cache  │───▶│ Indicators  │───▶│ Strategy    │
│             │    │             │    │             │    │             │
│ Real-time   │    │ Reduce API  │    │ RSI, EMA,   │    │ Signal Gen  │
│ OHLCV Data  │    │ Calls       │    │ MACD, ATR   │    │ ML Predict  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ SentiCrypt  │───▶│ Sentiment   │───▶│ Risk Mgmt   │◀───│ Position    │
│ API         │    │ Processing  │    │             │    │ Sizing      │
│             │    │             │    │ Stop Loss   │    │             │
│ Sentiment   │    │ Feature Eng │    │ Exposure    │    │ Entry/Exit  │
│ Scores      │    │             │    │ Limits      │    │ Logic       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Database    │◀───│ Order Exec  │◀───│ Position    │◀───│ Trade       │
│             │    │             │    │ Management  │    │ Execution   │
│ Log Trades  │    │ Binance API │    │ P&L Update  │    │             │
│ Track P&L   │    │             │    │ Stop Loss   │    │ Real Orders │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### ML Model Training Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Historical  │───▶│ Feature     │───▶│ Model       │───▶│ Model       │
│ Price Data  │    │ Engineering │    │ Training    │    │ Validation  │
│             │    │             │    │             │    │             │
│ OHLCV +     │    │ Normalize   │    │ CNN + LSTM  │    │ Backtest    │
│ Sentiment   │    │ Features    │    │ + Dense     │    │ Performance │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Model       │◀───│ ONNX Export │◀───│ Model       │◀───│ Performance │
│ Deployment  │    │             │    │ Selection   │    │ Analysis    │
│             │    │ Optimize    │    │             │    │             │
│ Live        │    │ Inference   │    │ Best Model  │    │ Sharpe,     │
│ Trading     │    │ Speed       │    │ Selection   │    │ Drawdown    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

**For detailed implementation guides, use:**
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["ml-models"])` - ML model training & integration
- `fetch_rules(["commands"])` - Complete command reference