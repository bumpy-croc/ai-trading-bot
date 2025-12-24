# System Architecture

> **Last Updated**: 2025-12-22
> **Maintainer Note**: This is a living document. Update after major architectural changes or new component additions. Use the `/update-docs` command to keep this in sync.

---

## Overview

The AI Trading Bot is a modular cryptocurrency trading system focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live modes), ML-driven predictions, and Railway deployment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI Trading Bot                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │    Data      │───▶│  Indicators  │───▶│   Strategy   │───▶│ Execution │ │
│  │  Providers   │    │              │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Binance    │    │  RSI, EMA    │    │  ML Models   │    │   Live/   │ │
│  │  Sentiment   │    │  MACD, ATR   │    │   Signals    │    │ Backtest  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                 │                           │
│                                                 ▼                           │
│                                          ┌──────────────┐                   │
│                                          │    Risk      │                   │
│                                          │   Manager    │                   │
│                                          │ (Position    │                   │
│                                          │   Sizing)    │                   │
│                                          └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.11+ | Core implementation |
| Database | PostgreSQL | Trade logging, session tracking |
| ORM | SQLAlchemy | Database abstraction |
| ML Framework | TensorFlow/Keras | Model training |
| ML Inference | ONNX Runtime | Fast model inference |
| Web Framework | Flask | Admin UI, health endpoints |
| Data Processing | pandas, numpy | Time series manipulation |
| Deployment | Railway | Production hosting |
| CLI | Click | Command-line interface |

---

## Directory Structure

```
ai-trading-bot/
├── cli/                          # CLI entry point and commands
│   ├── main.py                   # `atb` command definition
│   └── commands/                 # Subcommands (backtest, live, train, etc.)
│
├── src/                          # Core application code
│   ├── backtesting/              # Vectorized simulation engine
│   │   ├── engine.py             # Main backtest orchestration
│   │   └── portfolio.py          # Simulated portfolio management
│   │
│   ├── config/                   # Configuration system
│   │   ├── loader.py             # Environment variable loading
│   │   ├── constants.py          # Project-wide constants
│   │   └── feature_flags.py      # Feature toggles
│   │
│   ├── data_providers/           # Market data sources
│   │   ├── binance_provider.py   # Binance API integration
│   │   ├── coinbase_provider.py  # Coinbase API integration
│   │   └── caching.py            # Data caching layer
│   │
│   ├── database/                 # Persistence layer
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── manager.py            # DatabaseManager class
│   │   └── admin.py              # Flask-Admin UI
│   │
│   ├── infrastructure/           # Cross-cutting concerns
│   │   ├── logging/              # Centralized logging
│   │   └── runtime/              # Path resolution, secrets
│   │
│   ├── live/                     # Live trading engine
│   │   ├── trading_engine.py     # Real-time execution
│   │   └── strategy_manager.py   # Hot-swap strategy management
│   │
│   ├── ml/                       # ML models directory
│   │   ├── models/               # Trained model registry
│   │   │   └── {SYMBOL}/{TYPE}/{VERSION}/
│   │   └── training_pipeline/    # Training orchestration
│   │
│   ├── prediction/               # Model inference
│   │   ├── registry.py           # Model loading and versioning
│   │   └── predictor.py          # ONNX runtime inference
│   │
│   ├── position_management/      # Position sizing
│   │   ├── position_tracker.py   # Active position tracking
│   │   └── sizers/               # Sizing algorithms
│   │
│   ├── regime/                   # Market regime detection
│   │   └── detector.py           # Regime classification
│   │
│   ├── risk/                     # Risk management
│   │   └── risk_manager.py       # Global risk controls
│   │
│   ├── sentiment/                # Sentiment analysis
│   │   └── adapters/             # Sentiment data adapters
│   │
│   ├── strategies/               # Trading strategies
│   │   ├── ml_basic.py           # Core ML strategy
│   │   ├── ml_adaptive.py        # Regime-adaptive strategy
│   │   ├── ml_sentiment.py       # Sentiment-integrated strategy
│   │   └── components/           # Reusable strategy components
│   │
│   └── trading/                  # Trading utilities
│       ├── symbols/              # Symbol conversion
│       └── shared/               # Shared components
│
├── tests/                        # Test suite
│   ├── unit/                     # Fast, isolated tests
│   ├── integration/              # Database/API tests
│   └── run_tests.py              # Test runner
│
└── docs/                         # Documentation
    ├── architecture.md           # This file
    ├── changelog.md              # Change history
    ├── project_status.md         # Current milestones
    └── execplans/                # Feature execution plans
```

---

## Core Components

### 1. Data Providers (`src/data_providers/`)

Fetches market data from exchanges and applies caching for performance.

**Key Files:**
- `binance_provider.py` - Primary market data source
- `coinbase_provider.py` - Alternative data source
- `caching.py` - Multi-level cache (memory + disk)

**Data Flow:**
```
API Request → Cache Check → Fetch if Miss → Transform → Return DataFrame
```

### 2. ML Training Pipeline (`src/ml/training_pipeline/`)

Trains CNN+LSTM models for price prediction with automatic versioning.

**Pipeline Stages:**
1. **Ingestion** (`ingestion.py`) - Download OHLCV + sentiment data
2. **Feature Engineering** (`features.py`) - Create technical indicators
3. **Dataset Creation** (`datasets.py`) - Build sequences with sliding windows
4. **Model Training** (`models.py`) - CNN+LSTM with mixed precision
5. **Artifacts** (`artifacts.py`) - Save model, ONNX export, metadata

**Model Registry Structure:**
```
src/ml/models/
├── BTCUSDT/
│   ├── basic/
│   │   ├── 2025-10-27_14h_v1/
│   │   │   ├── model.keras      # TensorFlow model
│   │   │   ├── model.onnx       # ONNX export
│   │   │   ├── metadata.json    # Training params
│   │   │   └── feature_schema.json
│   │   └── latest -> 2025-10-27_14h_v1
│   └── sentiment/
└── ETHUSDT/
```

### 3. Strategy System (`src/strategies/`)

Component-based strategy architecture for modularity and testability.

**Design Pattern:**
```python
class Strategy:
    signal_generator: SignalGenerator
    risk_manager: RiskManager
    position_sizer: PositionSizer

    def process_candle(df, index, balance, positions) -> TradingDecision:
        signal = self.signal_generator.generate(df, index)
        risk_check = self.risk_manager.evaluate(signal, positions)
        size = self.position_sizer.calculate(balance, risk_check)
        return TradingDecision(signal, size, risk_metrics)
```

**Available Strategies:**
| Strategy | Description |
|----------|-------------|
| `ml_basic` | Core ML-driven trend following |
| `ml_adaptive` | Regime-aware dynamic strategy |
| `ml_sentiment` | ML with sentiment integration |
| `ensemble_weighted` | Weighted signal combination |
| `momentum_leverage` | Momentum with leverage (experimental) |

### 4. Risk Management (`src/risk/`)

Global risk controls applied across the entire system.

**Key Features:**
- Daily loss limits
- Position size constraints
- Correlation-aware exposure
- Drawdown protection

**Integration Points:**
- Strategies request risk approval before trades
- Engine enforces risk limits on execution
- Adapters share risk state between backtest and live modes

See [Component Risk Integration](architecture/component_risk_integration.md) for detailed adapter guidance.

### 5. Live Trading Engine (`src/live/`)

Real-time execution with safety controls.

**Key Files:**
- `trading_engine.py` - Main execution loop
- `strategy_manager.py` - Hot-swap orchestration

**Safety Features:**
- Paper trading mode (no real orders)
- Health endpoint for monitoring
- Graceful shutdown handling
- Position reconciliation

### 6. Backtesting Engine (`src/backtesting/`)

Vectorized historical simulation for strategy testing.

**Key Files:**
- `engine.py` - Backtest orchestration
- `portfolio.py` - Simulated portfolio

**Features:**
- Vectorized operations for speed
- Realistic slippage/commission modeling
- Multi-symbol support
- Performance metrics calculation

---

## Database Schema

**Core Tables:**

| Table | Purpose |
|-------|---------|
| `trading_sessions` | Session tracking with strategy config |
| `trades` | Complete trade history with P&L |
| `positions` | Active positions with unrealized P&L |
| `account_history` | Balance snapshots |
| `performance_metrics` | Aggregated metrics (Sharpe, drawdown) |

**Migrations:** Managed via Alembic (`alembic upgrade head`)

---

## Configuration System

**Priority Order:**
1. Railway environment variables (production)
2. Environment variables (Docker/CI)
3. `.env` file (local development)

**Essential Variables:**
```env
DATABASE_URL=postgresql://...
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TRADING_MODE=paper  # paper|live
INITIAL_BALANCE=1000
LOG_LEVEL=INFO
```

---

## Recent Architectural Changes

### December 2025
- Optimized ML training pipeline with batch processing improvements (#439)
- Refactored trading bot for better code quality (#437)
- Refactored prediction model registry and usage (#421)

### November 2025
- Enhanced component risk integration with adapters
- Improved strategy versioning system
- Consolidated indicator implementations

---

## Related Documentation

- [Development Workflow](development.md) - Setup and quality gates
- [Backtesting](backtesting.md) - Engine details and CLI usage
- [Live Trading](live_trading.md) - Safety controls and deployment
- [Prediction & Models](prediction.md) - ML registry and inference
- [Component Risk Integration](architecture/component_risk_integration.md) - Adapter patterns
