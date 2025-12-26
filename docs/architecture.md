# System Architecture

> **Last Updated**: 2025-12-26
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
│   ├── engines/                  # Trading engines (backtest + live)
│   │   ├── backtest/             # Vectorized simulation engine
│   │   │   ├── engine.py         # Main backtest orchestration
│   │   │   ├── portfolio.py      # Simulated portfolio management
│   │   │   └── execution/        # Entry/exit handling
│   │   │
│   │   ├── live/                 # Live trading engine
│   │   │   ├── trading_engine.py # Real-time execution
│   │   │   ├── strategy_manager.py # Hot-swap strategy management
│   │   │   └── execution/        # Entry/exit handling
│   │   │
│   │   └── shared/               # Shared engine logic
│   │       ├── models.py         # Unified Position, Trade types
│   │       ├── cost_calculator.py # Fee/slippage calculation
│   │       ├── dynamic_risk_handler.py # Dynamic risk adjustments
│   │       ├── performance_tracker.py  # Unified metrics tracking
│   │       ├── partial_operations_manager.py # Partial exit/scale-in
│   │       ├── policy_hydration.py # Runtime policy extraction
│   │       ├── risk_configuration.py # Risk config merging
│   │       └── trailing_stop_manager.py # Trailing stop updates
│   │
│   ├── infrastructure/           # Cross-cutting concerns
│   │   ├── logging/              # Centralized logging
│   │   └── runtime/              # Path resolution, secrets
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
│   ├── performance/              # Performance measurement
│   │   ├── metrics.py            # Pure metric calculations
│   │   └── tracker.py            # Performance tracking
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

### 5. Live Trading Engine (`src/engines/live/`)

Real-time execution with safety controls.

**Key Files:**
- `trading_engine.py` - Main execution loop
- `strategy_manager.py` - Hot-swap orchestration
- `execution/entry_handler.py` - Entry signal processing
- `execution/exit_handler.py` - Exit signal processing

**Safety Features:**
- Paper trading mode (no real orders)
- Health endpoint for monitoring
- Graceful shutdown handling
- Position reconciliation

### 6. Backtesting Engine (`src/engines/backtest/`)

Vectorized historical simulation for strategy testing.

**Key Files:**
- `engine.py` - Backtest orchestration
- `portfolio.py` - Simulated portfolio
- `execution/entry_handler.py` - Entry signal processing
- `execution/exit_handler.py` - Exit signal processing

**Features:**
- Vectorized operations for speed
- Realistic slippage/commission modeling
- Multi-symbol support
- Performance metrics calculation

### 7. Shared Engine Modules (`src/engines/shared/`)

Unified logic extracted from both backtest and live engines to ensure consistency.

**Key Modules:**

| Module | Purpose |
|--------|---------|
| `models.py` | Unified `Position`, `Trade`, `PositionSide` types |
| `cost_calculator.py` | Fee and slippage calculation |
| `dynamic_risk_handler.py` | Dynamic risk adjustments during drawdowns |
| `partial_operations_manager.py` | Partial exit and scale-in logic |
| `policy_hydration.py` | Extract policies from runtime decisions |
| `risk_configuration.py` | Merge strategy risk overrides with base config |
| `trailing_stop_manager.py` | Trailing stop update calculations |

**Benefits:**
- Single source of truth for trading logic
- Consistent behavior between backtest and live
- Easier testing and maintenance
- Reduced code duplication (eliminated ~500 lines)

### 8. Performance Tracking (`src/performance/`)

Unified performance measurement system shared across backtesting, live trading, and analytics.

**Architecture Overview:**

The performance tracking system follows a clean three-layer architecture:

```
┌─────────────────────────────────────────────────────┐
│  Engine Layer (Backtest/Live)                       │
│  - Orchestration logic                              │
│  - Engine-specific workflows                        │
│  - Persistence decisions                            │
└─────────────────────────────────────────────────────┘
                      ↓ uses
┌─────────────────────────────────────────────────────┐
│  Performance Tracker (src/performance/tracker.py)   │
│  - State management (trades, balance history)       │
│  - Metric aggregation                               │
│  - Thread-safe updates                              │
└─────────────────────────────────────────────────────┘
                      ↓ uses
┌─────────────────────────────────────────────────────┐
│  Metrics (src/performance/metrics.py)               │
│  - Pure calculation functions                       │
│  - Sharpe, Sortino, Calmar, VaR, etc.              │
│  - No side effects, fully testable                  │
└─────────────────────────────────────────────────────┘
```

**Key Files:**
- `metrics.py` - Pure metric calculation functions (Sharpe, Sortino, Calmar, VaR, expectancy)
- `tracker.py` - Stateful performance tracking with thread safety

**Tracked Metrics (30+ total):**

| Category | Metrics |
|----------|---------|
| **Returns** | Total return, annualized return, CAGR |
| **Risk-Adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio |
| **Risk** | Max drawdown, current drawdown, VaR (95%) |
| **Trade Quality** | Win rate, profit factor, expectancy, avg win/loss |
| **Efficiency** | Trade duration, consecutive streaks, trades per day |
| **Costs** | Total fees, total slippage |

**Design Principles:**
- **Metric Parity**: Both backtest and live engines calculate identical metrics using the same code
- **Separation of Concerns**: Calculation (metrics.py) separate from state management (tracker.py) separate from persistence (engines)
- **Reusability**: Used by engines, strategies, dashboards, and analysis tools
- **Thread Safety**: Lock-protected state updates for concurrent access in live trading

**Usage Example:**
```python
from src.performance.tracker import PerformanceTracker

# Initialize
tracker = PerformanceTracker(initial_balance=10000)

# Record trades
tracker.record_trade(trade, fee=2.5, slippage=0.5)
tracker.update_balance(balance=10050, timestamp=datetime.now())

# Get comprehensive metrics
metrics = tracker.get_metrics()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Sortino: {metrics.sortino_ratio:.2f}")
print(f"Win Rate: {metrics.win_rate * 100:.1f}%")
```

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
- **Performance Tracker Integration**: Unified performance tracking in `src/performance/` module
  - Enhanced tracker with 30+ metrics (Sharpe, Sortino, Calmar, VaR, expectancy, streaks)
  - Integrated into both backtest and live engines for metric parity
  - Added database schema migration for new performance columns
  - Comprehensive testing and validation framework
- **Engine Consolidation** (#454): Moved `src/backtesting/` and `src/live/` under `src/engines/` with shared modules
- Created `src/engines/shared/` with unified logic for both engines
- Extracted 8 shared modules eliminating ~500 lines of duplicate code
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
