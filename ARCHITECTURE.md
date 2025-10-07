# AI Trading Bot Architecture

This document provides a comprehensive overview of the AI Trading Bot's architecture, design principles, and system components.

## Table of Contents

- [System Overview](#system-overview)
- [Core Design Principles](#core-design-principles)
- [Architecture Layers](#architecture-layers)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Deployment Architecture](#deployment-architecture)
- [Technology Stack](#technology-stack)

## System Overview

The AI Trading Bot is a modular cryptocurrency trading system designed for:
- **Backtesting** - Historical strategy validation
- **Paper Trading** - Risk-free live testing
- **Live Trading** - Automated real-money trading
- **ML Integration** - ONNX-based prediction models
- **Risk Management** - Position sizing and stop-loss controls

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Trading Bot                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI (atb)   │  │ Dashboards   │  │ Admin UI     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│  ───────┴──────────────────┴──────────────────┴─────────     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │            Trading Engine Layer                     │     │
│  │  ┌──────────────┐         ┌──────────────┐        │     │
│  │  │ Backtesting  │         │ Live Trading │        │     │
│  │  │   Engine     │         │    Engine    │        │     │
│  │  └──────────────┘         └──────────────┘        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │            Strategy & ML Layer                      │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │     │
│  │  │Strategies│  │Prediction│  │  Regime  │        │     │
│  │  │          │  │  Engine  │  │ Detection│        │     │
│  │  └──────────┘  └──────────┘  └──────────┘        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │            Data & Risk Layer                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │     │
│  │  │   Data   │  │   Risk   │  │Position  │        │     │
│  │  │Providers │  │ Manager  │  │  Mgmt    │        │     │
│  │  └──────────┘  └──────────┘  └──────────┘        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Infrastructure Layer                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │     │
│  │  │PostgreSQL│  │  Config  │  │ Logging  │        │     │
│  │  │ Database │  │ Manager  │  │  System  │        │     │
│  │  └──────────┘  └──────────┘  └──────────┘        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Modularity
- **Pluggable components** - Easy to swap implementations
- **Clear interfaces** - Well-defined contracts between modules
- **Separation of concerns** - Each module has a single responsibility

### 2. Safety First
- **Paper trading default** - Safe testing mode
- **Explicit risk acknowledgment** - Required for live trading
- **Multiple safety layers** - Position limits, stop losses, circuit breakers

### 3. Testability
- **High test coverage** - Critical components at 95%+
- **Mock-friendly design** - Easy to test in isolation
- **Integration tests** - End-to-end validation

### 4. Observable
- **Structured logging** - JSON logs for production
- **Real-time monitoring** - Web-based dashboards
- **Database logging** - All trades and decisions persisted

### 5. Performance
- **Vectorized operations** - Pandas/NumPy for data processing
- **Connection pooling** - Efficient database access
- **Caching layers** - Reduce API calls and latency

## Architecture Layers

### 1. Interface Layer

**CLI (`cli/`)**
- Command-line interface using Click
- Commands: backtest, live, dashboards, data, db, train
- Entry point for all user interactions

**Dashboards (`src/dashboards/`)**
- Real-time monitoring dashboard
- Backtesting results visualization
- Market prediction analysis
- Flask + Socket.IO for live updates

**Admin UI (`src/database_manager/`)**
- Flask-Admin interface for database inspection
- Manual data management
- System health checks

### 2. Trading Engine Layer

**Backtesting Engine (`src/backtesting/`)**
- Vectorized historical simulation
- Strategy performance evaluation
- Risk metrics calculation
- Results persistence to database

**Live Trading Engine (`src/live/`)**
- Real-time strategy execution
- Order management and tracking
- Position monitoring
- Account synchronization
- Hot-swapping support for strategies

### 3. Strategy & ML Layer

**Strategies (`src/strategies/`)**
- Base strategy interface
- Built-in strategies (ML Basic, Sentiment, Adaptive, Bull/Bear)
- Component-based architecture
- Entry/exit signal generation
- Position sizing logic

**Prediction Engine (`src/prediction/`)**
- ONNX model registry and loading
- Feature extraction pipeline
- Prediction caching (TTL-based)
- Model metadata management

**Regime Detection (`src/regime/`)**
- Market condition classification
- Trend and volatility analysis
- Hysteresis for stability
- Strategy parameter adjustment

### 4. Data & Risk Layer

**Data Providers (`src/data_providers/`)**
- Exchange interfaces (Binance, Coinbase)
- Sentiment data providers
- Caching wrapper for performance
- Offline mode support

**Risk Manager (`src/risk/`)**
- Position sizing calculations
- Stop-loss placement
- Drawdown monitoring
- Daily risk limits

**Position Management (`src/position_management/`)**
- Correlation analysis
- Dynamic risk adjustment
- MFE/MAE tracking
- Partial position management

### 5. Infrastructure Layer

**Database (`src/database/`)**
- PostgreSQL-only architecture
- SQLAlchemy ORM models
- Connection pooling
- Alembic migrations

**Configuration (`src/config/`)**
- Multi-provider config system
- Railway/Environment/File sources
- Type-safe access methods
- Feature flags

**Logging (`src/utils/logging_config.py`)**
- Structured logging with context
- JSON output for production
- Sensitive data redaction
- Per-logger noise control

## Component Details

### Data Flow: Backtesting

```
1. User runs: atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
                    ↓
2. CLI parses args and loads strategy
                    ↓
3. DataProvider fetches historical data (with caching)
                    ↓
4. Strategy calculates indicators
                    ↓
5. Backtesting engine simulates trades
   - For each candle:
     - Check entry conditions
     - Calculate position size (via RiskManager)
     - Execute trade if conditions met
     - Track P&L and metrics
                    ↓
6. Results saved to database
                    ↓
7. Summary displayed to user
```

### Data Flow: Live Trading

```
1. User runs: atb live ml_basic --symbol BTCUSDT --paper-trading
                    ↓
2. LiveTradingEngine initializes
   - Loads strategy
   - Connects to exchange
   - Initializes database session
                    ↓
3. Main trading loop (every check_interval seconds):
   - Fetch latest market data
   - Update indicators
   - Check strategy signals
   - Calculate position size
   - Execute orders (if conditions met)
   - Update positions
   - Log to database
   - Monitor risk limits
                    ↓
4. Continuous monitoring until stopped
```

### ML Prediction Pipeline

```
1. Strategy requests prediction
                    ↓
2. Check prediction cache
                    ↓
3. If cache miss:
   - Extract features from market data
   - Normalize features
   - Load ONNX model (via PredictionModelRegistry)
   - Run inference
   - Cache result
                    ↓
4. Return prediction with confidence
                    ↓
5. Strategy uses prediction for decision making
```

## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────┐
│         Developer Machine                │
│                                          │
│  ┌──────────────┐    ┌──────────────┐  │
│  │  Trading Bot │ ←→ │  PostgreSQL  │  │
│  │   (Python)   │    │   (Docker)   │  │
│  └──────────────┘    └──────────────┘  │
│         ↓                                │
│  ┌──────────────┐                       │
│  │   Binance    │                       │
│  │   API (Test) │                       │
│  └──────────────┘                       │
└─────────────────────────────────────────┘
```

### Production (Railway)

```
┌─────────────────────────────────────────────────────┐
│                  Railway Platform                    │
│                                                       │
│  ┌─────────────────────┐    ┌──────────────────┐   │
│  │   Trading Service   │ ←→ │   PostgreSQL     │   │
│  │                     │    │    Service       │   │
│  │  - Live engine      │    │                  │   │
│  │  - Strategy exec    │    │  - Trades data   │   │
│  │  - Risk mgmt        │    │  - Performance   │   │
│  └─────────┬───────────┘    │  - Positions     │   │
│            │                └──────────────────┘   │
│            │                                        │
│  ┌─────────┴───────────┐    ┌──────────────────┐   │
│  │  Dashboard Service  │ ←→ │   PostgreSQL     │   │
│  │                     │    │  (same instance) │   │
│  │  - Real-time UI     │    └──────────────────┘   │
│  │  - Monitoring       │                            │
│  └─────────────────────┘                            │
│            ↓                                         │
└────────────┼─────────────────────────────────────────┘
             ↓
      ┌──────────────┐
      │   Binance    │
      │   API (Live) │
      └──────────────┘
```

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.9+ | Primary development language |
| **CLI Framework** | Click | Command-line interface |
| **Web Framework** | Flask | Dashboards and admin UI |
| **Database** | PostgreSQL | Data persistence |
| **ORM** | SQLAlchemy | Database abstraction |
| **Migrations** | Alembic | Schema versioning |
| **Data Processing** | Pandas, NumPy | Vectorized operations |
| **ML Runtime** | ONNX Runtime | Model inference |
| **Testing** | Pytest | Unit and integration tests |
| **Linting** | Ruff | Code quality |
| **Type Checking** | MyPy | Static type analysis |
| **Security** | Bandit | Security scanning |

### Key Libraries

#### Data & ML
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `onnxruntime` - ML model inference
- `scikit-learn` - ML utilities

#### Exchange & APIs
- `python-binance` - Binance API wrapper
- `ccxt` - Multi-exchange support
- `requests` - HTTP client

#### Database
- `sqlalchemy` - ORM and connection pooling
- `psycopg2-binary` - PostgreSQL adapter
- `alembic` - Migration management

#### Web & Monitoring
- `flask` - Web framework
- `flask-socketio` - Real-time updates
- `flask-admin` - Admin interface
- `gunicorn` - WSGI server

#### Testing & Quality
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `ruff` - Fast Python linter
- `mypy` - Type checker
- `bandit` - Security scanner

## Scalability Considerations

### Horizontal Scaling
- **Stateless design** - Easy to run multiple instances
- **Database pooling** - Shared connection pools
- **Message queues** (future) - Distribute order processing

### Vertical Scaling
- **Vectorized operations** - Efficient CPU usage
- **Connection pooling** - Efficient database access
- **Caching layers** - Reduce latency

### Data Scaling
- **Partitioned tables** (future) - Split large tables
- **Archive old data** - Keep active data small
- **Indexed queries** - Optimize common queries

## Security Architecture

### API Key Management
- Environment variables for secrets
- No secrets in code or logs
- Railway encrypted variables

### Database Security
- Connection over SSL/TLS
- Parameterized queries (SQLAlchemy)
- Limited database user permissions

### Application Security
- Input validation on all external data
- Sensitive data redaction in logs
- Rate limiting on API calls
- Security scanning with Bandit

## Monitoring & Observability

### Metrics
- Trade execution success/failure rates
- Strategy performance (win rate, Sharpe ratio)
- System health (CPU, memory, connections)
- API response times and errors

### Logging
- Structured JSON logs in production
- Context-aware logging (strategy, symbol, etc.)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Centralized log aggregation (Railway)

### Alerting
- Critical errors logged and monitored
- Position limit breaches
- API connectivity issues
- Database connection failures

## Future Architecture Enhancements

### Short-term
- [ ] WebSocket support for real-time data
- [ ] Multi-strategy portfolio management
- [ ] Enhanced regime detection with HMM
- [ ] Backtesting performance optimization

### Medium-term
- [ ] Multi-exchange support
- [ ] Advanced order types (OCO, trailing stops)
- [ ] ML model A/B testing framework
- [ ] Real-time alerting system

### Long-term
- [ ] Distributed backtesting across workers
- [ ] Advanced portfolio optimization
- [ ] Custom indicator DSL
- [ ] Machine learning model auto-training

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-07  
**Status:** Current

For implementation details, see individual module READMEs in `src/*/README.md`.
