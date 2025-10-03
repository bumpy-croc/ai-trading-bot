# Crypto Trend-Following Trading Bot

A modular cryptocurrency trading system focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live), ML-driven models (price and sentiment), PostgreSQL logging, and optional Railway deployment.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![DB](https://img.shields.io/badge/DB-PostgreSQL-informational)](docs/LOCAL_POSTGRESQL_SETUP.md) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

---

## What's inside

- Pluggable components: data providers, indicators, strategies, ML prediction engine, risk, backtesting, live engine, monitoring
- PostgreSQL-only database with connection pooling and Alembic migrations
- ONNX-based ML models with a central prediction engine
- Real-time monitoring dashboard and an admin UI for database inspection
- **Offline backtesting support** with pre-loaded cache for air-gapped environments

---

## Quick start

1) Install

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install CLI and dependencies
make install                 # Install CLI (atb) + upgrade pip
make deps                    # Install dev dependencies (can timeout - see workarounds)

# Alternative: use server dependencies for faster install
make deps-server             # Lighter production dependencies
```

**Note**: If pip install times out due to large packages (TensorFlow ~500MB), try:
```bash
.venv/bin/pip install --timeout 1000 <package>
```

2) Database (PostgreSQL)

- Local (Docker):
```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
```
- Verify:
```bash
python scripts/verify_database_connection.py
# optional migrations (tables are auto-created if missing)
alembic upgrade head
```

3) Backtest

```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90

# For offline/air-gapped environments, pre-load cache with historical data
atb data preload-offline --years-back 10                        # Pre-load 10 years of data
```

4) Monitoring dashboard

```bash
atb dashboards list
atb dashboards run monitoring --port 8000
```

5) Live trading (paper by default)

```bash
# Paper trading (safe)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Live trading (requires explicit confirmation)
atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks

# Live trading + health endpoint
atb live-health --port 8000 -- ml_basic --paper-trading
```

6) Utilities

```bash
# Cache tools
atb data cache-manager info
atb data cache-manager list --detailed
atb data cache-manager clear-old --hours 24

# Prefill cache for faster backtests
atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 4

# Control workflow (train, deploy, swap strategy)
atb live-control train --symbol BTCUSDT --days 365 --auto-deploy
atb live-control swap-strategy --strategy ml_basic

# Populate dummy data (for dashboards/testing)
atb data populate-dummy --trades 100 --confirm
```

7) Tests

```bash
pytest -q

# Diagnostics
atb db verify
atb tests heartbeat
```

8) Development setup

```bash
# Setup development environment
python -m venv .venv && source .venv/bin/activate
make install && make deps

# Run code quality checks
make code-quality  # ruff + black + mypy + bandit

# Or run individually:
ruff check . --fix
ruff format .
python bin/run_mypy.py
bandit -c pyproject.toml -r src
```

---

## Project structure

```text
src/
  backtesting/       # Vectorised simulation engine
  config/            # Typed configuration loader + constants + feature flags  
  dashboards/        # Web-based monitoring and analysis dashboards
  data/              # Data management and caching utilities
  data_providers/    # Market & sentiment providers (+ caching wrapper)
  database/          # SQLAlchemy models + DatabaseManager (PostgreSQL-only)
  database_manager/  # Flask-Admin UI for DB inspection
  examples/          # Minimal runnable examples demonstrating core features
  indicators/        # Technical indicators (pure functions)
  live/              # Live trading engine
  ml/                # Trained models (.onnx/.keras) + metadata
  monitoring/        # Real-time monitoring dashboard (Flask + Socket.IO)
  optimizer/         # Parameter optimization and strategy tuning
  performance/       # Performance metrics utilities
  position_management/  # Position sizing and portfolio management
  prediction/        # Centralized model registry, ONNX runtime, caching
  regime/            # Market regime detection and analysis
  risk/              # Risk parameters and position sizing utilities
  strategies/        # Built-in strategies (ML basic, sentiment, adaptive, bull/bear)
  trading/           # Core trading interfaces and shared functionality
  utils/             # Shared utilities (paths, symbols, etc.)
```

---

## Key components

- Data providers: `data_providers.BinanceProvider`, `CoinbaseProvider`, `CachedDataProvider`
- ML prediction: `prediction.models.registry.PredictionModelRegistry` (ONNX), caching in `prediction.utils.caching`
- Strategies: `strategies.ml_basic`
- Backtesting: `backtesting.engine.Backtester` (CLI: `atb backtest`)
- Live engine: `live.trading_engine.LiveTradingEngine` (CLI: `atb live`, `atb live-health`)
- Risk: `risk.risk_manager.RiskManager`
- Database: `database.manager.DatabaseManager` (PostgreSQL)
- Monitoring: `dashboards.monitoring.MonitoringDashboard` (CLI: `atb dashboards run monitoring`)
- Admin UI: `database_manager.app` (Flask-Admin), run `python src/database_manager/app.py`

---

## Documentation

See `docs/README.md` for the full documentation index.

**Key Guides:**
- [Offline Cache Pre-loading](docs/OFFLINE_CACHE_PRELOADING.md) - Pre-load data for air-gapped environments
- [Backtest Guide](docs/BACKTEST_GUIDE.md) - Comprehensive backtesting documentation
- [Live Trading Guide](docs/LIVE_TRADING_GUIDE.md) - Live trading setup and configuration

---

## Configuration

Priority order:
1. Railway environment variables (production/staging)
2. Environment variables (Docker/CI/local)
3. .env file (local development)

Minimal variables:

```env
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
TRADING_MODE=paper
INITIAL_BALANCE=1000
```

See `docs/CONFIGURATION_SYSTEM_SUMMARY.md`.

---

## Deployment

- Railway: 
  - Quick start: see [Railway Quickstart](docs/RAILWAY_QUICKSTART.md)
  - Database setup: see [Railway Database Centralization Guide](docs/RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md)

---

## Sentiment & models

Sentiment data and ML training are supported. Pretrained models live in `src/ml`. For details and training examples, see:
- `docs/LIVE_SENTIMENT_ANALYSIS.md`
- `docs/MODEL_TRAINING_AND_INTEGRATION_GUIDE.md`

---

## Getting help
- Open issues for bugs or questions
- Check `docs/README.md` for detailed guides
- Use `atb db verify` to diagnose DB issues quickly

---

## Security
- Do not commit secrets. Use `.env` (see `.env.example`) and environment variables.

## Logging
- Centralized logging via `src.utils.logging_config.configure_logging()` with env `LOG_LEVEL` and `LOG_JSON`.
- JSON logs default to enabled in production-like environments (Railway or ENV/APP_ENV=production).
- See `docs/LOGGING_GUIDE.md` for structured events, context, and operations guidance.

---

## Disclaimer

This project is for educational purposes only. Trading cryptocurrencies involves significant risk. Use paper trading, test thoroughly, and never risk funds you cannot afford to lose.
<!-- chore: trigger CI test -->
