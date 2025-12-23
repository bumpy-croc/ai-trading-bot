# Crypto Trend-Following Trading Bot

A modular cryptocurrency trading system focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live), ML-driven models (price and sentiment), PostgreSQL logging, and optional Railway deployment.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/) [![DB](https://img.shields.io/badge/DB-PostgreSQL-informational)](docs/database.md) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

> **Requirements**: Python 3.11+ (the repository uses 3.11+ typing features)

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
make deps-dev                # Install dev dependencies (can timeout - see workarounds)

# Alternative: use server dependencies for faster install
make deps-server             # Lighter server/production dependencies
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
atb db verify
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

# Live trading + health endpoint (set PORT or HEALTH_CHECK_PORT to override)
PORT=8000 atb live-health -- ml_basic --paper-trading
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
# Run test suite
atb test unit                # Unit tests only
atb test integration         # Integration tests
atb test all                 # All tests

# Or use pytest directly
pytest -q

# Diagnostics
atb db verify
atb tests heartbeat
atb tests parse-junit tests/reports/unit.xml --label "Unit Tests"
```

8) Development setup

```bash
# Setup development environment
python -m venv .venv && source .venv/bin/activate
make install && make deps-dev

# Run code quality checks
atb dev quality            # Run all: black + ruff + mypy + bandit

# Clean caches and build artifacts
atb dev clean              # Remove .pytest_cache, .ruff_cache, etc.

# Or run individual tools:
black .
ruff check . --fix
python bin/run_mypy.py
bandit -c pyproject.toml -r src
```

---

## Project structure

```text
src/
  backtesting/          # Vectorised simulation engine and utilities
  config/               # Typed configuration loader, constants, feature flags
  dashboards/           # Monitoring, backtesting, and prediction dashboards
  data_providers/       # Market, sentiment, and cache-aware data sources
  database/             # SQLAlchemy models, DatabaseManager, admin UI
  infrastructure/       # Logging config, runtime helpers, cache/secret tooling
  live/                 # Live trading engine, runners, sync utilities
  ml/                   # Training pipeline plus versioned model registry artifacts
  optimizer/            # Parameter optimization and analyzer tooling
  performance/          # Shared performance and diagnostics helpers
  position_management/  # Partial exits, trailing stops, dynamic risk policies
  prediction/           # Prediction engine, feature pipeline, ONNX runners
  regime/               # Market regime detection and analysis
  risk/                 # Global risk manager, exposure controls
  sentiment/            # Sentiment adapters and data mergers
  strategies/           # Component-based strategies and factories
  tech/                 # Indicator math, feature builders, adapters
  trading/              # Trading interfaces plus symbol utilities
  indicators/           # Legacy shim that re-exports `src.tech` (kept for compatibility)
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
- Monitoring dashboard: `src.dashboards.monitoring.dashboard.MonitoringDashboard` (CLI: `atb dashboards run monitoring`)
- Admin UI: `src.database.admin_ui.app:create_app` (Flask-Admin), run `python src/database/admin_ui/app.py`

---

## Documentation

See `docs/README.md` for the full documentation index.

**Key guides:**
- [Backtesting](docs/backtesting.md) – engine internals, CLI usage, and optimisation loop tips
- [Live trading](docs/live_trading.md) – safety controls, account synchronisation, and deployment helpers
- [Data pipeline](docs/data_pipeline.md) – offline cache preloading, download utilities, and cache management
- [Monitoring](docs/monitoring.md) – logging configuration, dashboards, and health endpoints
- [Prediction & models](docs/prediction.md) – model registry, inference workflow, and training controls
- [Configuration](docs/configuration.md) – provider chain, feature flags, and local workflow
- [Database](docs/database.md) – migrations, backups, and Railway operations
- [Development workflow](docs/development.md) – environment bootstrap, quality gates, and strategy versioning

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

Copy `.env.example` to `.env` and fill in your values. See `docs/configuration.md` for detailed configuration options.

---

## Deployment

- Railway: 
  - Quick start: see [Railway deployment quick start](docs/development.md#railway-deployment-quick-start) for environment automation
  - Database operations: see [Database](docs/database.md#railway-deployments)

---

## Sentiment & models

Sentiment data and ML training are supported. Pretrained models live in `src/ml`. For details and training examples, see
[docs/prediction.md](docs/prediction.md).

---

## Getting help
- Open issues for bugs or questions
- Check `docs/README.md` for detailed guides
- Use `atb db verify` to diagnose DB issues quickly

---

## Security
- Do not commit secrets. Use `.env` (see `.env.example`) and environment variables.

## Logging
- Centralized logging via `src.infrastructure.logging.config.configure_logging()` with env `LOG_LEVEL` and `LOG_JSON`.
- JSON logs default to enabled in production-like environments (Railway or ENV/APP_ENV=production).
- See `docs/monitoring.md` for structured events, context, and operations guidance.

---

## Disclaimer

This project is for educational purposes only. Trading cryptocurrencies involves significant risk. Use paper trading, test thoroughly, and never risk funds you cannot afford to lose.
