# Crypto Trend-Following Trading Bot

A modular cryptocurrency trading system focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live), ML-driven models (price and sentiment), PostgreSQL logging, and optional Railway deployment.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![DB](https://img.shields.io/badge/DB-PostgreSQL-informational)](docs/LOCAL_POSTGRESQL_SETUP.md) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

---

## What’s inside

- Pluggable components: data providers, indicators, strategies, ML prediction engine, risk, backtesting, live engine, monitoring
- PostgreSQL-only database with connection pooling and Alembic migrations
- ONNX-based ML models with a central prediction engine
- Real-time monitoring dashboard and an admin UI for database inspection

---

## Quick start

1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Database (PostgreSQL)

- Local (Docker):
```bash
docker-compose up -d postgres
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
# Install pre-commit hooks for code quality
python scripts/setup_pre_commit.py

# Or manually:
python3 -m pip install pre-commit
pre-commit install

# Run linting manually
ruff check . --fix
ruff format .
```

---

## Project structure

```text
src/
  backtesting/       # Vectorised simulation engine
  config/            # Typed configuration loader + constants + feature flags
  data_providers/    # Market & sentiment providers (+ caching wrapper)
  database/          # SQLAlchemy models + DatabaseManager (PostgreSQL-only)
  database_manager/  # Flask-Admin UI for DB inspection
  indicators/        # Technical indicators (pure functions)
  live/              # Live trading engine
  ml/                # Trained models (.onnx/.keras) + metadata
  monitoring/        # Real-time monitoring dashboard (Flask + Socket.IO)
  performance/       # Performance metrics utilities
  prediction/        # Centralized model registry, ONNX runtime, caching
  risk/              # Risk parameters and position sizing utilities
  strategies/        # Built-in strategies (ML basic only)
  utils/             # Shared utilities (paths, symbols, etc.)
```

---

## Key components

- Data providers: `data_providers.BinanceProvider`, `CoinbaseProvider`, `SentiCryptProvider`, `CachedDataProvider`
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

---

## Configuration

Priority order:
1. Railway environment variables (production/staging)
2. Environment variables (Docker/CI/local)
3. .env file (local development)

For MCP tooling, use `mcp.example.json` as a template and create your own `mcp.json` locally with `RAILWAY_API_KEY` set in your environment. Do not commit secrets.

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

- Railway: see `docs/RAILWAY_QUICKSTART.md` and `docs/RAILWAY_DEPLOYMENT_GUIDE.md` (includes DB setup and app start commands)
- Quick helper: `./bin/railway-setup.sh`

---

## Sentiment & models

Sentiment data (SentiCrypt) and ML training are supported. Pretrained models live in `src/ml`. For details and training examples, see:
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
- MCP: use `mcp.example.json` and keep your local `mcp.json` untracked.

## Logging
- Centralized logging via `utils.logging_config.configure_logging()` with env `LOG_LEVEL` and `LOG_JSON`.
- JSON logs default to enabled in production-like environments (Railway or ENV/APP_ENV=production).
- See `docs/LOGGING_GUIDE.md` for structured events, context, and operations guidance.

---

## Disclaimer

This project is for educational purposes only. Trading cryptocurrencies involves significant risk. Use paper trading, test thoroughly, and never risk funds you cannot afford to lose.
<!-- chore: trigger CI test -->
