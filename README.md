# Crypto Trend-Following Trading Bot

A modular cryptocurrency trading system focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live), ML-driven models (price and sentiment), PostgreSQL logging, and optional Railway deployment.

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
python scripts/run_backtest.py ml_basic --symbol BTCUSDT --days 90  # --provider binance|coinbase
```

4) Monitoring dashboard

```bash
# Uses DATABASE_URL; updates every 1 hour by default
python scripts/start_dashboard.py
# open http://localhost:8000
```

5) Live trading (paper by default)

```bash
# Paper trading (safe)
python scripts/run_live_trading.py ml_basic --symbol BTCUSDT --paper-trading

# Live trading (requires explicit confirmation)
python scripts/run_live_trading.py ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks
```

6) Utilities

```bash
# Cache tools
python scripts/cache_manager.py info
python scripts/cache_manager.py list --detailed
python scripts/cache_manager.py clear-old --hours 24

# Control workflow (train, deploy, swap strategy)
python scripts/live_trading_control.py --help
python scripts/live_trading_control.py swap-strategy --strategy ml_basic
```

7) Tests

```bash
pytest -q
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
  strategies/        # Built-in strategies (ML basic/adaptive/sentiment)
  utils/             # Shared utilities (paths, symbols, etc.)
```

---

## Key components

- Data providers: `data_providers.BinanceProvider`, `CoinbaseProvider`, `SentiCryptProvider`, `CachedDataProvider`
- ML prediction: `prediction.models.registry.PredictionModelRegistry` (ONNX), caching in `prediction.utils.caching`
- Strategies: `strategies.ml_basic`, `strategies.ml_with_sentiment`, `strategies.ml_adaptive`
- Backtesting: `backtesting.engine.Backtester` (CLI: `scripts/run_backtest.py`)
- Live engine: `live.trading_engine.LiveTradingEngine` (CLI: `scripts/run_live_trading.py`)
- Risk: `risk.risk_manager.RiskManager`
- Database: `database.manager.DatabaseManager` (PostgreSQL)
- Monitoring: `scripts/start_dashboard.py` → `monitoring.dashboard.MonitoringDashboard`
- Admin UI: `database_manager.app` (Flask-Admin), run `python src/database_manager/app.py`

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

- Railway: see `docs/RAILWAY_QUICKSTART.md` and `docs/RAILWAY_DEPLOYMENT_GUIDE.md` (includes DB setup and app start commands)
- Quick helper: `./bin/railway-setup.sh`

---

## Sentiment & models

Sentiment data (SentiCrypt) and ML training are supported. Pretrained models live in `src/ml`. For details and training examples, see:
- `docs/LIVE_SENTIMENT_ANALYSIS.md`
- `docs/MODEL_TRAINING_AND_INTEGRATION_GUIDE.md`

---

## Disclaimer

This project is for educational purposes only. Trading cryptocurrencies involves significant risk. Use paper trading, test thoroughly, and never risk funds you cannot afford to lose. 