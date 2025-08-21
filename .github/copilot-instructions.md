# AI Trading Bot - Copilot Instructions

## Repository Overview

This is a modular cryptocurrency trading bot focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live), ML-driven predictions (price and sentiment), PostgreSQL logging, and Railway deployment.

**Size & Tech Stack:**
- Medium-large Python project (~50+ modules)
- Python 3.9+ required, primarily Python 3.11/3.12 tested
- Key dependencies: Flask, SQLAlchemy, TensorFlow/ONNX, pandas, scikit-learn, python-binance, ccxt
- Database: PostgreSQL only (no SQLite fallback)
- Runtime: CLI tool (`atb`) with web dashboards
- Deployment: Railway (primary), Docker support
- Target: Long-running trading strategies with real-time monitoring

## Build & Development Setup

### Essential Commands (Always run in this order)

```bash
# 1. Bootstrap environment
make venv                    # Create .venv (required first)
make install                 # Install CLI (atb) + upgrade pip
make deps                    # Install dev dependencies (can timeout - see workarounds)

# 2. Database setup (PostgreSQL required for most functionality)
docker compose up -d postgres  # Start local PostgreSQL (note: 'docker compose', not 'docker-compose')
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
python scripts/verify_database_connection.py  # Verify connection (requires sqlalchemy)

# 3. Test installation
atb --help                   # Verify CLI works
make test                    # Run test suite (requires DB)
```

### Build System Details

- **Virtual Environment**: Always use `.venv` via `make venv` first
- **Editable Install**: CLI installed via `pip install -e .` in Makefile
- **Dependencies**: Two requirement files:
  - `requirements.txt`: Full dev environment (includes TensorFlow - can timeout)
  - `requirements-server.txt`: Lighter production build
- **CLI Tool**: `atb` command provides all functionality

### Common Build Issues & Workarounds

**⚠️ CRITICAL: Pip Install Timeouts (Very Common)**
```bash
# PyPI timeouts are frequent due to large packages (TensorFlow ~500MB)
# If ANY pip install times out, try:
.venv/bin/pip install --timeout 1000 <package>
# Or use server requirements only:
make deps-server             # Lighter requirements
```

**Dependency Installation Timeout:**
```bash
# If 'make deps' times out (common due to TensorFlow size), install server deps first:
make deps-server             # Lighter requirements
# Then manually install missing dev dependencies as needed:
# .venv/bin/pip install pytest pytest-mock ruff black mypy
```

**CLI Import Errors:**
- Ensure `make install` completed successfully
- CLI requires core dependencies (ccxt, python-binance) to import
- Run in venv: `.venv/bin/atb --help`

**Database Connection:**
- PostgreSQL is required - no fallback database
- Always verify connection before running commands
- Migration auto-runs, but can use `alembic upgrade head`

## Testing & Validation

### Test Commands
```bash
make test                    # Run all tests (pytest -n 4)
pytest -q                   # Quick test run
pytest -m integration       # Integration tests (requires DB)
pytest -m "not integration" # Unit tests only
pytest tests/unit/test_backtesting.py  # Specific test file
```

### Test Structure & Markers
- **Test Organization**: `tests/unit/` and `tests/integration/`
- **Key Markers**: `integration`, `live_trading`, `risk_management`, `database`, `fast`, `slow`
- **Parallel Testing**: CI uses 4-way split testing (`--splits 4 --group N`)
- **Database**: Uses testcontainers[postgresql] for isolation

### CI/CD Pipeline Validation Steps
1. **Quality Checks** (PR): ruff (lint), black (format), mypy (types), bandit (security)
2. **Unit Tests** (PR): Split 4-way parallel execution, 30min timeout
3. **Integration Tests** (nightly): Full DB + external provider tests

**Reproducing CI Locally:**
```bash
# Quality checks
ruff check .
black --check .
mypy --config-file mypy.ini src
bandit -c pyproject.toml -r src

# Tests (match CI environment)
PYTHONPATH=./src:. pytest -m "not integration" -n 2 --dist=loadgroup
```

### Coverage Requirements
- Overall: 85% minimum
- Live Trading Engine: 95%
- Risk Management: 95%
- Strategies: 85%
- Generate: `pytest --cov=src --cov-report=html`

## Project Architecture & Key Files

### Source Structure (`src/`)
```
backtesting/       # Vectorized simulation engine
config/            # Typed configuration + constants + feature flags
data_providers/    # Market & sentiment providers + caching
database/          # SQLAlchemy models + DatabaseManager
live/              # Live trading engine
ml/                # Trained models (.onnx/.keras) + metadata
monitoring/        # Real-time dashboards (Flask + SocketIO)
prediction/        # Model registry, ONNX runtime, caching
risk/              # Risk parameters & position sizing
strategies/        # Built-in strategies (ml_basic)
utils/             # Shared utilities
```

### Key Configuration Files
- `pyproject.toml`: Build config, ruff/black settings
- `pytest.ini`: Test configuration, markers, paths
- `mypy.ini`: Type checking configuration
- `.coveragerc`: Coverage settings
- `docker-compose.yml`: Local PostgreSQL setup
- `Makefile`: Build automation
- `railway.json`: Railway deployment config

### Entry Points & Commands
- **CLI**: `atb` command (installed via setup.py)
- **Main Commands**: `backtest`, `live`, `live-health`, `dashboards`, `data`, `optimizer`
- **Utilities**: `atb db verify`, `atb tests heartbeat`

### Database Schema & Migrations
- **Database**: PostgreSQL only, managed by SQLAlchemy
- **Migrations**: Alembic (`alembic upgrade head`)
- **Connection**: Via `DATABASE_URL` environment variable
- **Local Setup**: `docker-compose up -d postgres`

## Common Tasks & Workflows

### Development Workflow
```bash
# Setup once
make venv && make install && make deps
docker compose up -d postgres  # Note: 'docker compose', not 'docker-compose'
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# Regular development
make test                    # Run tests
make lint && make fmt        # Code quality
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30  # Test functionality
```

### Railway Deployment
```bash
./bin/railway-setup.sh       # Initial setup (requires Railway CLI)
# Set environment variables in Railway dashboard
./bin/railway-deploy.sh      # Deploy
```

### Performance & Monitoring
```bash
atb dashboards run monitoring --port 8000  # Real-time monitoring
atb data cache-manager info  # Cache status
python src/database_manager/app.py          # DB admin UI
```

## Configuration System

**Priority Order:**
1. Railway environment variables (production)
2. Environment variables (Docker/CI/local)
3. `.env` file (local development)

**Essential Variables:**
```env
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
BINANCE_API_KEY=your_key     # Required for live trading
BINANCE_API_SECRET=your_secret
TRADING_MODE=paper           # paper|live
LOG_LEVEL=INFO
```

## Important Notes for Agents

1. **Always install dependencies first** - CLI will fail without core packages
2. **PostgreSQL is mandatory** - no fallback database, verify connection early
3. **Use server requirements for faster builds** when possible
4. **Test in paper mode first** - live trading requires explicit confirmation
5. **Check Railway docs** for deployment (`docs/RAILWAY_QUICKSTART.md`)
6. **Use existing test infrastructure** - comprehensive markers and parallel testing
7. **Follow existing patterns** - well-established module structure and conventions

**Trust these instructions** - they reflect current codebase state. Only search if instructions are incomplete or incorrect.
