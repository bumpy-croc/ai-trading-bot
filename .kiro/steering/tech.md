# Technology Stack

## Core Technologies

- **Python 3.9+**: Primary language with type hints and modern features
- **PostgreSQL**: Primary database with connection pooling and migrations
- **SQLAlchemy 2.0+**: ORM with async support and modern query patterns
- **Alembic**: Database migrations and schema management
- **ONNX Runtime**: ML model inference with cross-platform compatibility
- **Flask**: Web framework for dashboards and admin interfaces
- **Socket.IO**: Real-time communication for monitoring dashboards

## Key Libraries

- **Data & Analysis**: pandas, numpy, scikit-learn, pyarrow
- **Trading APIs**: python-binance, ccxt (multi-exchange support)
- **ML/AI**: tensorflow, onnx, onnxruntime, tf2onnx
- **Web**: Flask, Flask-SocketIO, Flask-Admin, gevent
- **Database**: psycopg2-binary, sqlalchemy, alembic
- **Utilities**: python-dotenv, tqdm, psutil, requests

## Build System & CLI

The project uses a **Makefile-based build system** with a custom CLI tool (`atb`):


### Installation & Setup
```bash
# Development setup
make dev-setup              # Full development environment
make install                # Install CLI (atb) + upgrade pip
make deps                   # Install dev dependencies
make deps-server            # Production dependencies only
```

### Common Commands
```bash
# Testing
make test                   # Run pytest with 4 workers
pytest -q                   # Quick test run
pytest -n 4                # Parallel testing

# Code Quality
make code-quality           # Run all quality checks
ruff check . --fix          # Linting with auto-fix
ruff format .               # Code formatting
python bin/run_mypy.py      # Type checking
bandit -c pyproject.toml -r src  # Security analysis

# Database
make migrate                # Run database migrations
atb db verify               # Verify database connection
alembic upgrade head        # Manual migration

# Trading Operations
atb backtest ml_basic --symbol BTCUSDT --days 30
atb live ml_basic --paper-trading
atb dashboards run monitoring --port 8000
```

## Configuration Management

**Priority Order** (highest to lowest):
1. Railway environment variables (production)
2. Environment variables (Docker/CI/local)
3. `.env` file (local development)

**Key Environment Variables**:
- `DATABASE_URL`: PostgreSQL connection string
- `BINANCE_API_KEY` / `BINANCE_API_SECRET`: Trading API credentials
- `TRADING_MODE`: `paper` or `live`
- `LOG_LEVEL`: Logging verbosity
- `LOG_JSON`: Enable JSON logging (auto-enabled in production)

## Development Tools

- **Code Quality**: ruff (linting), black (formatting), mypy (type checking), bandit (security)

- **Testing**: pytest with plugins (asyncio, mock, xdist, timeout, cov, randomly)

- **Database**: testcontainers for integration tests
- **Monitoring**: Built-in Flask dashboards with real-time updates

## Deployment

- **Railway**: Primary deployment platform with PostgreSQL
- **Docker**: Containerized deployment with docker-compose
- **Local**: Direct Python execution with local PostgreSQL

## Performance Considerations

- **Vectorized Operations**: pandas/numpy for backtesting performance
- **Connection Pooling**: SQLAlchemy connection management
- **Caching**: Multi-layer caching for market data and ML predictions