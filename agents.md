# AI Trading Bot - Agent Guide

A modular cryptocurrency trading system focused on long-term, risk-balanced trend following with ML-driven predictions and comprehensive safety measures.

## Project Overview

## Operating rules

- Always follow the coding standards in `docs/CODE_QUALITY.md`
- Run non-interactive merge and rebase commands so that my input is not required
- Never commit secrets. Use environment variables; see `src/utils/secrets.py`.

## Quick Setup

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
make install  # Install CLI tool 'atb'
make deps     # Install dependencies (may timeout - use deps-server for lighter build)

# Database (PostgreSQL required)
docker compose up -d postgres
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
python scripts/verify_database_connection.py

# Verify installation
atb --help
```

## Core Architecture

```
src/
├── data_providers/    # Market & sentiment data (Binance, SentiCrypt, cached)
├── strategies/        # Trading logic (ml_basic, ml_adaptive, ml_with_sentiment)
├── risk/             # Position sizing, stop-loss, exposure limits
├── live/             # Live trading engine with hot-swapping
├── backtesting/      # Vectorized historical simulation
├── prediction/       # ONNX model registry and caching
├── database/         # PostgreSQL models and management
├── monitoring/       # Real-time Flask dashboards
└── ml/              # Trained models (.onnx/.keras) + metadata
```

## Essential Commands

### Development & Testing
```bash
# Quick backtest (development)
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

# Run tests
pytest -q                    # Quick test run
python tests/run_tests.py smoke    # Smoke tests
python tests/run_tests.py critical # Live trading + risk tests

# Code quality
ruff check . --fix
black .
```

### Live Trading (Always start with paper!)
```bash
# Paper trading (safe)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Live trading (requires explicit confirmation)
atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks

# Health monitoring
atb live-health --port 8000 -- ml_basic --paper-trading
```

### Monitoring & Utilities
```bash
# Dashboard
atb dashboards run monitoring --port 8000

# Cache management
atb data cache-manager info
atb data prefill-cache --symbols BTCUSDT --timeframes 1h --years 2

# Database
atb db verify
atb db migrate
```

## Development Workflow

### 1. Code Standards
- **Security**: Never commit secrets, use environment variables
- **Style**: Google Python Style Guide, type hints for public APIs
- **Imports**: Absolute imports under `src/`
- **Comments**: Better Comments style (`*` important, `!` warning, `TODO:` future)

### 2. Strategy Development
```python
class YourStrategy(BaseStrategy):
    def calculate_indicators(self, df):
        # Calculate strategy-specific indicators
        pass
    
    def check_entry_conditions(self, df, index):
        # Entry signal logic - return True/False
        pass
    
    def check_exit_conditions(self, df, index, entry_price):
        # Exit signal logic - return True/False
        pass
```

### 3. Testing Approach
- **Unit tests**: Fast, mocked dependencies (`pytest -m "not integration"`)
- **Integration tests**: Database/network interactions (sequential)
- **Critical tests**: Live trading engine + risk management
- **Keep tests stateless**: Use fixtures, avoid global state

### 4. Risk Management
- Max 1-2% risk per trade
- Position sizing based on ATR and account balance
- Stop-loss enforcement at strategy and system level
- Drawdown monitoring with automatic throttling

## Security & Safety

### Security Rules
- Never embed API keys/secrets in code
- Use parameterized SQL queries
- Validate all external inputs
- Environment variables for credentials (`src/utils/secrets.py`)

### Trading Safety
- **Always paper trade first** before live money
- Validate strategies with backtesting on multiple market conditions
- Use `--i-understand-the-risks` flag for live trading
- Emergency stop: `atb live-control emergency-stop`

## File Patterns

### Configuration
- Constants in `src/config/constants.py`
- Config loading via `src/config/config_manager.py`
- Feature flags in `feature_flags.json`

### Database
- Models in `src/database/models.py`
- Migrations via Alembic (`alembic upgrade head`)
- Connection pooling through `DatabaseManager`

### ML Models
- ONNX format preferred for inference
- Metadata stored alongside models
- Central registry in `prediction.models.registry`

## Key Conventions

### Error Handling
```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return fallback_value  # Always have fallback
```

### Logging
- Use Python logging module, not print
- Levels: DEBUG (dev), INFO (events), WARNING (issues), ERROR (failures)
- Include context in messages

### Testing
```bash
# Test categories via markers
pytest -m "fast"              # Quick tests only
pytest -m "not integration"   # Unit tests only
pytest -m "critical"          # Live trading + risk tests
```

## Common Pitfalls

1. **Network timeouts**: Use `pip install --timeout 1000` for large packages
2. **Database dependency**: PostgreSQL required - no SQLite fallback
3. **Paper trading first**: Never skip paper trading validation
4. **Risk validation**: Always backtest risk management changes
5. **Model format**: Prefer ONNX for production inference

## Resources

- **Architecture**: `.cursor/rules/architecture.md`
- **Commands**: `.cursor/rules/commands.md`
- **Testing**: `docs/TESTING_GUIDE.md`
- **Risk Management**: `docs/RISK_AND_POSITION_MANAGEMENT.md`
- **Railway Deployment**: `docs/RAILWAY_QUICKSTART.md`

---

**Remember**: This handles real money. Validate changes thoroughly. When in doubt, backtest more. 🛡️