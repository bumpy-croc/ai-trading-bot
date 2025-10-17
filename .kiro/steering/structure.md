# Project Structure & Organization

## Directory Layout

The project follows a **src-layout** pattern with clear separation of concerns:

```
├── src/                    # Main source code (importable package)
├── cli/                    # Command-line interface (atb tool)
├── tests/                  # Test suite with unit/integration tests
├── docs/                   # Comprehensive documentation
├── scripts/                # Utility scripts and tools
├── migrations/             # Alembic database migrations
├── data/                   # Historical market data (feather files)
├── logs/                   # Application and trading logs
├── cache/                  # Cached market data and ML predictions
└── artifacts/              # Generated reports and outputs
```

## Core Source Modules (`src/`)

### Trading Core
- **`strategies/`**: Trading strategies (ml_basic, sentiment, adaptive)
- **`backtesting/`**: Vectorized simulation engine
- **`live/`**: Live trading engine and execution
- **`trading/`**: Core trading interfaces and shared functionality

### Data & Analysis
- **`data/`**: Data management and caching utilities
- **`data_providers/`**: Market & sentiment data providers with caching
- **`indicators/`**: Technical indicators (pure functions)
- **`prediction/`**: ML model registry, ONNX runtime, caching

### Risk & Portfolio
- **`risk/`**: Risk parameters and position sizing utilities
- **`position_management/`**: Position sizing and portfolio management
- **`performance/`**: Performance metrics and analysis

### Infrastructure
- **`database/`**: SQLAlchemy models and DatabaseManager
- **`config/`**: Typed configuration loader, constants, feature flags
- **`utils/`**: Shared utilities (paths, symbols, logging)

### ML & Intelligence
- **`ml/`**: Trained models (.onnx/.keras) and metadata
- **`regime/`**: Market regime detection and analysis

### Monitoring & UI
- **`dashboards/`**: Web-based monitoring and analysis dashboards
- **`database_manager/`**: Flask-Admin UI for database inspection
- **`monitoring/`**: Real-time monitoring dashboard (Flask + Socket.IO)

### Optimization
- **`optimizer/`**: Parameter optimization and strategy tuning
- **`examples/`**: Minimal runnable examples

## Architecture Patterns

### Import Conventions
- **Absolute imports**: Always use `from src.module import ...`
- **Type hints**: Required for all public APIs and complex functions
- **Module organization**: One main class per file, related utilities grouped

### Configuration Pattern
```python
# Priority: Railway env vars > env vars > .env file
from src.config.config_loader import ConfigLoader
config = ConfigLoader.load()
```

### Database Pattern
```python
# Single DatabaseManager instance, PostgreSQL-only
from src.database.manager import DatabaseManager
db_manager = DatabaseManager()
```

### Strategy Pattern
```python
# All strategies use component composition
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generators.base import SignalGenerator
from src.strategies.components.risk_managers.base import RiskManager
from src.strategies.components.position_sizers.base import PositionSizer

def create_my_strategy() -> Strategy:
    signal_generator = MySignalGenerator()
    risk_manager = MyRiskManager()
    position_sizer = MyPositionSizer()
    
    return Strategy(
        name="my_strategy",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer
    )
```

### Data Provider Pattern
```python
# Cached wrapper around actual providers
from src.data_providers.cached_data_provider import CachedDataProvider
provider = CachedDataProvider(underlying_provider)
```

## Testing Organization

### Test Structure
```
tests/
├── unit/                   # Fast, isolated unit tests
├── integration/            # Database and system integration tests
├── strategies/             # Strategy-specific tests
├── data/                   # Test data fixtures
├── mocks/                  # Mock objects and utilities
└── conftest.py            # Shared fixtures and configuration
```

### Test Markers
- `@pytest.mark.unit`: Fast unit tests (< 1 second)
- `@pytest.mark.integration`: Database/system tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.database`: Tests requiring database

### Test Naming
- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`
- Use descriptive names: `test_ml_basic_strategy_generates_buy_signals_on_uptrend`


## CLI Organization (`cli/`)

### Command Structure
```
atb
├── backtest <strategy>     # Run backtesting
├── live <strategy>         # Live trading
├── live-health            # Live trading with health endpoint
├── dashboards             # Dashboard management
├── data                   # Data management utilities
├── db                     # Database operations
└── optimizer              # Parameter optimization
```

### Command Implementation
- Each command group in `cli/commands/`
- Shared utilities in `cli/core/`
- Main entry point: `cli/__main__.py`

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

## Documentation Structure

- **`docs/README.md`**: Documentation index
- **Guide naming**: `TOPIC_GUIDE.md` (e.g., `BACKTEST_GUIDE.md`)
- **Summary naming**: `TOPIC_SUMMARY.md` (e.g., `MONITORING_SUMMARY.md`)
- **Inline docs**: Comprehensive docstrings for all public APIs

## Security & Secrets

- **Never commit secrets**: Use `.env` file (gitignored)
- **Environment variables**: Preferred for production
- **API keys**: Store in environment, load via ConfigLoader