# Contributing to AI Trading Bot

Thank you for your interest in contributing to the AI Trading Bot! This document provides guidelines and best practices for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Commit Guidelines](#commit-guidelines)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors
- Remember that this project handles real financial data - prioritize security and safety

## Getting Started

### Prerequisites

1. **Python 3.9+** installed
2. **PostgreSQL** for database (via Docker or local install)
3. **Git** for version control
4. **Virtual environment** for dependency isolation

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/bumpy-croc/ai-trading-bot.git
cd ai-trading-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install  # Install CLI tool
make deps     # Install all dependencies

# Setup database
docker compose up -d postgres
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# Verify setup
python scripts/verify_database_connection.py
pytest -q
```

## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `docs/*` - Documentation updates

### Creating a Feature Branch

```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name
```

## Code Quality Standards

We maintain high code quality standards. Please review [docs/CODE_QUALITY.md](docs/CODE_QUALITY.md) for detailed guidelines.

### Core Principles

1. **Readability** - Code should be self-documenting
2. **Simplicity** - Favor simple solutions over complex ones
3. **Testability** - Write code that can be easily tested
4. **Safety** - This handles real money - prioritize safety
5. **Performance** - Optimize for production use

### Code Style

#### Python
- Follow PEP 8 style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use meaningful variable names

```python
# Good
def calculate_position_size(
    balance: float,
    risk_per_trade: float,
    stop_loss_distance: float
) -> float:
    """Calculate position size based on risk parameters."""
    return (balance * risk_per_trade) / stop_loss_distance

# Bad
def calc(b, r, s):
    return (b * r) / s
```

### Linting and Formatting

Run code quality checks before committing:

```bash
# Run all checks
make code-quality

# Or individually
ruff check . --fix        # Linting
ruff format .             # Formatting
python bin/run_mypy.py    # Type checking
bandit -c pyproject.toml -r src  # Security checking
```

### Pre-commit Hooks

Install pre-commit hooks to automate quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing Requirements

All code changes must include appropriate tests. See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for comprehensive testing documentation.

### Test Coverage

Minimum coverage requirements:
- **Live Trading Engine:** 95%
- **Risk Management:** 95%
- **Strategies:** 85%
- **Data Providers:** 80%
- **Overall:** 85%

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m live_trading     # Live trading tests
pytest -m risk_management  # Risk management tests
pytest -m unit             # Unit tests only
pytest -m integration      # Integration tests only

# Run tests in parallel
pytest -n 4
```

### Writing Tests

Follow the Arrange-Act-Assert (AAA) pattern:

```python
def test_position_size_calculation():
    # Arrange
    balance = 10000
    risk_per_trade = 0.02
    stop_loss_distance = 100
    
    # Act
    position_size = calculate_position_size(
        balance, risk_per_trade, stop_loss_distance
    )
    
    # Assert
    assert position_size == 2.0
```

## Documentation

### Documentation Requirements

- **All new features** must be documented
- **API changes** require updated documentation
- **Configuration changes** need .env.example updates
- **Breaking changes** must be clearly noted

### Documentation Standards

1. **Module READMEs** - Each `src/` module should have a README.md
2. **Docstrings** - All public functions and classes need docstrings
3. **Type Hints** - Use type hints for all function signatures
4. **Examples** - Include working code examples
5. **Updates** - Keep documentation in sync with code

### Writing Documentation

```python
def process_market_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Process historical market data for a trading pair.
    
    Fetches OHLCV data from the configured provider and applies
    necessary preprocessing for strategy analysis.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        start_date: Start date for data range
        end_date: End date for data range
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
        
    Raises:
        ValueError: If symbol format is invalid
        DataProviderError: If data fetch fails
        
    Example:
        >>> df = process_market_data('BTCUSDT', '1h', start, end)
        >>> print(df.head())
    """
    ...
```

## Pull Request Process

### Before Submitting

1. ✅ All tests pass locally
2. ✅ Code quality checks pass
3. ✅ Documentation is updated
4. ✅ Commit messages follow guidelines
5. ✅ Branch is up-to-date with develop

### PR Checklist

- [ ] Tests added/updated for changes
- [ ] Documentation updated
- [ ] Code quality checks pass
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are descriptive
- [ ] PR description explains the change

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks** run via CI/CD
2. **Code review** by maintainers
3. **Testing** in staging environment
4. **Approval** required before merge
5. **Merge** to develop branch

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

### Examples

```bash
# Feature
git commit -m "feat(strategies): add momentum-based entry signals"

# Bug fix
git commit -m "fix(risk): correct position sizing calculation for volatile markets"

# Documentation
git commit -m "docs(backtest): update guide with offline cache instructions"

# Refactor
git commit -m "refactor(database): improve connection pooling efficiency"
```

## Specific Contribution Areas

### Adding a New Strategy

1. Create strategy file in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement required methods
4. Add comprehensive tests
5. Update `src/strategies/README.md`
6. Add example usage in documentation

### Adding a New Data Provider

1. Create provider in `src/data_providers/`
2. Implement `DataProvider` interface
3. Add caching support
4. Include error handling
5. Write integration tests
6. Document API requirements

### Improving ML Models

1. Train model with sufficient data (6+ months)
2. Validate on out-of-sample data
3. Test in paper trading first
4. Document model architecture
5. Include metadata JSON file
6. Update `docs/MODEL_TRAINING_AND_INTEGRATION_GUIDE.md`

## Security Considerations

### Never Commit

- ❌ API keys or secrets
- ❌ Private keys
- ❌ Database passwords
- ❌ Personal data
- ❌ Production configuration

### Always

- ✅ Use environment variables
- ✅ Update `.env.example` for new variables
- ✅ Validate user inputs
- ✅ Sanitize database queries
- ✅ Log security-relevant events

## Performance Guidelines

### Database Queries

- Use connection pooling
- Avoid N+1 queries
- Use appropriate indexes
- Batch operations when possible

### Data Processing

- Use vectorized operations (pandas/numpy)
- Minimize API calls with caching
- Process data in chunks for large datasets
- Profile code for bottlenecks

## Getting Help

### Resources

- **Documentation:** [docs/README.md](docs/README.md)
- **Issues:** [GitHub Issues](https://github.com/bumpy-croc/ai-trading-bot/issues)
- **Discussions:** [GitHub Discussions](https://github.com/bumpy-croc/ai-trading-bot/discussions)

### Questions?

- Check existing documentation first
- Search closed issues for similar problems
- Ask in GitHub Discussions
- Open an issue if you find a bug

## Recognition

Contributors are recognized in:
- Repository contributors page
- Release notes
- Special mentions for significant contributions

Thank you for contributing to making AI Trading Bot better! 🚀

---

**Last Updated:** 2025-10-07
