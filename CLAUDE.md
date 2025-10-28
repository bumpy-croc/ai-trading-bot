# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular cryptocurrency trading system focused on long-term, risk-balanced trend following. It supports backtesting, live trading (paper and live), ML-driven predictions (price and sentiment), PostgreSQL logging, and Railway deployment.

**Tech Stack**: Python 3.11+, PostgreSQL, TensorFlow/ONNX, Flask, SQLAlchemy, pandas, scikit-learn

## Essential Commands

### Environment Setup
```bash
# Create virtual environment and install
python -m venv .venv && source .venv/bin/activate
make install                 # Install CLI (atb) + upgrade pip
make deps-dev                # Install dev dependencies

# If pip install times out (common with TensorFlow ~500MB):
.venv/bin/pip install --timeout 1000 <package>
# Or use lighter production dependencies:
make deps-prod
```

### Database Setup (PostgreSQL Required)
```bash
# Start local PostgreSQL (note: 'docker compose', not 'docker-compose')
docker compose up -d postgres
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# Verify connection
atb db verify
```

### Testing
```bash
# Run test suite (recommended)
atb test unit                # Unit tests only
atb test integration         # Integration tests
atb test smoke               # Quick smoke tests
atb test all                 # All tests

# Or use test runner directly
python tests/run_tests.py unit          # Unit tests with parallelism
python tests/run_tests.py integration   # Integration tests

# Or use pytest directly
pytest -q                               # All tests
pytest -m "not integration and not slow"  # Fast tests only
pytest tests/unit/test_backtesting.py    # Single file
```

### Code Quality
```bash
atb dev quality              # Run all: black + ruff + mypy + bandit

# Individual tools
black .                      # Format code
ruff check . --fix          # Lint with auto-fix
python bin/run_mypy.py      # Type checking
bandit -c pyproject.toml -r src  # Security scan

# Clean caches
atb dev clean                # Remove .pytest_cache, .ruff_cache, etc.
```

### Backtesting & Live Trading
```bash
# Quick backtest (development)
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

# Live trading (paper mode - safe)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Live trading with health endpoint
atb live-health --port 8000 -- ml_basic --paper-trading
```

### ML Model Training & Deployment
```bash
# Train new model (writes to src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/)
atb live-control train --symbol BTCUSDT --days 365 --epochs 50 --auto-deploy

# List available models
atb live-control list-models

# Deploy specific model version
atb live-control deploy-model --model-path BTCUSDT/basic/2025-10-27_14h_v1

# Training with custom parameters
atb train model BTCUSDT --start-date 2023-01-01 --end-date 2024-12-01 \
  --epochs 100 --batch-size 64 --sequence-length 120 \
  --skip-plots --skip-robustness  # For faster experiments
```

### Monitoring & Utilities
```bash
# Start monitoring dashboard
atb dashboards run monitoring --port 8000

# Cache management
atb data cache-manager info
atb data cache-manager list --detailed
atb data cache-manager clear-old --hours 24

# Prefill cache for faster backtests
atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 4
```

## Architecture & Key Concepts

### High-Level Data Flow
```
Data Providers → Indicators → Strategy → Risk Manager → Execution
     ↓              ↓            ↓            ↓             ↓
  (Binance)     (RSI, EMA)  (ML Models)  (Position  (Live/Backtest)
  (Sentiment)   (MACD, ATR) (Signals)     Sizing)
```

### Directory Structure

**Core Application** (`src/`):
- `backtesting/` - Vectorized simulation engine for strategy testing
- `config/` - Typed configuration loader, constants, feature flags
- `data_providers/` - Market & sentiment providers (Binance, Coinbase) with caching
- `database/` - SQLAlchemy models, DatabaseManager, Flask-Admin UI
- `infrastructure/` - Cross-cutting concerns:
  - `logging/` - Centralized logging config, context, structured events
  - `runtime/` - Path resolution, geo detection, cache TTL, secrets
- `live/` - Live trading engine with real-time execution
- `ml/` - Trained models and metadata
  - **Legacy**: Root-level `*.onnx` files (currently used by strategies)
  - **Registry**: `models/{SYMBOL}/{TYPE}/{VERSION}/` (new versioned structure)
- `prediction/` - Model registry, ONNX runtime, caching, feature pipeline
- `position_management/` - Position sizing policies
- `regime/` - Market regime detection and analysis
- `risk/` - Global risk management applied across entire system
- `sentiment/` - Sentiment adapters that merge provider data onto market series
- `strategies/` - Built-in strategies (ml_basic, ml_adaptive, ml_with_sentiment)
- `trading/` - Trading-specific helpers:
  - `symbols/` - Symbol conversion utilities (BTCUSDT ↔ BTC-USD)
  - `shared/` - Reusable trading components

**CLI** (`cli/`):
- Entry point: `atb` command installed via `pip install -e .`
- Commands organized by function: `commands/backtest.py`, `commands/live.py`, etc.

**Tests** (`tests/`):
- `unit/` - Fast, isolated unit tests
- `integration/` - Database and external provider integration tests
- Use markers: `@pytest.mark.integration`, `@pytest.mark.fast`, `@pytest.mark.slow`

### ML Training Pipeline Architecture

The training pipeline (`src/ml/training_pipeline/`) is modular and versioned:

**Pipeline Modules**:
- `config.py` - Configuration dataclasses (TrainingConfig, DiagnosticsOptions)
- `ingestion.py` - Download price data and load sentiment data
- `features.py` - Create robust features, merge price/sentiment data
- `datasets.py` - Build TensorFlow datasets with sequences
- `models.py` - Create adaptive CNN+LSTM models
- `artifacts.py` - Save models, metadata, ONNX exports, plots
- `pipeline.py` - Orchestrate complete training workflow

**Key Flow**:
1. **Ingestion**: Download historical OHLCV + sentiment data
2. **Feature Engineering**: Create technical indicators + sentiment features
3. **Dataset Creation**: Build sequences with sliding windows
4. **Model Training**: CNN + LSTM architecture with mixed precision
5. **Artifact Saving**: Write to `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/`
   - `model.keras` - TensorFlow SavedModel format
   - `model.onnx` - ONNX export for fast inference
   - `metadata.json` - Training params, dataset info, performance metrics
   - `feature_schema.json` - Required features for inference
6. **Symlink Update**: Point `latest` to new version for automatic deployment

**Model Registry Structure**:
```
src/ml/models/
├── BTCUSDT/
│   ├── basic/
│   │   ├── 2025-10-27_14h_v1/
│   │   │   ├── model.keras
│   │   │   ├── model.onnx
│   │   │   ├── metadata.json
│   │   │   └── feature_schema.json
│   │   └── latest -> 2025-10-27_14h_v1
│   └── sentiment/
│       └── ...
└── ETHUSDT/
    └── ...
```

### Strategy System

All strategies follow a component-based design pattern:
- `Strategy` class composes `SignalGenerator`, `RiskManager`, `PositionSizer`
- Main interface: `process_candle(df, index, balance, positions) -> TradingDecision`
- `TradingDecision` contains signal, position size, regime context, risk metrics
- Components are independently testable and reusable

### Database Schema

**Core Tables**:
- `trading_sessions` - Track sessions with strategy configuration
- `trades` - Complete trade history with entry/exit prices and P&L
- `positions` - Active positions with real-time unrealized P&L
- `account_history` - Balance snapshots for performance tracking
- `performance_metrics` - Aggregated metrics (win rate, Sharpe, drawdown)

**Migration**: Alembic manages schema changes (`alembic upgrade head`)

### Configuration System

**Priority Order**:
1. Railway environment variables (production/staging)
2. Environment variables (Docker/CI/local)
3. `.env` file (local development)

**Essential Variables**:
```env
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TRADING_MODE=paper  # paper|live
INITIAL_BALANCE=1000
LOG_LEVEL=INFO
LOG_JSON=true  # Enable structured logging
```

## Operational Guidelines

### Core Principles

1. **Use the `atb` CLI** for all interactions with the application (not standalone scripts)
2. **Delete temporary scripts** after use - do not leave them in the repository
3. **Ensure test coverage** - all changes must have corresponding unit/integration tests
4. **Tests must pass** - run relevant tests before committing changes

### Making Changes

1. **Create branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature
   ```

2. **Write code** following conventions (see AGENTS.md for full details):
   - **Functions**: Keep concise, self-evident purpose, always include docstrings
   - **Types**: Use type hints, avoid `Any` where possible
   - **Variables**: Use descriptive names, extract magic numbers to constants
   - **Error Handling**: Use specific exception types, log with context, no empty catch blocks
   - **Code Structure**: Early returns, avoid deep nesting, favor composition over inheritance
   - **Comments**: Explain the "why" (goal), not the "what" (mechanics)
   - **Comments must be present tense**: Describe what code *does*, not history of changes
     - ❌ Bad: `# New enhanced v2 API.`
     - ✅ Good: `# Fetches user data from the v2 API.`

3. **Add tests** - Coverage requirements:
   - Overall: 85% minimum
   - Live Trading Engine: 95%
   - Risk Management: 95%
   - Strategies: 85%
   - Keep tests FIRST: Fast, Isolated, Repeatable, Self-validating, Timely
   - Use AAA pattern (Arrange, Act, Assert)

4. **Run quality checks**:
   ```bash
   atb dev quality
   atb test unit
   ```

5. **Commit frequently** with clear messages:
   ```bash
   # Use imperative, present-tense
   git commit -m "Add short-entry guardrails to ml_basic strategy"
   git commit -m "fix: resolve race condition in position manager"

   # Only commit changes made during the session, not whole working tree
   # Verify correct branch before committing
   ```

### Using ExecPlans for Complex Work

For significant features or refactors, create an ExecPlan following `.agents/PLANS.md`:

**ExecPlan Structure**:
- **Purpose**: User-visible behavior enabled by this change
- **Progress**: Checkbox list with timestamps of granular steps
- **Surprises & Discoveries**: Unexpected behaviors encountered
- **Decision Log**: Key decisions with rationale and dates
- **Outcomes & Retrospective**: Summary of achievements and lessons

**Location**: Store in `docs/execplans/descriptive_name.md`

**Key Requirements**:
- Self-contained - includes all context needed
- Living document - update as work progresses
- Observable outcomes - describe commands to run and expected outputs
- Commit often with clear messages as plan is executed

### Testing Guidelines

**Test Organization**:
- Use `python tests/run_tests.py` to run tests - do not add custom runners
- Default parallelism: 4 workers locally, 2 in CI
- Categories: `smoke`, `unit`, `integration`, `critical`, `fast`, `slow`, `grouped`, `benchmark`
- Prefer markers over fragile selection: `-m "not integration and not slow"`
- Keep tests stateless with fixtures - avoid global state
- Mock external APIs - never hit real exchanges in unit tests
- Integration tests run sequentially
- Keep test data under `tests/data/` - avoid large new binaries

**Running Tests**:
```bash
# Use official test runner
python tests/run_tests.py unit          # Unit tests with parallelism
python tests/run_tests.py integration   # Integration tests (sequential)
python tests/run_tests.py smoke         # Quick smoke tests

# Or use pytest directly with markers
pytest -m "not integration and not slow"  # Fast tests only
pytest tests/unit/test_backtesting.py     # Specific file
```

**Test Structure (AAA Pattern)**:
```python
@pytest.mark.fast
def test_position_sizing():
    # Arrange
    balance = 10000
    risk_per_trade = 0.02

    # Act
    position_size = calculate_position_size(balance, risk_per_trade)

    # Assert
    assert position_size == 200
```

### Git & Pull Request Guidelines

**Git Commits**:
- Write imperative, present-tense commits: `Add short-entry guardrails` (not "Added" or "Adding")
- Use optional scope prefixes when helpful: `docs:`, `fix:`, `feat:`
- Only commit changes made during your session, not the whole working tree
- Verify correct branch before committing
- When merging/rebasing: fetch from remote first, use non-interactive mode

**Pull Requests**:
Follow `.github/pull_request_template.md`:
- Describe user-facing impact
- Reference issues with `(#123)` when relevant
- Include testing performed (`atb test unit`, `atb dev quality`)
- Attach strategy metrics or dashboard screenshots when behavior changes
- Document migrations or scripts in `docs/` if they change operational steps
- Request review only after CI is green

**GitHub Interaction**:
- Prefer GitHub MCP server if available
- Otherwise use `gh` CLI (authenticated with GITHUB_TOKEN env variable)

### Handling PR Review Comments

**Efficient Comment Retrieval**:
When addressing review comments from a PR link:
1. Extract the PR number from the URL (e.g., `https://github.com/user/repo/pull/123` → PR #123)
2. Use the GitHub MCP server `pull_request_read` with `method: "get_review_comments"` to fetch only review comments
3. Do NOT retrieve the entire PR contents - this wastes tokens
4. If given a direct comment link with `#comment-ID`, use the `gh` CLI to extract just that comment:
   ```bash
   gh api repos/OWNER/REPO/pulls/comments/COMMENT_ID
   ```

**Example Workflow**:
- User provides: `https://github.com/user/repo/pull/123#discussion_r123456789`
- Extract: PR #123, comment ID `r123456789`
- Fetch: Use MCP `pull_request_read` method `get_review_comments` with appropriate pagination
- Address the specific feedback identified in the comment
- Commit changes with clear message referencing the issue
- Do not fetch the full PR diff or all comments unless necessary

## Important Context

### Model Training & Deployment Flow

1. **Training**: `atb train model` or `atb live-control train`
   - Writes to `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/`
   - Updates `latest` symlink automatically
   - Includes metadata.json with training params and performance

2. **Deployment**: Strategies load via `latest` symlink
   - Atomic updates via symlink repointing
   - No downtime for model changes
   - Can rollback with `atb live-control deploy-model`

3. **Validation**: Before production deployment
   - Check metadata.json for performance metrics
   - Run backtest with new model
   - Verify ONNX export matches Keras model

### Path Validation Pattern

When working with user-provided paths (especially for models):
```python
def _resolve_version_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (MODEL_REGISTRY / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")
    if MODEL_REGISTRY not in candidate.parents:
        raise ValueError("Model path must be inside the registry")
    return candidate
```

This pattern:
- Uses `.resolve()` to normalize paths and resolve symlinks
- Validates resolved path is within expected directory
- Prevents path traversal and symlink escape attacks

### Railway Deployment

Quick deployment steps:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Initialize and configure
railway init
railway add postgresql
railway variables set BINANCE_API_KEY=your_key
railway variables set TRADING_MODE=paper

# Deploy
railway up

# Verify
railway run atb db verify
railway logs --environment production
```

### Common Pitfalls

1. **Import Errors**: Always run `make install` first - CLI requires core packages
2. **Database Required**: PostgreSQL is mandatory, no fallback
3. **Timeout on Deps**: Use `--timeout 1000` or `make deps-prod` for large packages
4. **Docker Compose**: Use `docker compose` not `docker-compose` (v2 syntax)
5. **Model Loading**: Strategies use `latest` symlink - ensure it points to valid version
6. **Test Isolation**: Use fixtures, avoid global state, mock external APIs

### Security & Configuration

**Security Best Practices**:
- Never commit API keys, passwords, or personal data - use environment variables
- Always use placeholders in examples: `YOUR_API_KEY`, `DATABASE_CONNECTION_STRING`
- Validate and sanitize all external inputs rigorously
- Validate all user-provided paths (use `.resolve()` + parent checks)
- Use subprocess array form, not `shell=True`
- Limit pickle deserialization to trusted local files only
- Add `nosec` annotations only when justified with comments
- Implement retries, exponential backoff, and timeouts on external calls

**Configuration**:
- Never commit API keys or live trading credentials
- Load via environment variables consumed by `src/config` loaders
- Keep `.env` files out of version control
- Document required settings in PRs
- For local databases: run `atb db migrate` after schema updates
- Ensure backups in `backups/` are encrypted or excluded from commits

## Documentation References

- **Full Docs**: `docs/README.md` - Complete documentation index
- **Backtesting**: `docs/backtesting.md` - Engine internals and CLI usage
- **Live Trading**: `docs/live_trading.md` - Safety controls and deployment
- **Data Pipeline**: `docs/data_pipeline.md` - Offline cache and download utilities
- **Monitoring**: `docs/monitoring.md` - Logging config and dashboards
- **Prediction**: `docs/prediction.md` - Model registry and inference workflow
- **Configuration**: `docs/configuration.md` - Provider chain and feature flags
- **Database**: `docs/database.md` - Migrations, backups, Railway operations
- **Development**: `docs/development.md` - Environment setup and quality gates
