# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Modular cryptocurrency trading system supporting multiple strategies, backtesting, live trading (paper and live), ML-driven predictions (price and sentiment), PostgreSQL logging, and Railway deployment.

**Tech Stack**: Python 3.11+, PostgreSQL, TensorFlow/ONNX, Flask, SQLAlchemy, pandas, scikit-learn

**Coding rules in `CODE.md` must be followed at all times.**

## Autonomous Operation (daemon mode)

This repo is set up to be operated by a persistent Claude Code daemon (e.g. Claudeclaw) acting as the **PM**, delegating to specialist subagents (`.claude/agents/`) and reading/writing shared state (`.claude/state/`).

**If you are the daemon (main session), you are the PM.** Before making any material decision:

1. Read `.claude/state/charter.md` — the Board-owned mandate. If it has unfilled `TODO` markers for mission / autonomy envelope / escalation, stop and ask the human to fill them.
2. Read `.claude/state/risk-limits.json` — the hard lines.
3. Tail `.claude/state/log.md` (last ~50 lines) — recent institutional memory.
4. Check `.claude/state/incidents/*.md` (filter `status: open`) and `gh issue list --label type:incident --state open` — if any P0, scope the session to that incident.
5. Check `gh issue list --state open --label state:proposed,state:paper,state:building` — active WIP on the backlog.

**Primary slash commands:**
- `/standup` — full situational cycle: market read + ops snapshot + risk snapshot + synthesis. Run on schedule.
- `/triage` — sweep open proposals and incidents; dispatch reviewers; decide or escalate.
- `/heartbeat` — cheap (bash-only) dead-man's-switch. Run frequently (e.g., every 15–30 min).

**State layout (see `.claude/state/README.md` for full schema):**
- `charter.md`, `risk-limits.json` — human-owned config.
- `log.md` — append-only chronological record of every material action.
- `proposals/*.md`, `incidents/*.md` — flat directories; lifecycle via `status:` frontmatter.
- **Live backlog** — GitHub Issues + the Project board (labels: `state:*`, `type:*`, `area:*`, `owned-by:*`, `priority:*`, `needs:*`, `source:*`).

**Hard rules for the daemon:**
- Never change `.claude/state/charter.md` or `.claude/state/risk-limits.json` — those are human-owned.
- Never rewrite history in `log.md` or closed incidents — append-only; corrections are new entries referencing the earlier one.
- Never execute a `board_required: true` action without a human approving the proposal.
- Never promote a model's `latest` symlink for a live-trading symbol without human sign-off.
- If the charter is missing or invalid, refuse to make material decisions.

Full schema and lifecycle: `.claude/state/README.md`.

## Essential Commands

### Environment Setup

**Remote Environments (Claude Code Web)** — if `CLAUDE_CODE_REMOTE == true`:
- The venv is created by the `sessionStart` hook with `requirements-server.txt`.
- Use the existing `.venv` — it's already activated.

**Local Development**:
```bash
python -m venv .venv && source .venv/bin/activate
make install        # Install CLI (atb) + upgrade pip
make deps-dev       # Install dev dependencies
# If pip times out (TensorFlow ~500MB): .venv/bin/pip install --timeout 1000 <package>
# Or use lighter deps: make deps-server
```

### Database (PostgreSQL Required)
```bash
docker compose up -d postgres  # Note: 'docker compose', not 'docker-compose'
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
atb db verify
```

### Testing
```bash
atb test unit                # Unit tests (recommended)
atb test integration         # Integration tests
atb test smoke               # Quick smoke tests
atb test all                 # All tests

# Or pytest directly
pytest -m "not integration and not slow"  # Fast tests only
pytest tests/unit/test_backtesting.py     # Single file
```

**Test runner**: `python tests/run_tests.py unit|integration|smoke` (4 workers locally, 2 in CI).

**Coverage requirements**: Overall 85%, Live Trading Engine 95%, Risk Management 95%, Strategies 85%.

**Guidelines**: Use markers (`@pytest.mark.fast`, `@pytest.mark.integration`). Mock external APIs. Keep tests stateless with fixtures. Integration tests run sequentially. Keep test data under `tests/data/`.

### Code Quality
```bash
atb dev quality     # Run all: black + ruff + mypy + bandit
atb dev clean       # Remove .pytest_cache, .ruff_cache, etc.
```

### Backtesting & Live Trading
```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30
atb live ml_basic --symbol BTCUSDT --paper-trading
PORT=8000 atb live-health -- ml_basic --paper-trading  # With health endpoint
```

### ML Model Training
```bash
atb live-control train --symbol BTCUSDT --days 365 --epochs 50 --auto-deploy
atb live-control list-models
atb live-control deploy-model --model-path BTCUSDT/basic/2025-10-27_14h_v1
```

Models write to `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/` (model.onnx, metadata.json, feature_schema.json). The `latest` symlink auto-updates. See `docs/prediction.md` for pipeline details.

### Monitoring & Utilities
```bash
atb dashboards run monitoring --port 8000
atb data cache-manager info
atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 4
```

## Architecture

### Data Flow
```
Data Providers → Indicators → Strategy → Risk Manager → Execution
     ↓              ↓            ↓            ↓             ↓
  (Binance)     (RSI, EMA)  (ML Models)  (Position  (Live/Backtest)
  (Sentiment)   (MACD, ATR) (Signals)     Sizing)
```

### Directory Structure

**`src/`**:
- `config/` — Typed config loader, constants, feature flags
- `data_providers/` — Market & sentiment providers (Binance, Coinbase) with caching
- `database/` — SQLAlchemy models, DatabaseManager, Flask-Admin UI
- `engines/backtest/` — Vectorized simulation engine
- `engines/live/` — Live trading engine with real-time execution
- `engines/shared/` — Unified logic for both engines (models, cost calculator, risk handlers)
- `infrastructure/` — Logging, path resolution, geo detection, cache TTL, secrets
- `ml/` — Trained models: `models/{SYMBOL}/{TYPE}/{VERSION}/` with `latest` symlinks
- `prediction/` — Model registry, ONNX runtime, caching, feature pipeline
- `position_management/` — Position sizing policies
- `regime/` — Market regime detection
- `risk/` — Global risk management
- `sentiment/` — Sentiment adapters
- `strategies/` — Built-in strategies (ml_basic, ml_adaptive, ml_with_sentiment)
- `trading/` — Symbol conversion, reusable trading components

**`cli/`** — `atb` CLI entry point, commands organized by function.

**`tests/`** — `unit/` (fast, isolated) and `integration/` (DB, external providers).

### Strategy System

Strategies compose `SignalGenerator`, `RiskManager`, `PositionSizer`. Main interface: `process_candle(df, index, balance, positions) -> TradingDecision`. Components are independently testable.

### Database Schema

Core tables: `trading_sessions`, `trades`, `positions`, `account_history`, `performance_metrics`. Alembic manages migrations (`alembic upgrade head`).

### Configuration Priority

1. Railway environment variables (production/staging)
2. Environment variables (Docker/CI/local)
3. `.env` file (local development)

```env
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TRADING_MODE=paper  # paper|live
INITIAL_BALANCE=1000
LOG_LEVEL=INFO
LOG_JSON=true
```

## Workflow

### Making Changes

1. **Branch** from `develop`: `git checkout -b feature/your-feature`
2. **Code** — follow `CODE.md` at all times.
3. **Test** — `atb test unit && atb dev quality`
4. **Review** — run `architecture-reviewer` and `code-reviewer` agents after significant changes (features, refactoring, trading/financial logic). Run both in parallel for comprehensive reviews.
5. **Commit** — imperative present-tense, optional scope prefix (`fix:`, `feat:`, `docs:`). Only commit changes from your session. Verify correct branch first.

### Git & Pull Requests

- Write imperative commits: `Add short-entry guardrails` (not "Added" or "Adding").
- When merging/rebasing: fetch from remote first, use non-interactive mode.
- Follow `.github/pull_request_template.md`: describe user-facing impact, reference issues, include testing performed, request review after CI is green.
- Prefer GitHub MCP server if available, otherwise `gh` CLI.

### Handling PR Review Comments

When addressing review comments from a PR link:
1. Extract PR number from URL.
2. Use GitHub MCP `pull_request_read` with `method: "get_review_comments"` — do NOT fetch the entire PR.
3. For direct comment links: `gh api repos/OWNER/REPO/pulls/comments/COMMENT_ID`
4. Address the feedback, commit with a clear message, don't fetch the full diff unless necessary.

### Railway Environments

Project name: **innovative-transformation**. Use Railway MCP server or `railway` CLI.

| Environment | Branch | Notes |
|-------------|--------|-------|
| development | `develop` | |
| staging | `staging` | |
| main | `main` | **PRODUCTION — never destructive operations** |

See `docs/database.md` for Railway deployment and DB operations.

## Common Pitfalls

1. **Import Errors**: Run `make install` first — CLI requires core packages.
2. **Database Required**: PostgreSQL is mandatory, no fallback.
3. **Timeout on Deps**: Use `--timeout 1000` or `make deps-server` for large packages.
4. **Docker Compose**: Use `docker compose` not `docker-compose` (v2 syntax).
5. **Model Loading**: Strategies use `latest` symlink — ensure it points to a valid version.
6. **Use `--paper-trading`** when testing live trading changes.
7. **Check for existing branches** before creating new ones to avoid duplicates.
8. **Verify database connection** before running integration tests.

## What To Read For Your Task

`CODE.md` applies to **all tasks** — read it before writing any code.

| If you're working on... | Read these CODE.md sections | Reference docs |
|------------------------|----------------------------|----------------|
| Live trading / order execution | State Management, Position Fields, Error Handling, Thread Safety | `docs/live_trading.md` |
| Reconciliation / crash recovery | State Management, Position Fields, Database, Planning | `docs/live_trading.md`, `docs/database.md` |
| Exchange modes / margin / futures | Exchange Mode & Account Type Safety, Error Handling, State Management | `docs/live_trading.md` |
| Strategies / backtesting | Backtest-Live Parity, Arithmetic, Architecture | `docs/backtesting.md` |
| ML models / prediction | Input Validation, Resource Management, External API Calls | `docs/prediction.md` |
| Database operations | Database & Transactions, Error Handling | `docs/database.md` |
| API integrations / data providers | External API Calls, Timezone, Security | `docs/data_pipeline.md`, `docs/configuration.md` |
| UI / dashboard | Security (XSS, auth, redirect) | `docs/monitoring.md` |
| New modules / refactoring | Architecture, full CODE.md | `docs/architecture.md` |
| Railway / deployment | Security, Database | `docs/database.md` |

### Living Documents (keep updated)

| Document | Update when |
|----------|-------------|
| `docs/changelog.md` | After each feature/fix |
| `docs/project_status.md` | Start/end of sessions |
| `docs/architecture.md` | After architectural changes |

Use `/update-docs` to refresh. Full docs index: `docs/README.md`.
