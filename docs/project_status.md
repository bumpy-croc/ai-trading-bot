# Project Status

> **Last Updated**: 2026-02-18
> **Maintainer Note**: This is a living document. Update at the start and end of each development session. Use the `/update-docs` command to keep this in sync.

---

## Current Focus

Implementing PSB (Plan, Setup, Build) framework improvements: automated documentation, slash commands, and regression prevention patterns.

---

## Milestones

### Completed

- [x] **Core Trading System** - Backtesting engine, live trading, paper mode
- [x] **ML Prediction Pipeline** - CNN+LSTM models, ONNX export, model registry
- [x] **Data Infrastructure** - Binance/Coinbase/CoinGecko providers, caching, sentiment integration
- [x] **Database Layer** - PostgreSQL, SQLAlchemy models, Alembic migrations
- [x] **CLI Interface** - `atb` command with comprehensive subcommands
- [x] **Testing Infrastructure** - Unit/integration tests, markers, parallel execution
- [x] **Monitoring** - Logging, dashboards, health endpoints
- [x] **Railway Deployment** - Production deployment configuration
- [x] **Code Quality Gates** - Black, Ruff, MyPy, Bandit integration
- [x] **ExecPlans System** - Structured approach for complex features
- [x] **Engine Consolidation** - Unified backtest/live engines with shared modules (#527)
- [x] **Risk Management Architecture** - Three-layer risk system with comprehensive docs (#518)
- [x] **Race Condition Fixes** - Thread-safe position tracking (#528)
- [x] **Feature Schema Saving** - ML models save feature schemas for validation (#530)
- [x] **Cloud Training Automation** - Auto data download/upload for cloud training (#532)
- [x] **CI/CD Pipeline** - Claude Code GitHub Workflow with tests (#551)

### In Progress

- [ ] **PSB System Implementation** - Automated docs, slash commands, regression prevention
- [ ] **Performance Optimization** - Ongoing ML pipeline improvements

### Planned

- [ ] **Multi-Asset Portfolio Support** - Trade multiple symbols simultaneously
- [ ] **Advanced Risk Management** - Enhanced position sizing, correlation-aware risk
- [ ] **Sentiment Analysis V2** - Improved sentiment integration and weighting
- [ ] **Short-Selling Improvements** - Better short entry/exit logic

### Backlog (Future Consideration)

- [ ] Alternative data sources (on-chain metrics, social sentiment)
- [ ] Reinforcement learning strategies
- [ ] Mobile monitoring app
- [ ] Multi-exchange support (Kraken, Coinbase Pro)

---

## Active Strategies

| Strategy | Status | Description |
|----------|--------|-------------|
| `ml_basic` | Production | Core ML-driven trading strategy |
| `ml_adaptive` | Production | Regime-adaptive ML strategy |
| `ml_sentiment` | Production | ML with sentiment integration |
| `ensemble_weighted` | Production | Weighted ensemble of signals |
| `momentum_leverage` | Experimental | Momentum-based with leverage |

---

## Recent ExecPlans

| Plan | Status | Location |
|------|--------|----------|
| Training Pipeline Optimization | Completed | `docs/execplans/training_pipeline_optimization.md` |
| Indicator Refactor | Completed | `docs/execplans/indicator_refactor_plan.md` |
| Platform Modularization | Completed | `docs/execplans/platform_modularization_plan.md` |
| Remove Safe Trainer | Completed | `docs/execplans/remove_safe_trainer.md` |
| Backtesting Engine Audit | Completed | `docs/execplans/backtesting_engine_audit.md` |
| Shared Engine Consolidation | Completed | `docs/execplans/shared_engine_consolidation.md` |
| Performance Tracker Integration | Completed | `docs/execplans/performance_tracker_integration.md` |

---

## Last Session Summary

**Date**: 2026-02-18

**Work Completed**:
- Implemented PSB high-priority improvements: updated changelog, project status, architecture docs
- Added `#` hashtag regression prevention pattern to CLAUDE.md
- Verified `/update-docs` slash command is complete

**Ended At**:
- PSB high-priority documentation improvements

**Next Steps**:
- Implement PSB medium-priority items (feature reference docs, additional slash commands)
- Continue performance optimization work

---

## Key Metrics (Latest Backtest)

> Update this section after running backtests with strategy performance metrics.

```
Strategy: ml_basic
Symbol: BTCUSDT
Timeframe: 1h
Period: Last 30 days

[Run `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30` to update]
```

---

## Quick Commands

```bash
# Check project health
atb dev quality

# Run tests
atb test unit

# Quick backtest
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

# Start paper trading
atb live ml_basic --symbol BTCUSDT --paper-trading

# Update documentation
/update-docs
```

---

## Notes & Reminders

- Always run `atb dev quality` before committing
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`
- Create feature branches from `develop`
- Update this file at session end with "Last Session Summary"
