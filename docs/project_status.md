# Project Status

> **Last Updated**: 2026-01-02
> **Maintainer Note**: This is a living document. Update at the start and end of each development session. Use the `/update-docs` command to keep this in sync.

---

## Current Focus

Reducing backtest/live engine duplication by centralizing shared entry and exit logic.

---

## Milestones

### Completed

- [x] **Core Trading System** - Backtesting engine, live trading, paper mode
- [x] **ML Prediction Pipeline** - CNN+LSTM models, ONNX export, model registry
- [x] **Data Infrastructure** - Binance/Coinbase providers, caching, sentiment integration
- [x] **Database Layer** - PostgreSQL, SQLAlchemy models, Alembic migrations
- [x] **CLI Interface** - `atb` command with comprehensive subcommands
- [x] **Testing Infrastructure** - Unit/integration tests, markers, parallel execution
- [x] **Monitoring** - Logging, dashboards, health endpoints
- [x] **Railway Deployment** - Production deployment configuration
- [x] **Code Quality Gates** - Black, Ruff, MyPy, Bandit integration
- [x] **ExecPlans System** - Structured approach for complex features

### In Progress

- [ ] **PSB System Implementation** - Automated docs, slash commands, regression prevention
- [ ] **Performance Optimization** - Ongoing ML pipeline improvements (#439)

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
| `ml_basic` | Production | Core ML-driven trend following |
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
| Codex Auto Review | In Progress | `docs/execplans/codex_auto_review.md` |

---

## Last Session Summary

**Date**: 2026-01-02

**Work Completed**:
- Added shared entry and validation helpers under `src/engines/shared/`
- Wired backtest/live entry handlers and partial-exit loops to shared helpers
- Added unit coverage for shared entry and validation utilities

**Ended At**:
- Shared entry/exit helper wiring and targeted unit tests

**Next Steps**:
- Run full unit test suite for shared helper changes

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
