# Project Status

> **Last Updated**: 2026-06-15
> **Maintainer Note**: This is a living document. Update at the start and end of each development session. Use the `/update-docs` command to keep this in sync.

---

## In Flight

- **Backtest determinism / parity fingerprint (#486 parity, PR #811)** — the
  parity oracle the refactor series depends on. Found the ml_basic backtest
  varied across processes under load (49/50/51 trades on identical inputs).
  Root cause: multi-threaded BLAS/OpenMP float non-associativity (ONNX,
  `PYTHONHASHSEED`, prediction cache all ruled out; NOT the refactors). Fix:
  `Backtester.run` pins BLAS/OpenMP to 1 thread (`threadpoolctl`); ONNX stays
  multi-threaded. Verified byte-identical across 5 processes + a mechanism
  regression test. Perf neutral-to-faster. Lands before resuming refactors.

- **Live-capital bug-audit remediation (2026-06-10)** — a multi-agent code
  audit produced 17 tracked findings (#734–#750) across order execution,
  reconciliation, balance integrity, risk enforcement, and sync. Fix PRs in
  flight: fail-closed stop-loss lookup (#733 → fixes #713), partial-ops
  hard-disable (#734 interim), order-tracker tracking-lost handling, and
  periodic-reconciler SL-fill PnL booking. Highest open items after the
  PR train: close-quantity derivation (#737), balance-ledger serialization
  (#735/#736), dead rate-limit retries (#738), stale-snapshot stop re-arm
  (#739), live max-drawdown enforcement (#749), backtest partial-PnL
  inflation (#748).

- **Monitoring dashboard V2 redesign** — chart-led layout (left rail + KPI
  strip + hero equity chart + position inspector) replacing the legacy
  Bootstrap+Chart.js dashboard. Stack swap: React 18 (UMD) +
  Babel-standalone + socket.io-client, CDN scripts pinned with SRI hashes.
  Adds `GET /api/dashboard/state` bundled endpoint and `_get_bot_meta()`
  helper that surfaces real session metadata (strategy / symbols /
  timeframe / mode / `max_open_positions` / `risk_per_trade`). On branch
  `claude/brave-jackson-d9e7ee` — under deep-review-auto-fix iteration 3,
  awaiting clean exit before merge to develop.

---

## Current Focus

Hardening `hyper_growth` on top of the declarative experimentation framework
(`src/experiments/`). A sweep surfaced a silent-SELL bug — the factory fed
the sentiment ONNX model a price-only feature tensor and the model returned
its fallback sentinel on every bar. Fix landed in #603
(`model_type="sentiment"` → `"basic"`, `stop_loss_pct` default 0.20 → 0.10).
Seven experimentation-framework gaps surfaced by the same sweep are being
addressed in a follow-up (#604): factory_kwargs plumbing, FlatRiskManager
override contract, base_fraction routing through wrapping sizers, clearer
regime-attr errors, signal-quality diagnostic, bitwise-identical-variant
warning, per-trade sequence tie-break.

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
- [x] **Live Trading Engine Modularization (#486)** - Decomposed the `LiveTradingEngine` god-class (~6,560 lines at the start of the effort) into a thin orchestrator over a coordinator family. The handover-doc plan (Steps A–E) landed via #823–#826: `__init__` 534→~110 lines (15 phase helpers), `start()` 327→~18 lines (7 phase helpers), `_trading_loop` 390→~250 lines (short-entry + periodic-account extracted), and entry/exit coordinator `Protocol`s tightened to concrete types. Every step proven byte-identical against the deterministic backtest parity fingerprint. Engine now ~2,620 lines; optional further extraction (e.g. `LiveEngineBuilder`, `LiveStartupSequencer`) remains as future polish. Plan: `docs/refactor/live_engine_modularization.md`.

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

**Date**: 2026-06-15

**Work Completed**:
- Completed the #486 live-engine modularization plan (Steps A–E) across four
  PRs (#823–#826): decomposed `__init__`, `start()`, and `_trading_loop` into
  cohesive helpers; tightened the entry/exit coordinator `Protocol`s to concrete
  types and moved their tests to `create_autospec`; documented the deliberate
  non-extractions. Every PR proven byte-identical against the backtest parity
  fingerprint and merged CI-green.

**Ended At**:
- #486 modularization plan complete; engine is a thin ~2,620-line orchestrator.

**Next Steps**:
- Optional further extraction if pursued: `_process_legacy_short_entry` →
  `LiveEntryCoordinator`; a `LiveEngineBuilder` for construction; a
  `LiveStartupSequencer` for the bootstrap sequence.

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
