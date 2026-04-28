# Changelog

All notable changes to the AI Trading Bot project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Maintainer Note**: This is a living document. Update after completing features, bug fixes, or significant changes. Use the `/update-docs` command to auto-populate entries.

---

## [Unreleased]

### Added
- **Monitoring dashboard mobile layout**: V2 dashboard reflows below 768px to a
  bottom tab bar + stacked content + inline inspector. Reuses the same React
  store and data flow; layout swap driven by `useIsMobile()` hook backed by
  `window.matchMedia` with a resize listener so it adapts live. iOS safe-area
  insets respected via `viewport-fit=cover` + `env(safe-area-inset-*)`.
- **Monitoring dashboard V2 redesign**: chart-led layout with left-rail nav
  (Dash / Pos / Strat / Trades / Risk / Logs), KPI strip, hero equity chart
  with overlay toggles (benchmark / trades / drawdown), positions strip, and
  a swappable right inspector. Light + dark themes (toggle persisted to
  `localStorage`). Tech stack swap: Bootstrap + Chart.js → React 18 (UMD) +
  Babel-standalone + socket.io-client. CDN scripts pinned with SRI hashes.
- New `GET /api/dashboard/state` endpoint bundles metrics + positions +
  trades + bot meta in a single request to keep first paint snappy. Accepts
  `?trades_limit=` (clamped to 1..500). Falls back to per-resource fetches
  in the JS adapter if the bundled endpoint is unavailable.
- New `MonitoringDashboard._get_bot_meta()` reads strategy / symbols /
  timeframe / mode / `max_open_positions` from the most recent **running**
  `trading_sessions` row (falls back to the most recent overall row),
  matching the "Exchange Mode & Account Type Safety" guidance so a stale
  paper-mode session can't mask an active live one.
- `.claude/launch.json` — preview-server configurations for all three
  dashboards plus live-health.
- Experimentation framework (`src/experiments/`) with declarative YAML suites,
  `atb experiment run|list|show|promote` CLI, ranked reporter with statistical
  verdicts, file-based ledger under `experiments/.history/`, and promotion
  writer for `StrategyVersionRecord`/`ChangeRecord` plus patch YAML emission.
- ML signal generators now expose `long_entry_threshold`, `short_entry_threshold`,
  `confidence_multiplier`, and regime-specific thresholds as overridable instance
  attributes (class constants remain as defaults).
- `ConfidenceWeightedSizer` gained `min_confidence_floor` parameter.
- `create_ml_basic_strategy`, `create_ml_adaptive_strategy`, and
  `create_ml_sentiment_strategy` accept the new tuning knobs.

### Removed
- Deleted the unused first-attempt optimizer layer: `src/optimizer/analyzer.py`,
  `validator.py`, `strategy_drift.py`, the `atb optimizer` CLI, the
  `OptimizationCycle` DB model/table, `DatabaseManager.record_optimization_cycle`,
  `fetch_optimization_cycles`, and the `/api/optimizer/cycles` dashboard route.
  Alembic migration `0011_drop_optimization_cycles` drops the table.
- Renamed `src/optimizer/` → `src/experiments/` now that the package reflects
  its actual purpose. `atb walk-forward` continues to work via
  `src/experiments/walk_forward.py`.

### Fixed
- Add ban-aware retry to Binance client startup — parses `-1003` ban expiry and sleeps until lifted instead of crashing (#590)
- `hyper_growth`: fix silent-SELL bug caused by feature-shape mismatch
  (#603). The factory wired `MLBasicSignalGenerator(model_type="sentiment")`
  but fed the sentiment model the 5-column price-only feature tensor
  instead of the 10 columns it was trained on. The model returned 0.0 on
  every bar, which the generator converted to `predicted_return=-1.0` and
  emitted as a constant SELL with confidence=1.0. Swapped to
  `model_type="basic"` (real directional edge of 55-57% BUY accuracy at
  12-24h horizons). Also tightened the default `stop_loss_pct` from 0.20
  to 0.10. On BTCUSDT 1h 2024: 14.16% → 99.80% return, 7.24% → 4.74% max
  drawdown, 0.055 → 0.259 Sharpe.

---

## 2026-02-18

### Infrastructure
- Added minimal CI dependencies and enabled tests in Claude GitHub workflow (#551)
- Added Claude Code GitHub Workflow (#543)

---

## 2026-01-15

### Added
- Automated cloud training with auto data download/upload (#532)
- CoinGecko data provider as Binance alternative (#538)
- Feature schema saving with trained ML models (#530)
- `--changed` flag to run quality checks only on modified files (#529)
- Code review agents and deployment slash commands
- Automated quality checks hook for Python files
- Side utilities and validation utility modules (#500)
- Order-type execution modeling for live and backtest (#493)

### Changed
- Consolidated backtesting and live engines into unified architecture (#527)
- Removed deprecated `src/indicators` directory (#515)
- Refactored strategies for improved code quality and maintainability (#501)
- Improved ML training and cloud module code quality (#502)
- Used shared `pnl_percent` function for engine parity (#505)

### Fixed
- Prevented race conditions in position tracking (#528)
- Addressed infrastructure code quality and safety issues (#513)
- Resolved database manager bugs and improved financial data safety (#512)
- Comprehensive position management code quality and safety improvements (#507)
- Critical issues in risk management module (#509)
- Comprehensive input validation for performance module (#508)
- Made regime regression test deterministic with dependency injection (#504)
- Used relative comparison in cache performance test (#540)

### Documentation
- Comprehensive risk management architecture documentation (#518)
- Updated docs and CLI commands for cache and migrations (#533)
- Added common PR review issues to CLAUDE.md (#499)
- Added instructions to run review agents after significant changes

---

## 2025-12-28

### Added
- Stop hook with completion detection for Claude Code Web
- PSB system analysis documentation (`docs/PSB_SYSTEM_ANALYSIS.md`)
- Automated documentation system (changelog.md, project_status.md, architecture.md)
- `/update-docs` slash command for documentation maintenance
- Shared entry utilities and validation helpers for consistent engine behavior
- Comprehensive engine parity test coverage (#487)
- Correlation sizing adjustments for runtime entries (#483)

### Changed
- Enhanced CLAUDE.md with Railway environment guidelines
- Unified backtest/live entry and partial-exit logic via shared helpers
- Refactored live entry execution to use LiveEntryHandler & LiveExecutionEngine (#482)
- Routed filled live exits through LiveExitHandler (#485)
- Completed shared engine models consolidation (#475)

### Fixed
- Fixed post-fee entry balance in live entry paths (#491)
- Aligned live engine dynamic risk handling (#490)
- Honored take-profit limit pricing (#489)
- Added missing order tracking columns to positions table migration
- Recorded live exits even when filled prices exceed deviation thresholds

### Documentation
- Updated documentation links in READMEs (#488)
- Added comprehensive backtesting engine audit report (#476)
- Added performance tracker integration execplan (#467)

---

## 2025-12-22

### Changed
- Removed outdated workflows for cursor reviews and nightly code quality

---

## 2025-12-21

### Added
- Nightly performance test workflow for CI (#438)

### Changed
- Optimized ML training pipeline with performance improvements (#439)
  - Batch processing enhancements
  - Memory efficiency improvements

### Documentation
- Clarified merge-develop command in documentation
- Updated AGENTS.md with detailed execplan storage guidelines
- Enhanced PR creation guidelines for clarity

---

## 2025-12-20

### Changed
- Refactored trading bot for better code quality (#437)
  - Code organization improvements
  - Enhanced maintainability

### Documentation
- Updated CLI command consistency and accuracy across docs
- Clarified live-health invocation across guides (#429)
- Fixed broken link in prediction README (#428)

---

## 2025-12-19

### Changed
- Refactored prediction model registry and usage (#421)
  - Improved model loading patterns
  - Enhanced registry structure

### Documentation
- Updated data pipeline and model registry docs (#416)
- Refreshed nightly documentation set (#427)
- Changed documentation scan workflow from nightly to weekly

---

## Earlier Changes

For changes prior to December 2025, see the git history:
```bash
git log --oneline --since="2025-01-01"
```

---

## Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes
- **Documentation**: Documentation-only changes
- **Infrastructure**: CI/CD, deployment, and tooling changes
