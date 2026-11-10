# Documentation

> **Last Updated**: 2026-10-04
> **Maintained By**: AI Trading Bot Team

This folder contains reference guides for the main subsystems that make up the AI trading platform. Each document focuses on how the runtime works today and includes links to relevant code and operational commands.

## Table of Contents

### Living Documents (Auto-Updated)
- [Changelog](changelog.md) - Timeline of all project changes
- [Project Status](project_status.md) - Current milestones, focus, and session tracking
- [Architecture](architecture.md) - System design overview and components

### Getting Started
- [Development workflow](development.md) - Environment setup, quality checks, and Railway deployment
- [Configuration](configuration.md) - Provider chain, feature flags, and local workflow
- [Performance tracker validation](testing/performance_tracker_validation.md) - Validation notes for the shared performance tracker

### Core Systems
- [Data pipeline](data_pipeline.md) - Market data providers, caching, and offline support
- [Backtesting](backtesting.md) - Historical simulation engine and optimization
- [Live trading](live_trading.md) - Real-time execution, risk controls, and safety features
- [Prediction & models](prediction.md) - ML model registry, inference, training, and artifact retention
- [Technical indicators](tech_indicators.md) - Shared indicator math and adapters
- [GPU configuration](ml/gpu_configuration.md) - Training on GPU
- [Coingecko provider](coingecko_alternative.md) - Binance alternative data provider

### Operations
- [Database](database.md) - PostgreSQL setup, migrations, and backups
- [Monitoring & observability](monitoring.md) - Logging, dashboards, and health endpoints
- [Operations runbook](operations_runbook.md) - Production troubleshooting and recovery procedures
- [GitHub labels](github_labels.md) - Issue label taxonomy
- [Technical debt](technical_debt.md) - Measured code-health state and standing backlog themes

### Architecture
- [Component risk integration](architecture/component_risk_integration.md) - Coordination between position management, risk, and strategy layers
- [Risk management architecture](risk_management_architecture.md) - Three-layer risk design
- [Memory system](architecture/memory_system.md) - Operational memory design
- [Model evaluation system](architecture/model_evaluation_system.md) - Frozen-exam model verification
- [Architecture proposals](architecture/proposals/) - Design proposals
- [Refactor plans and reviews](refactor/) - Live-engine modularization and backtest-live parity

### Research, Audits and Analysis
- [Research](research/) - Experiments, market briefs, model promotions, ops and risk snapshots
- [Backtest engine audit](backtest_engine_audit_report.md) - 2025-12-27 audit
- [Observability audit](observability_audit_2026-06-08.md) - 2026-06-08 alerting audit
- [Risk refactoring analysis](risk_refactoring_analysis.md) - Possible follow-on risk refactors
- [ML architecture research](ml_architecture_research.md), [ML model research report](ml_model_research_report.md), [ML model implementation guide](ml_model_implementation_guide.md)
- [Strategy optimization journal](strategy_optimization_journal.md) - Adaptive trend strategy tuning log
- [Testing improvements summary](TESTING_IMPROVEMENTS_SUMMARY.md) - 2025-11 testing review
- [PSB system analysis](PSB_SYSTEM_ANALYSIS.md) - Project-setup framework review

### Plans and Archive
- [AWS Claude Code instance plan](plans/aws-claude-code-instance.md) and [superpowers plans](superpowers/plans/) - Design plans
- [Archive](archive/) - Completed phase docs, ExecPlans and issue write-ups kept for history. Archived files carry a header naming the archive date and the doc that supersedes them; do not treat them as current.

## ExecPlan lifecycle

An ExecPlan is a living design-and-progress document (format: `.agents/PLANS.md`).

- **Active**: the plan lives in `.agents/plans/` (see `AGENTS.md`) and its header says `Status: Active`.
- **Closed**: the work has shipped or been abandoned. Set `Status: Closed` with the date and outcome, add a one-line archive header naming the doc that supersedes it, and move the file to `docs/archive/execplans/`.
- Closed plans are history, not reference. The shipped behaviour is described in the subsystem docs above and in the changelog.

## Quick Links

- **First time setup**: Start with [Development workflow](development.md#environment-setup)
- **Running backtests**: See [Backtesting](backtesting.md#cli-usage)
- **Live trading**: Review [Live trading](live_trading.md#engine-highlights) safety controls
- **Troubleshooting**: Check [Database](database.md#cli-tooling) diagnostics
