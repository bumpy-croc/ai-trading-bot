# Documentation

> **Last Updated**: 2025-12-17  
> **Maintained By**: AI Trading Bot Team

This folder contains reference guides for the main subsystems that make up the AI trading platform. Each document focuses on how the runtime works today and includes links to relevant code and operational commands.

## Table of Contents

### Getting Started
- [Development workflow](development.md) - Environment setup, quality checks, and Railway deployment
- [Configuration](configuration.md) - Provider chain, feature flags, and local workflow

### Core Systems
- [Data pipeline](data_pipeline.md) - Market data providers, caching, and offline support
- [Backtesting](backtesting.md) - Historical simulation engine and optimization
- [Live trading](live_trading.md) - Real-time execution, risk controls, and safety features
- [Prediction & models](prediction.md) - ML model registry, inference, and training
- [Technical indicators](tech_indicators.md) - Shared indicator math and adapters

### Operations
- [Database](database.md) - PostgreSQL setup, migrations, and backups
- [Monitoring & observability](monitoring.md) - Logging, dashboards, and health endpoints

### Architecture
- [Component risk integration](architecture/component_risk_integration.md) - Coordination between position management, risk, and strategy layers

## Quick Links

- **First time setup**: Start with [Development workflow](development.md#environment-setup)
- **Running backtests**: See [Backtesting](backtesting.md#cli-usage)
- **Live trading**: Review [Live trading](live_trading.md#engine-highlights) safety controls
- **Troubleshooting**: Check [Database](database.md#cli-tooling) diagnostics

## Maintenance & validation

- Run `python -m cli docs validate` after documentation edits (or nightly) to scan for broken relative links, outdated `docker compose` references, and missing module READMEs.
- The validator reports counts for both errors and warnings so you can fail CI on regressions while still tracking informational notices locally.
