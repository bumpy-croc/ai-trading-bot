# Scripts

Utility scripts for database management, data analysis, and system maintenance.

## Database & Setup

### verify_database_connection.py
Verify PostgreSQL database connectivity and configuration.

```bash
python scripts/verify_database_connection.py
```

### railway_database_setup.py
Initialize and configure PostgreSQL database on Railway.

```bash
python scripts/railway_database_setup.py
```

### reset_railway_database.py
Reset Railway database (WARNING: destructive operation).

```bash
python scripts/reset_railway_database.py
```

### backup_database.py
Create backups of PostgreSQL database.

```bash
python scripts/backup_database.py
```

### fix_collation_mismatch.py
Fix PostgreSQL collation mismatches in database.

```bash
python scripts/fix_collation_mismatch.py
```

## Data Management

### download_binance_data.py
Download historical market data from Binance.

```bash
python scripts/download_binance_data.py
```

### populate_dummy_data.py
Populate database with dummy trading data for testing dashboards.

```bash
python scripts/populate_dummy_data.py --trades 100 --confirm
```

### analyze_btc_data.py
Analyze Bitcoin historical data and generate statistics.

```bash
python scripts/analyze_btc_data.py
```

## Backtesting & Optimization

### run_backtest.py
Run backtesting programmatically (prefer CLI: `atb backtest`).

```bash
python scripts/run_backtest.py
```

### run_optimizer.py
Run parameter optimization for strategies.

```bash
python scripts/run_optimizer.py
```

### compare_btc_eth_ml_basic.py
Compare ML Basic strategy performance on BTC vs ETH.

```bash
python scripts/compare_btc_eth_ml_basic.py
```

## Live Trading

### run_live_trading_with_health.py
Start live trading with health monitoring endpoint.

```bash
python scripts/run_live_trading_with_health.py
```

## Machine Learning

### deep_learning_sentiment_script.py
Train deep learning models with sentiment data integration.

```bash
python scripts/deep_learning_sentiment_script.py
```

## Visualization & Analysis

### regime_visualization.py
Visualize market regime detection results.

```bash
python scripts/regime_visualization.py
```

## Testing & Validation

### validate_docs.py
Validate documentation for broken links, outdated commands, and consistency.

```bash
python scripts/validate_docs.py
```

Checks for:
- Broken internal links
- Missing referenced files
- Outdated command syntax (e.g., docker-compose)
- Missing module READMEs
- SQLite references (project is PostgreSQL-only)

### parse_junit_failures.py
Parse JUnit XML test results and extract failure information.

```bash
python scripts/parse_junit_failures.py <junit_xml_file>
```

## Database Setup Files

### postgres-init.sql
PostgreSQL initialization SQL script for Docker/Railway setup.

## Usage Notes

- Most scripts require `DATABASE_URL` environment variable to be set
- Database scripts should be run with caution in production
- Use `atb` CLI commands when available (preferred over scripts)
- Scripts in `/scripts` are meant for one-off tasks, automation, or CI/CD

## Adding New Scripts

When adding new scripts:
1. Add appropriate shebang line: `#!/usr/bin/env python3`
2. Make executable: `chmod +x scripts/your_script.py`
3. Add docstring explaining purpose
4. Document in this README
5. Consider if it should be a CLI command instead
