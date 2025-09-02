# Railway Log Analysis System

Automated system for analyzing Railway logs, detecting error patterns, and generating fixes.

## Quick Start

```bash
# * Setup and validate system
python scripts/setup_railway_log_analysis.py

# * Fetch and analyze logs manually
atb logs fetch --environment staging --hours 1 --save
atb logs analyze --environment staging --hours 1 --output report.md

# * Run complete daily pipeline
atb logs daily --environment staging --hours 1 --dry-run
```

## Components

- **`log_analyzer.py`**: Core log parsing and error pattern detection
- **`railway_log_fetcher.py`**: Railway CLI integration for log retrieval
- **`auto_fixer.py`**: Automated fix generation and PR creation

## Integration

- **CLI Commands**: `atb logs` command group
- **GitHub Actions**: Daily automated execution
- **Database**: Historical analysis storage
- **Monitoring**: Dashboard integration

See `docs/RAILWAY_LOG_ANALYSIS_SYSTEM.md` for complete documentation.