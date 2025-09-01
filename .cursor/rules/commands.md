---
description: Trading bot commands (concise)
alwaysApply: false
---

### Testing
- Quick smoke: `python tests/run_tests.py smoke`
- Unit tests (parallel, default 4 workers): `python tests/run_tests.py unit`
- Integration tests: `python tests/run_tests.py integration`
- Critical (live + risk): `python tests/run_tests.py critical`
- Specific file: `python tests/run_tests.py -f tests/test_ml_adaptive.py`
- Coverage: `python tests/run_tests.py --coverage`

### Backtesting
- 30d dev run (no DB): `python scripts/run_backtest.py ml_basic --days 30 --no-db`
- 1y run: `python scripts/run_backtest.py ml_with_sentiment --days 365`

### Live Trading
- Paper: `python scripts/run_live_trading.py ml_basic --paper-trading`
- Live (requires explicit ack): `python scripts/run_live_trading.py ml_basic --live-trading --i-understand-the-risks`
- Health: `python scripts/health_check.py`
- Stop: `python scripts/run_live_trading.py --stop`

### Monitoring & Utilities
- Dashboard: `python scripts/start_dashboard.py`
- Cache check: `python scripts/cache_manager.py --check`
- DB test: `python scripts/test_database.py`
- Backup DB: `python scripts/backup_database.py`
