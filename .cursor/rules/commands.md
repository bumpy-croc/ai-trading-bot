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
- 30d dev run: `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30`
- 1y run: `atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 365`

### Live Trading
- Paper: `atb live ml_basic --symbol BTCUSDT --paper-trading`
- Live (requires explicit ack): `atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks`
- Health: `PORT=8000 atb live-health ml_basic --symbol BTCUSDT --paper-trading`

### Monitoring & Utilities
- Dashboard: `atb dashboards run monitoring --port 8000`
- Cache check: `atb data cache-manager info`
- DB test: `atb db verify`
