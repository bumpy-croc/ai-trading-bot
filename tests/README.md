# Test Suite

Comprehensive tests for reliability and correctness across components.

## Quick start
```bash
python tests/run_tests.py smoke
python tests/run_tests.py unit
python tests/run_tests.py integration
python tests/run_tests.py all --coverage
```

Run specific file or markers:
```bash
python tests/run_tests.py --file tests/test_strategies.py
python tests/run_tests.py -m "strategy and not slow"
```

## Markers
- unit, integration, live_trading, risk_management, strategy, slow, network

## Coverage
```bash
python tests/run_tests.py --coverage
# HTML at htmlcov/index.html
```

## Notes
- Uses PostgreSQL Testcontainers for database tests
- See `pytest.ini` and `tests/run_tests.py` for configuration and options 