# CI Setup Requirements

## Integration Tests Database Setup

The integration tests require a PostgreSQL database to be properly configured. The CI environment must ensure:

### PostgreSQL Service
1. PostgreSQL service must be running before tests start
2. Database user `trading_bot` with superuser privileges
3. Test database `trading_bot_test` owned by `trading_bot` user
4. Proper DATABASE_URL environment variable set

### Environment Variables
```bash
DATABASE_URL=postgresql://trading_bot:test_password@localhost:5432/trading_bot_test
TEST_TYPE=integration  # Helps database manager detect integration test context
```

### Database Schema
The database schema should be created using SQLAlchemy models rather than Alembic migrations in CI:

```python
from src.database.manager import DatabaseManager
from src.database.models import Base
db = DatabaseManager()
Base.metadata.create_all(db.engine)
```

### Common Issues
- **"Database setup/reset failed before tests"**: PostgreSQL service not running
- **"relation does not exist"**: Database schema not properly created
- **SQLite fallback in CI**: DATABASE_URL not set or PostgreSQL not available

### Verification
Run integration tests locally to verify setup:
```bash
python tests/run_tests.py integration
```

All 130 integration tests should pass, including correlation control tests.