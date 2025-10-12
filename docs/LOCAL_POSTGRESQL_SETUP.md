# Local PostgreSQL Development Setup

## Overview

The AI Trading Bot uses **PostgreSQL for all environments** to provide complete environment parity between development and production. This eliminates database-specific issues and provides a consistent development experience.

## 🚀 PostgreSQL Benefits

**PostgreSQL provides:**
- **Environment Parity**: Identical to production
- **Performance Testing**: Realistic performance characteristics
- **Complex Queries**: Full PostgreSQL features
- **Concurrent Access**: True concurrent testing
- **Transaction Testing**: Full ACID compliance
- **Team Development**: Consistent across team

## 🚀 Quick Setup

### Quick Setup (Recommended)
```bash
# 1. Copy environment configuration
cp .env.example .env

# 2. Start PostgreSQL with Docker
docker compose up -d postgres

# 3. Set up environment
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# 4. Verify connection
atb db verify
```

### Manual Setup
```bash
# 1. Copy environment configuration
cp .env.example .env

# 2. Edit .env file and uncomment the PostgreSQL line:
# DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# 3. Start PostgreSQL with Docker
docker compose up -d postgres

# 4. Verify connection
atb db verify
```

## 🐘 PostgreSQL Management

### Starting PostgreSQL
```bash
# Start PostgreSQL service
docker compose up -d postgres

# Check if PostgreSQL is ready
docker compose exec postgres pg_isready -U trading_bot -d ai_trading_bot
```

### Stopping PostgreSQL
```bash
# Stop PostgreSQL service
docker compose down

# Stop and remove volumes (reset database)
docker compose down -v
```

### Database Access
```bash
# Connect to PostgreSQL CLI
docker compose exec postgres psql -U trading_bot -d ai_trading_bot

# View database logs
docker compose logs postgres

# Follow logs in real-time
docker compose logs -f postgres
```

### Database Management
```sql
-- Inside PostgreSQL CLI (psql)

-- List all tables
\dt

-- Describe a table
\d trading_sessions

-- View recent trades
SELECT * FROM trades ORDER BY exit_time DESC LIMIT 10;

-- Check database size
SELECT pg_size_pretty(pg_database_size('ai_trading_bot'));

-- Reset all data (careful!)
TRUNCATE TABLE trades, positions, account_history, system_events, strategy_executions CASCADE;
```

## 🔧 Configuration Details

### Environment Variables
```bash
# PostgreSQL connection (in .env file)
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379
```

### Database Configuration
- **Host**: localhost (or postgres container name)
- **Port**: 5432
- **Database**: ai_trading_bot
- **Username**: trading_bot
- **Password**: dev_password_123
- **Connection Pool**: 5 connections + 10 overflow

### Docker Compose Services
```yaml
# PostgreSQL database
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: ai_trading_bot
    POSTGRES_USER: trading_bot
    POSTGRES_PASSWORD: dev_password_123
  ports:
    - "5432:5432"

# Optional Redis for caching
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
```

## 🧪 Testing and Verification

### Database Connection Test
```bash
# Test database connection and operations
atb db verify

# Railway-style verification
atb db setup-railway --verify
```

### Performance Testing
```bash
# Run backtest with database logging
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

# Check database performance
docker compose exec postgres psql -U trading_bot -d ai_trading_bot -c "
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public';"
```

### Schema Verification
```bash
# Verify database schema
python3 -c "
import sys
import os
sys.path.insert(0, 'src')
from src.database.manager import DatabaseManager
db = DatabaseManager()
info = db.get_database_info()
print(f'Database Type: PostgreSQL')
print(f'Connection Pool: {info[\"connection_pool_size\"]} connections')
"
```

## 💡 Development Workflows

### Start Development Session
```bash
# 1. Start PostgreSQL
docker compose up -d postgres

# 2. Verify connection
atb db verify

# 3. Run your development commands
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 7
atb live ml_basic --symbol BTCUSDT --paper-trading
```

### PostgreSQL Configuration
```bash
# Local PostgreSQL
# Edit .env file:
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
```

### Data Management
```bash
# Verify database connection
atb db verify
```

## 🔍 Troubleshooting

### Common Issues

#### PostgreSQL Won't Start
```bash
# Check if port 5432 is in use
lsof -i :5432

# Remove existing containers
docker compose down -v
docker compose up -d postgres
```

#### Connection Refused
```bash
# Check PostgreSQL status
docker compose ps

# Check PostgreSQL logs
docker compose logs postgres

# Verify environment configuration
cat .env | grep DATABASE_URL
```

#### Permission Issues
```bash
# Reset PostgreSQL data directory
docker compose down -v
docker volume prune
docker compose up -d postgres
```

#### Database Schema Issues
```bash
# Reset database schema
docker compose exec postgres psql -U trading_bot -d ai_trading_bot -c "
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO trading_bot;
"

# Restart application to recreate tables
atb db verify
```

### Performance Issues
```bash
# Monitor PostgreSQL performance
docker compose exec postgres psql -U trading_bot -d ai_trading_bot -c "
SELECT
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;"
```


## 📊 Performance Comparison

### PostgreSQL Advantages for Development
- **Realistic Query Performance**: Same execution plans as production
- **Connection Pool Testing**: Test connection pool behavior locally
- **Advanced Features**: Test PostgreSQL-specific functions and features
- **Concurrent Development**: Multiple developers can connect simultaneously
- **Transaction Isolation**: Test complex transaction scenarios

### PostgreSQL Usage

**PostgreSQL is used for**:
- All development and production environments
- Complex database features
- Performance-critical operations
- Team collaboration
- Database migrations
- Testing and deployment

---

## 🎯 Best Practices

1. **Use PostgreSQL for all development** that involves database operations
2. **Test database migrations** on PostgreSQL before production
3. **Monitor resource usage** with PostgreSQL locally
4. **Keep all environments consistent** with the same schema changes
5. **Use version control** for database configuration files
6. **Document database changes** for team members

This setup provides consistent PostgreSQL experience across all environments.
