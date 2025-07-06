# Local PostgreSQL Development Setup

## Overview

The AI Trading Bot now supports **PostgreSQL for local development** to provide complete environment parity between development and production. This eliminates database-specific issues and provides a more realistic development experience.

## 🆚 Database Options Comparison

| Feature | PostgreSQL (Recommended) | SQLite (Simple) |
|---------|-------------------------|-----------------|
| **Environment Parity** | ✅ Identical to production | ❌ Different from production |
| **Setup Complexity** | ⚠️ Requires Docker | ✅ Zero setup |
| **Performance Testing** | ✅ Realistic performance | ❌ Different performance characteristics |
| **Complex Queries** | ✅ Full PostgreSQL features | ⚠️ Limited SQL features |
| **Concurrent Access** | ✅ True concurrent testing | ❌ File-based limitations |
| **Transaction Testing** | ✅ Full ACID compliance | ⚠️ Limited transaction support |
| **Team Development** | ✅ Consistent across team | ⚠️ Potential inconsistencies |
| **Resource Usage** | ❌ Higher memory/CPU | ✅ Minimal resources |
| **Startup Time** | ❌ Service startup required | ✅ Instant |
| **Data Reset** | ⚠️ Docker volume management | ✅ Simple file deletion |

## 🚀 Quick Setup

### Automated Setup (Recommended)
```bash
# Run the setup script and choose PostgreSQL
python scripts/setup_local_development.py
```

### Manual Setup
```bash
# 1. Copy environment configuration
cp .env.example .env

# 2. Edit .env file and uncomment the PostgreSQL line:
# DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# 3. Start PostgreSQL with Docker
docker-compose up -d postgres

# 4. Verify connection
python scripts/verify_database_connection.py
```

## 🐘 PostgreSQL Management

### Starting PostgreSQL
```bash
# Start PostgreSQL service
docker-compose up -d postgres

# Check if PostgreSQL is ready
docker-compose exec postgres pg_isready -U trading_bot -d ai_trading_bot
```

### Stopping PostgreSQL
```bash
# Stop PostgreSQL service
docker-compose down

# Stop and remove volumes (reset database)
docker-compose down -v
```

### Database Access
```bash
# Connect to PostgreSQL CLI
docker-compose exec postgres psql -U trading_bot -d ai_trading_bot

# View database logs
docker-compose logs postgres

# Follow logs in real-time
docker-compose logs -f postgres
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
python scripts/verify_database_connection.py

# Railway-style verification
python scripts/railway_database_setup.py --verify
```

### Performance Testing
```bash
# Run backtest with database logging
python scripts/run_backtest.py adaptive --days 30

# Check database performance
docker-compose exec postgres psql -U trading_bot -d ai_trading_bot -c "
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
from database.manager import DatabaseManager
db = DatabaseManager()
info = db.get_database_info()
print(f'Database Type: PostgreSQL' if info['is_postgresql'] else 'SQLite')
print(f'Connection Pool: {info[\"connection_pool_size\"]} connections')
"
```

## 💡 Development Workflows

### Start Development Session
```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Verify connection
python scripts/verify_database_connection.py

# 3. Run your development commands
python scripts/run_backtest.py adaptive --days 7
python scripts/run_live_trading.py adaptive
```

### Switch Between SQLite and PostgreSQL
```bash
# Switch to PostgreSQL
# Edit .env file:
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# Switch to SQLite
# Edit .env file (comment out DATABASE_URL):
# DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
```

### Data Migration Between Environments
```bash
# Export from SQLite
python scripts/export_sqlite_data.py

# Import to PostgreSQL (ensure PostgreSQL is running)
python scripts/import_to_postgresql.py

# Verify migration
python scripts/verify_database_connection.py
```

## 🔍 Troubleshooting

### Common Issues

#### PostgreSQL Won't Start
```bash
# Check if port 5432 is in use
lsof -i :5432

# Remove existing containers
docker-compose down -v
docker-compose up -d postgres
```

#### Connection Refused
```bash
# Check PostgreSQL status
docker-compose ps

# Check PostgreSQL logs
docker-compose logs postgres

# Verify environment configuration
cat .env | grep DATABASE_URL
```

#### Permission Issues
```bash
# Reset PostgreSQL data directory
docker-compose down -v
docker volume prune
docker-compose up -d postgres
```

#### Database Schema Issues
```bash
# Reset database schema
docker-compose exec postgres psql -U trading_bot -d ai_trading_bot -c "
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO trading_bot;
"

# Restart application to recreate tables
python scripts/verify_database_connection.py
```

### Performance Issues
```bash
# Monitor PostgreSQL performance
docker-compose exec postgres psql -U trading_bot -d ai_trading_bot -c "
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

## 🔄 Switching Back to SQLite

If you need to switch back to SQLite:

1. **Edit .env file**:
   ```bash
   # Comment out PostgreSQL
   # DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
   ```

2. **Stop PostgreSQL** (optional):
   ```bash
   docker-compose down
   ```

3. **Restart application**:
   ```bash
   python scripts/verify_database_connection.py
   ```

## 📊 Performance Comparison

### PostgreSQL Advantages for Development
- **Realistic Query Performance**: Same execution plans as production
- **Connection Pool Testing**: Test connection pool behavior locally
- **Advanced Features**: Test PostgreSQL-specific functions and features
- **Concurrent Development**: Multiple developers can connect simultaneously
- **Transaction Isolation**: Test complex transaction scenarios

### When to Use Each Option

**Use PostgreSQL when**:
- Working on complex database features
- Testing performance-critical operations
- Collaborating with a team
- Preparing for production deployment
- Testing database migrations

**Use SQLite when**:
- Quick prototyping or experimentation
- Simple single-developer tasks
- Limited development resources
- Testing basic functionality
- Working offline or with limited connectivity

---

## 🎯 Best Practices

1. **Use PostgreSQL for feature development** that involves database operations
2. **Use SQLite for quick tests** and experimentation
3. **Test database migrations** on PostgreSQL before production
4. **Monitor resource usage** with PostgreSQL locally
5. **Keep both environments updated** with the same schema changes
6. **Use version control** for database configuration files
7. **Document database changes** for team members

This setup provides the best of both worlds: the simplicity of SQLite when you need it, and the power and compatibility of PostgreSQL when you're building production features.