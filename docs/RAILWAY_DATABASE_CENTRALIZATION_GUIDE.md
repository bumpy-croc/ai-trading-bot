# Railway PostgreSQL Database Setup Guide

## Overview

This guide outlines the setup and configuration of a centralized PostgreSQL database for the AI Trading Bot system on Railway. The system uses PostgreSQL exclusively for both development and production environments.

## Architecture

### Centralized PostgreSQL Setup
- **Trading Bot Service**: Connects to PostgreSQL database
- **Dashboard Service**: Connects to PostgreSQL database
- **Local Development**: PostgreSQL database (Docker or native)
- **Railway Production**: PostgreSQL database service

## Database Choice: PostgreSQL

### Why PostgreSQL?

1. **SQLAlchemy Compatibility**: Perfect integration with existing ORM models
2. **ACID Transactions**: Critical for financial trading data integrity
3. **Complex Queries**: Supports joins, subqueries, and complex analytics
4. **Data Persistence**: Reliable storage for trading history and performance metrics
5. **Relational Structure**: Handles foreign keys and table relationships
6. **Backup and Recovery**: Built-in backup features on Railway
7. **Scalability**: Better for growing datasets and complex reporting
8. **Connection Pooling**: Efficient resource management for multiple services

## Implementation Steps

### Step 1: Railway Project Setup

1. **Create PostgreSQL Database Service**
   ```bash
   # In Railway dashboard:
   # 1. Go to your project
   # 2. Click "+ New" button
   # 3. Select "Database" > "PostgreSQL"
   # 4. Deploy the service
   ```

2. **Database Connection Variables**
   Railway automatically provides these environment variables:
   - `DATABASE_URL`: Complete PostgreSQL connection string
   - `PGHOST`: Database host
   - `PGPORT`: Database port (default: 5432)
   - `PGUSER`: Database username
   - `PGPASSWORD`: Database password
   - `PGDATABASE`: Database name

### Step 2: Local Development Setup

#### Option 1: Docker PostgreSQL (Recommended)
```bash
# Use docker-compose for local PostgreSQL
docker-compose up -d postgres

# Set environment variable
export DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db
```

#### Option 2: Native PostgreSQL Installation
```bash
# Install PostgreSQL locally
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib

# macOS:
brew install postgresql

# Create database and user
createdb trading_db
createuser trading_user
```

### Step 3: Environment Configuration

#### Local Development
```bash
# .env file
DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db
```

#### Railway Production
```bash
# Railway automatically provides:
DATABASE_URL=postgresql://user:password@host:port/database
```

### Step 4: Database Schema Creation

The system automatically creates database tables on first connection:
- PostgreSQL database will auto-create tables on first run
- SQLAlchemy handles table creation via `Base.metadata.create_all()`

## Connection Details

### Private Network Connection (Railway)
Services within the same Railway project can connect via private network:
```
postgresql://postgres:password@postgres.railway.internal:5432/railway
```

### Public Connection (External Access)
For external connections (development, debugging):
```
postgresql://postgres:password@host.proxy.railway.app:port/railway
```

### Local Development Connection
```
postgresql://trading_user:trading_pass@localhost:5432/trading_db
```

## Configuration Management

The DatabaseManager requires a PostgreSQL connection and will:
1. Check `DATABASE_URL` environment variable
2. Connect to PostgreSQL with connection pooling
3. Create database tables automatically
4. Provide error messages if PostgreSQL is not available

Connection pool configuration:
- Pool size: 5 connections
- Max overflow: 10 connections
- Pool pre-ping: Enabled
- SSL mode: Prefer

## Benefits

### Performance
- **Concurrent Access**: Multiple services can safely access the same database
- **Connection Pooling**: Efficient database connection management (5+10 connections)
- **Query Optimization**: PostgreSQL's advanced query planner
- **Indexing**: Automatic and custom indexes for performance

### Reliability
- **Data Consistency**: ACID transactions ensure data integrity
- **Backup/Recovery**: Railway's built-in backup features
- **High Availability**: PostgreSQL's proven reliability
- **Connection Recovery**: Automatic connection recovery with pre-ping

### Scalability
- **Horizontal Scaling**: Services can scale independently
- **Data Growth**: PostgreSQL handles large datasets efficiently
- **Query Performance**: Optimized for complex analytical queries
- **Connection Management**: Pool scaling based on demand

### Cost Efficiency
- **Single Database**: No duplication of database resources
- **Shared Costs**: Database costs shared across services
- **Efficient Resource Usage**: Connection pooling reduces overhead

## Database Testing

### Running Database Tests
```bash
# Run database tests only
python tests/run_tests.py database

# Run with coverage
python tests/run_tests.py database --coverage

# Run specific database test file
python tests/run_tests.py --file test_database.py
```

### Test Coverage
The database tests cover:
- Connection management and pooling
- All DatabaseManager methods
- Error handling and recovery
- Transaction management
- Session management

## Monitoring and Maintenance

### Database Health
- Railway provides built-in database monitoring
- Connection metrics and query performance
- Built-in database UI for management

### Connection Pool Monitoring
```python
# Get connection statistics
db_manager = DatabaseManager()
stats = db_manager.get_connection_stats()
print(stats)
```

### Backup Strategy
- Railway's native backup feature
- Automated daily backups
- Point-in-time recovery capabilities

### Security
- Encrypted connections (SSL/TLS)
- Network isolation within Railway project
- Environment variable security
- Connection pooling security

## Troubleshooting

### Connection Issues
1. Check DATABASE_URL environment variable
2. Verify PostgreSQL service is running
3. Check private network connectivity
4. Verify connection pool status

### Performance Issues
1. Monitor connection pool usage
2. Analyze slow queries
3. Check database resource usage
4. Review connection pool configuration

### Data Consistency
1. Ensure proper transaction management
2. Use database locks for concurrent operations
3. Monitor for deadlocks

### Common Errors
- `DATABASE_URL environment variable is required`: Set PostgreSQL connection string
- `Only PostgreSQL databases are supported`: Check URL format
- Connection timeout: Check network connectivity

## Cost Considerations

### PostgreSQL Costs
- **Compute**: Based on CPU/RAM usage
- **Storage**: Based on data size
- **Network**: Minimal within private network
- **Connection Overhead**: Minimized by connection pooling

### Cost Optimization
- **App Sleeping**: Enable for non-critical environments
- **Resource Sizing**: Right-size database resources
- **Connection Pooling**: Reduce connection overhead
- **Query Optimization**: Use indexes and efficient queries

## Security Best Practices

1. **Environment Variables**: Never commit database credentials
2. **Network Access**: Use private network for internal communication
3. **SSL/TLS**: Always use encrypted connections (sslmode=prefer)
4. **Access Control**: Limit database access permissions
5. **Monitoring**: Track database access and queries
6. **Connection Security**: Secure connection pool management

## Migration from Previous Systems

<!-- Migration from SQLite section removed: project is now PostgreSQL-only -->

---

This PostgreSQL-centered approach provides a robust, scalable, and maintainable database solution for the AI Trading Bot system.
