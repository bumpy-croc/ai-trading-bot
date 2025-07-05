# Railway Database Centralization Guide

## Overview

This guide outlines the process of centralizing the database for the AI Trading Bot system on Railway, moving from individual SQLite databases per service to a shared PostgreSQL database.

## Architecture Changes

### Before (Current State)
- **Trading Bot Service**: Individual SQLite database (`/src/data/trading_bot.db`)
- **Dashboard Service**: Individual SQLite database (`/src/data/trading_bot.db`)
- **Local Development**: SQLite database in `src/data/trading_bot.db`

### After (Centralized)
- **Trading Bot Service**: Connects to shared PostgreSQL database
- **Dashboard Service**: Connects to shared PostgreSQL database  
- **Local Development**: SQLite database (unchanged for local dev)
- **Railway Production**: Single PostgreSQL database shared between services

## Database Choice: PostgreSQL

### Why PostgreSQL over Redis?

1. **SQLAlchemy Compatibility**: Perfect integration with existing ORM models
2. **ACID Transactions**: Critical for financial trading data integrity
3. **Complex Queries**: Supports joins, subqueries, and complex analytics
4. **Data Persistence**: Reliable storage for trading history and performance metrics
5. **Relational Structure**: Handles foreign keys and table relationships
6. **Backup and Recovery**: Built-in backup features on Railway
7. **Scalability**: Better for growing datasets and complex reporting

### Redis Limitations for This Use Case
- NoSQL nature would require complete rewrite of data models
- No built-in relationships, requiring manual relationship management
- Limited query capabilities (no SQL, no complex joins)
- Memory-only storage increases data loss risk
- More expensive for persistent storage of large datasets

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

### Step 2: Code Configuration

The system already supports PostgreSQL through the `DATABASE_URL` environment variable in the `DatabaseManager` class.

### Step 3: Migration Strategy

1. **Database Schema Creation**
   - PostgreSQL database will auto-create tables on first run
   - SQLAlchemy will handle table creation via `Base.metadata.create_all()`

2. **Data Migration** (if needed)
   - Export data from existing SQLite databases
   - Import into PostgreSQL database
   - Use provided migration scripts

### Step 4: Environment Configuration

#### Local Development (No Change)
```bash
# .env file - continues to use SQLite
# DATABASE_URL is not set, falls back to SQLite
```

#### Railway Production
```bash
# Railway automatically provides:
DATABASE_URL=postgresql://user:password@host:port/database
```

### Step 5: Service Configuration

Both services will automatically use the shared PostgreSQL database when `DATABASE_URL` is available.

## Database Migration

### Export from SQLite (if needed)
```bash
# Run locally to export existing data
python scripts/export_sqlite_data.py
```

### Import to PostgreSQL
```bash
# Run once PostgreSQL is set up
python scripts/import_to_postgresql.py
```

## Connection Details

### Private Network Connection
Services within the same Railway project can connect via private network:
```
postgresql://postgres:password@postgres.railway.internal:5432/railway
```

### Public Connection (External Access)
For external connections (development, debugging):
```
postgresql://postgres:password@host.proxy.railway.app:port/railway
```

## Configuration Management

The system uses a configuration hierarchy:
1. Railway Provider (checks Railway environment variables)
2. AWS Secrets Manager (if available)
3. Environment variables
4. .env file

This ensures:
- Railway deployment uses PostgreSQL automatically
- Local development continues using SQLite
- Seamless switching between environments

## Benefits

### Performance
- **Concurrent Access**: Multiple services can safely access the same database
- **Connection Pooling**: Efficient database connection management
- **Query Optimization**: PostgreSQL's advanced query planner

### Reliability
- **Data Consistency**: ACID transactions ensure data integrity
- **Backup/Recovery**: Railway's built-in backup features
- **High Availability**: PostgreSQL's proven reliability

### Scalability
- **Horizontal Scaling**: Services can scale independently
- **Data Growth**: PostgreSQL handles large datasets efficiently
- **Query Performance**: Optimized for complex analytical queries

### Cost Efficiency
- **Single Database**: No duplication of database resources
- **Shared Costs**: Database costs shared across services
- **Efficient Resource Usage**: Better resource utilization

## Monitoring and Maintenance

### Database Health
- Railway provides built-in database monitoring
- Connection metrics and query performance
- Built-in database UI for management

### Backup Strategy
- Railway's native backup feature
- Automated daily backups
- Point-in-time recovery capabilities

### Security
- Encrypted connections (SSL/TLS)
- Network isolation within Railway project
- Environment variable security

## Troubleshooting

### Connection Issues
1. Check DATABASE_URL environment variable
2. Verify PostgreSQL service is running
3. Check private network connectivity

### Performance Issues
1. Monitor connection pool usage
2. Analyze slow queries
3. Check database resource usage

### Data Consistency
1. Ensure proper transaction management
2. Use database locks for concurrent operations
3. Monitor for deadlocks

## Cost Considerations

### PostgreSQL Costs
- **Compute**: Based on CPU/RAM usage
- **Storage**: Based on data size
- **Network**: Minimal within private network

### Cost Optimization
- **App Sleeping**: Enable for non-critical environments
- **Resource Sizing**: Right-size database resources
- **Connection Pooling**: Reduce connection overhead

## Security Best Practices

1. **Environment Variables**: Never commit database credentials
2. **Network Access**: Use private network for internal communication
3. **SSL/TLS**: Always use encrypted connections
4. **Access Control**: Limit database access permissions
5. **Monitoring**: Track database access and queries

---

This centralized database approach provides a robust, scalable, and maintainable solution for the AI Trading Bot system on Railway.