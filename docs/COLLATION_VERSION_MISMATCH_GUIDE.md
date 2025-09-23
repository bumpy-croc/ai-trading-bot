# PostgreSQL Collation Version Mismatch Guide

## Overview

This guide addresses the PostgreSQL collation version mismatch warning and provides strategies to prevent it in future deployments.

## What is the Collation Version Mismatch Issue?

PostgreSQL collation version mismatches occur when:
- A database is created using one version of ICU (International Components for Unicode) collation data
- The PostgreSQL server is later upgraded to use a different version of ICU collation data
- This causes collation version mismatches between the database and the operating system

**Error Message:**
```
WARNING: database "railway" has a collation version mismatch
DETAIL: The database was created using collation version 2.36, but the operating system provides version 2.41.
HINT: Rebuild all objects in this database that use the default collation and run ALTER DATABASE railway REFRESH COLLATION VERSION, or build PostgreSQL with the right library version.
```

## Immediate Fix

### Using the CLI Tool

The trading bot now includes built-in commands to fix collation issues:

```bash
# Check collation status
atb db check-collation

# Fix collation version mismatch
atb db fix-collation
```

### Manual Fix (PostgreSQL 15+)

If you have direct database access:

```sql
-- Method 1: Refresh collation version (PostgreSQL 15+)
ALTER DATABASE your_database_name REFRESH COLLATION VERSION;
```

### Manual Fix (All PostgreSQL Versions)

For older PostgreSQL versions or when REFRESH COLLATION VERSION isn't available:

```sql
-- Method 2: Rebuild objects with explicit collation
-- This rebuilds all tables with explicit collation specifications

-- For each affected table:
ALTER TABLE your_table ALTER COLUMN text_column TYPE TEXT COLLATE "en_US.UTF-8";
```

## Prevention Strategies

### 1. Use Explicit Collations in Schema Definitions

**❌ Bad (uses default collation):**
```sql
CREATE TABLE users (
    name TEXT,
    email TEXT
);
```

**✅ Good (explicit collation):**
```sql
CREATE TABLE users (
    name TEXT COLLATE "en_US.UTF-8",
    email TEXT COLLATE "en_US.UTF-8"
);
```

### 2. Update Your SQLAlchemy Models

In your SQLAlchemy models, specify collations explicitly:

```python
from sqlalchemy import Column, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)

    # In SQLAlchemy, you can specify collation in column definitions
    # The collation will be applied when creating the table
```

For more complex collation specifications, you can use:

```python
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import VARCHAR

class User(Base):
    __tablename__ = 'users'

    name = Column(VARCHAR(100, collation='en_US.UTF-8'), nullable=False)
    email = Column(VARCHAR(255, collation='en_US.UTF-8'), nullable=False)
```

### 3. Database Migration Strategy

Create a migration to update existing collations:

```python
# In your Alembic migration file
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Update collation for existing text columns
    op.execute('ALTER TABLE users ALTER COLUMN name TYPE TEXT COLLATE "en_US.UTF-8"')
    op.execute('ALTER TABLE users ALTER COLUMN email TYPE TEXT COLLATE "en_US.UTF-8"')

    # Add this for other tables as needed
    # op.execute('ALTER TABLE posts ALTER COLUMN title TYPE TEXT COLLATE "en_US.UTF-8"')
    # op.execute('ALTER TABLE posts ALTER COLUMN content TYPE TEXT COLLATE "en_US.UTF-8"')

def downgrade():
    # Revert to default collation (not recommended for production)
    op.execute('ALTER TABLE users ALTER COLUMN name TYPE TEXT')
    op.execute('ALTER TABLE users ALTER COLUMN email TYPE TEXT')
```

### 4. Railway-Specific Prevention

#### Environment Consistency
- **Use the same PostgreSQL version** across all Railway environments
- **Pin PostgreSQL version** in your Railway configuration
- **Test deployments** on staging before production

#### Railway Configuration

Create a `railway.json` file to specify PostgreSQL version:

```json
{
  "build": {
    "builder": "dockerfile"
  },
  "deploy": {
    "preDeployCommand": "atb db verify --apply-migrations",
    "startCommand": "atb live-health ml_basic",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 10,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  },
  "databases": [
    {
      "name": "postgresql",
      "version": "15"
    }
  ]
}
```

#### Environment Variables

Set Railway environment variables to ensure consistency:

```bash
# Pin PostgreSQL version
railway variables set POSTGRESQL_VERSION=15

# Set locale explicitly
railway variables set LC_COLLATE=en_US.UTF-8
railway variables set LC_CTYPE=en_US.UTF-8
```

### 5. Docker-Based Prevention

If using Docker for local development:

```dockerfile
FROM postgres:15-alpine

# Set explicit locale
ENV LANG en_US.UTF-8
ENV LC_COLLATE en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8

# Pin ICU version to prevent collation mismatches
RUN apk add --no-cache icu-data-full=72.1-r1
```

### 6. CI/CD Pipeline Checks

Add collation checks to your CI/CD pipeline:

```yaml
# In your .github/workflows/deploy.yml
- name: Check PostgreSQL Collation
  run: |
    atb db check-collation
    if [ $? -ne 0 ]; then
      echo "Collation mismatch detected!"
      exit 1
    fi
```

### 7. Monitoring and Alerts

#### Application-Level Monitoring

Add collation monitoring to your health checks:

```python
# In your health check endpoint
def check_database_collation():
    """Check for collation version mismatches."""
    try:
        # Query to detect collation issues
        result = db.session.execute("""
            SELECT COUNT(*) as mismatch_count
            FROM pg_database
            WHERE datcollate != (SELECT setting FROM pg_settings WHERE name = 'lc_collate')
        """).scalar()

        if result > 0:
            return {"status": "warning", "message": "Collation version mismatch detected"}
        return {"status": "ok", "message": "Collation version is consistent"}

    except Exception as e:
        return {"status": "error", "message": f"Could not check collation: {e}"}
```

#### Railway Monitoring

Set up Railway monitoring for collation warnings:

```bash
# Check logs for collation warnings
railway logs | grep -i collation

# Set up alerts for collation-related errors
railway variables set LOG_COLLATION_WARNINGS=true
```

## Troubleshooting

### Common Issues and Solutions

#### 1. REFRESH COLLATION VERSION Not Available

**Problem:** PostgreSQL version < 15 doesn't support `ALTER DATABASE ... REFRESH COLLATION VERSION`

**Solution:**
```sql
-- Use the rebuild approach for older versions
-- The CLI tool handles this automatically
atb db fix-collation
```

#### 2. Permission Denied

**Problem:** User doesn't have permission to alter database collation

**Solution:**
```sql
-- Connect as superuser or database owner
GRANT ALL PRIVILEGES ON DATABASE your_database TO your_user;
```

#### 3. Long-Running Rebuild Operations

**Problem:** Rebuilding large tables takes too long

**Solution:**
```sql
-- Rebuild in smaller batches
-- Use the CLI tool which handles this gracefully
atb db fix-collation
```

#### 4. Application Errors During Rebuild

**Problem:** Application fails during table rebuild due to locks

**Solution:**
- Schedule maintenance window for collation fixes
- Use the CLI tool during low-traffic periods
- Consider temporary read-only mode during rebuild

### Verification Steps

After applying fixes, verify the resolution:

```bash
# Check collation status
atb db check-collation

# Verify no warnings in application logs
grep -i "collation" /path/to/logs/*.log

# Test database operations
atb db verify
```

## Best Practices

### 1. Regular Maintenance

- **Monthly collation checks** in production
- **Weekly checks** in staging environments
- **Pre-deployment verification** before major releases

### 2. Documentation

- **Document collation specifications** in your schema documentation
- **Include collation requirements** in deployment checklists
- **Maintain version compatibility matrix** for PostgreSQL and ICU versions

### 3. Backup Strategy

- **Always backup before collation operations**
- **Test restore procedures** after collation changes
- **Keep multiple backup generations** during maintenance windows

### 4. Team Communication

- **Notify team members** before collation maintenance
- **Document incident response** for collation issues
- **Include collation checks** in deployment verification scripts

## Resources

- [PostgreSQL Collation Documentation](https://www.postgresql.org/docs/current/collation.html)
- [ICU Collation Version Compatibility](https://unicode-org.github.io/icu/userguide/collation/)
- [Railway PostgreSQL Documentation](https://docs.railway.app/databases/postgresql)
- [SQLAlchemy Collation Support](https://docs.sqlalchemy.org/en/14/dialects/postgresql.html#postgresql-specific-column-options)

## Support

If you encounter persistent collation issues:

1. **Check the current status:** `atb db check-collation`
2. **Review PostgreSQL version compatibility**
3. **Contact Railway support** for platform-specific issues
4. **Consider PostgreSQL upgrade** for long-term resolution

---

**Note:** Collation version mismatches are generally harmless warnings but should be addressed to prevent potential sorting and comparison inconsistencies in your application.
