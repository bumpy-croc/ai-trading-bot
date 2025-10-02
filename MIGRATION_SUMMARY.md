# Strategy Management System - Migration Summary

## Issue Addressed

The PR introduced new ORM tables (`StrategyRegistry`, `StrategyVersion`, `StrategyPerformance`, `StrategyLineage`) in `src/database/models.py` but lacked the corresponding Alembic migration. This would cause "relation does not exist" errors when attempting to persist strategies or performance data on existing deployments.

## Solution Implemented

### 1. Created Alembic Migration ✅

**File:** `migrations/versions/0002_add_strategy_management_tables.py`

This migration creates the complete strategy management database schema:

#### Tables Created:

1. **`strategy_registry`** - Core strategy metadata and configuration
   - Primary key: `id` (Integer)
   - Unique identifier: `strategy_id` (String, indexed)
   - Version tracking: `name`, `version` with unique constraint
   - Lineage: `parent_id`, `lineage_path`, `branch_name`, `merge_source`
   - Metadata: `created_at`, `created_by`, `description`, `tags`, `status`
   - Configurations: `signal_generator_config`, `risk_manager_config`, `position_sizer_config`, `regime_detector_config` (all JSONB)
   - Additional: `parameters`, `performance_summary`, `validation_results` (JSONB)
   - Integrity: `config_hash`, `component_hash`
   - Timestamps: `created_at_db`, `updated_at`

2. **`strategy_versions`** - Version history tracking
   - Links to: `strategy_registry.strategy_id`
   - Fields: `version`, `created_at`, `changes` (JSONB), `performance_delta` (JSONB), `is_major`, `component_changes` (JSONB), `parameter_changes` (JSONB)

3. **`strategy_performance`** - Performance metrics storage
   - Links to: `strategy_registry.strategy_id`
   - Period: `period_start`, `period_end`, `period_type`
   - Core metrics: `total_return`, `total_return_pct`, `sharpe_ratio`, `max_drawdown`, `win_rate`
   - Trade stats: `total_trades`, `winning_trades`, `losing_trades`, `avg_trade_duration_hours`
   - Risk metrics: `volatility`, `sortino_ratio`, `calmar_ratio`, `var_95`
   - Attribution: `signal_generator_contribution`, `risk_manager_contribution`, `position_sizer_contribution`
   - Additional: `regime_performance` (JSONB), `additional_metrics` (JSONB), test configuration fields

4. **`strategy_lineage`** - Evolutionary relationship tracking
   - Links: `ancestor_id` and `descendant_id` to `strategy_registry.strategy_id`
   - Fields: `relationship_type`, `generation_distance`, `created_at`, `evolution_reason`, `change_impact` (JSONB)
   - Performance: `performance_improvement`, `risk_change`

#### Enum Created:

- **`strategystatus`**: `'EXPERIMENTAL'`, `'TESTING'`, `'PRODUCTION'`, `'RETIRED'`, `'DEPRECATED'`

#### Indexes Created:

The migration includes comprehensive indexing for optimal query performance:
- Single column indexes on frequently queried fields
- Composite indexes for common query patterns
- Foreign key indexes for join optimization

### 2. Fixed Enum Value Mismatch ✅

**File:** `src/strategies/components/strategy_registry.py`

Changed `StrategyStatus` enum values from lowercase to uppercase to match the database enum:
- `experimental` → `EXPERIMENTAL`
- `testing` → `TESTING`
- `production` → `PRODUCTION`
- `retired` → `RETIRED`
- `deprecated` → `DEPRECATED`

This prevents `LookupError` when persisting strategies to the database.

### 3. Updated Test Files ✅

Updated all test references to use uppercase enum values:
- `tests/strategies/components/test_strategy_manager.py`
- `tests/strategies/components/test_strategy_registry.py`

Changed all occurrences of `'status': 'experimental'` to `'status': 'EXPERIMENTAL'`, etc.

## Migration Usage

### To Apply the Migration:

```bash
# Apply all pending migrations
alembic upgrade head

# Or apply this specific migration only
alembic upgrade 0002_strategy_management
```

### To Rollback (if needed):

```bash
# Rollback to previous migration
alembic downgrade 0001_initial_schema
```

### To Check Migration Status:

```bash
# Show current migration version
alembic current

# Show migration history
alembic history
```

## Migration Features

1. **Idempotent Enum Creation**: Checks if enum exists before creating to handle re-runs gracefully
2. **Foreign Key Constraints**: Properly defines relationships between tables
3. **JSONB Storage**: Uses PostgreSQL JSONB for flexible JSON storage with indexing support
4. **Comprehensive Indexing**: Creates all necessary indexes for performance
5. **Self-Referential Foreign Key**: `parent_id` in `strategy_registry` references same table
6. **Cascade Deletes**: Properly configured for child records (versions, performance, lineage)
7. **Complete Downgrade**: Full rollback support to remove all tables and enum

## Database Schema Diagram

```
strategy_registry (parent table)
    ├─→ strategy_versions (one-to-many)
    ├─→ strategy_performance (one-to-many)
    ├─→ strategy_lineage (many-to-many via ancestor/descendant)
    └─→ strategy_registry (self-referential for parent_id)
```

## Verification

The migration file has been validated:
- ✅ Python syntax is correct
- ✅ All required columns match ORM models
- ✅ Enum values match model definitions
- ✅ Foreign keys properly reference existing tables
- ✅ Indexes cover all common query patterns
- ✅ Both upgrade() and downgrade() functions implemented

## Next Steps

1. Review the migration file to ensure it matches your requirements
2. Test the migration on a development database
3. Apply to staging environment for validation
4. Deploy to production when ready

## Notes

- The migration creates PostgreSQL-specific JSONB columns
- Enum creation is conditional to support re-running migrations
- All timestamps use `datetime.utcnow` for consistency
- The migration is backwards compatible with existing schema
