# Plan: Write-Ahead Log (WAL) for Fault-Tolerant Database Persistence

## Overview

Implement a Write-Ahead Log system that ensures trading continues during PostgreSQL outages (up to hours) with eventual consistency. Critical operations are written to durable local storage before database writes, with background sync and automatic retry.

## Problem Statement

Currently, if a database write fails during trading (e.g., balance update after closing a position), the code logs an error but continues trading with inconsistent state. This can lead to:
- Incorrect position sizing (using stale balance)
- Over-leveraged positions
- Lost audit trail

## Solution: WAL Architecture

**Location**: `src/infrastructure/wal/` (follows existing infrastructure patterns)

**Approach**: Hybrid - Critical Operations Registry (Option D)
- Critical operations (trades, positions, balance) go through WAL
- Non-critical operations (analytics, metrics) go direct to DB
- WAL wraps DatabaseManager via composition (DurableDatabaseManager)

## Critical Operations (WAL-Protected)

| Operation | Method | Why Critical |
|-----------|--------|--------------|
| Balance changes | `atomic_balance_update` | Position sizing depends on accurate balance |
| Trade logging | `log_trade` | Audit trail, P&L calculation |
| Position creation | `log_position` | Track open positions |
| Position updates | `update_position` | Stop-loss, take-profit levels |
| Position closure | `close_position` | Realize P&L |
| Session management | `create/end_trading_session` | Session continuity |

## File Structure

```
src/infrastructure/wal/
├── __init__.py           # Public exports
├── entry.py              # WALEntry dataclass with idempotency
├── storage.py            # File-based storage with fsync
├── manager.py            # WALManager with background sync
└── durable_db.py         # DurableDatabaseManager wrapper
```

---

## Implementation Plan

### Phase 1: Core WAL Infrastructure

#### 1.1 Create `src/infrastructure/wal/entry.py`
- `OperationType` enum for critical operations
- `WALEntryStatus` enum (PENDING, SYNCING, COMPLETED, FAILED)
- `WALEntry` dataclass with:
  - UUID entry_id
  - Idempotency key (hash of operation + args)
  - Checksum for integrity verification
  - Timestamp tracking (created_at, synced_at)
  - Serialization methods (to_dict, from_dict)

#### 1.2 Create `src/infrastructure/wal/storage.py`
- `WALStorage` class with:
  - File-based append-only log (JSON lines format)
  - Atomic writes via `fcntl.flock()` + `os.fsync()`
  - File rotation when max size reached
  - Status update records (append-only updates)
  - `get_pending_entries()` - returns entries needing sync
  - `compact()` - removes old completed entries
  - `get_stats()` - monitoring metrics

#### 1.3 Create `src/infrastructure/wal/manager.py`
- `WALManager` class with:
  - Background sync thread (daemon)
  - Circuit breaker integration for DB availability
  - Exponential backoff with jitter
  - Idempotency cache (prevents duplicate processing)
  - `write()` - append to WAL + attempt immediate sync
  - `start()/stop()` - lifecycle management
  - `force_sync()` - manual sync trigger
  - `get_stats()` - comprehensive metrics

#### 1.4 Create `src/infrastructure/wal/__init__.py`
- Export public API

### Phase 2: Database Integration

#### 2.1 Create `src/infrastructure/wal/durable_db.py`
- `DurableDatabaseManager` class:
  - Wraps `DatabaseManager` via composition
  - Routes critical operations through WAL
  - Passes non-critical operations directly to DB
  - `atomic_balance_update()` - special handling for context manager
  - `__getattr__` - forwards unknown methods to underlying DB

#### 2.2 Update `src/engines/live/trading_engine.py`
- Import and use `DurableDatabaseManager`
- Call `db_manager.start()` in `start()`
- Call `db_manager.stop()` in `stop()`
- Add `get_wal_status()` method for monitoring

### Phase 3: Configuration & CLI

#### 3.1 Update `src/config/` for WAL settings
- Add to config loader:
  ```python
  WAL_ENABLED: bool = True
  WAL_DIR: str = "data/wal"
  WAL_SYNC_INTERVAL: float = 5.0
  WAL_MAX_RETRIES: int = 10
  WAL_RETENTION_HOURS: int = 168
  ```

#### 3.2 Add CLI commands in `cli/commands/`
- `atb wal status` - Show WAL stats and pending entries
- `atb wal sync` - Force immediate sync
- `atb wal compact` - Manual compaction

### Phase 4: Testing

#### 4.1 Create `tests/unit/infrastructure/wal/`
- `test_entry.py`:
  - Idempotency key generation
  - Checksum verification
  - Serialization round-trip
- `test_storage.py`:
  - Append and read entries
  - Status updates
  - File rotation
  - Compaction
  - Concurrent access (thread safety)
- `test_manager.py`:
  - Background sync worker
  - Circuit breaker integration
  - Exponential backoff
  - Idempotency (duplicate detection)
  - Start/stop lifecycle
- `test_durable_db.py`:
  - Critical operation routing
  - Non-critical passthrough
  - Context manager handling

#### 4.2 Create `tests/integration/wal/`
- `test_wal_recovery.py`:
  - Simulate DB outage
  - Verify entries persist to WAL
  - Verify sync after DB recovery
  - Verify idempotent replay

### Phase 5: Documentation

#### 5.1 Create `docs/wal.md`
- Overview and rationale
- Architecture diagram
- Configuration options
- CLI usage
- Monitoring and alerting
- Troubleshooting guide
- Recovery procedures

#### 5.2 Update `docs/architecture.md`
- Add WAL section to data flow diagram
- Document critical vs non-critical operations
- Add fault tolerance section

#### 5.3 Update `docs/database.md`
- Add WAL integration section
- Document durability guarantees
- Add recovery procedures

#### 5.4 Update `docs/live_trading.md`
- Document WAL startup/shutdown
- Add monitoring section for WAL status
- Document behavior during DB outages

#### 5.5 Update `CLAUDE.md`
- Add WAL to Essential Commands section
- Add WAL configuration to environment variables

---

## Key Design Decisions

### 1. File-based storage (not SQLite)
- Simpler, fewer dependencies
- JSON lines format is human-readable
- Easier debugging during development
- Can upgrade to SQLite later if needed

### 2. Append-only with status updates
- Never modify existing entries (crash-safe)
- Status changes are new records
- Compaction removes old completed entries

### 3. Idempotency via content hash
- Hash of (operation + args + kwargs)
- Prevents duplicate processing on replay
- Survives process restarts

### 4. Circuit breaker integration
- Prevents overwhelming failed DB
- Auto-recovery when DB returns
- Exponential backoff reduces load

### 5. Composition over modification
- DurableDatabaseManager wraps DatabaseManager
- No changes to core database code
- Easy to disable (set `WAL_ENABLED=false`)

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/infrastructure/wal/__init__.py` | Package exports |
| `src/infrastructure/wal/entry.py` | WAL entry dataclass |
| `src/infrastructure/wal/storage.py` | File-based storage |
| `src/infrastructure/wal/manager.py` | WAL manager with sync worker |
| `src/infrastructure/wal/durable_db.py` | DatabaseManager wrapper |
| `tests/unit/infrastructure/wal/test_entry.py` | Entry tests |
| `tests/unit/infrastructure/wal/test_storage.py` | Storage tests |
| `tests/unit/infrastructure/wal/test_manager.py` | Manager tests |
| `tests/unit/infrastructure/wal/test_durable_db.py` | Integration tests |
| `docs/wal.md` | WAL documentation |

## Files to Modify

| File | Changes |
|------|---------|
| `src/engines/live/trading_engine.py` | Use DurableDatabaseManager |
| `src/config/config_manager.py` | Add WAL config options |
| `cli/commands/dev.py` or new `wal.py` | Add WAL CLI commands |
| `docs/architecture.md` | Add WAL to architecture |
| `docs/database.md` | Add WAL section |
| `docs/live_trading.md` | Add WAL lifecycle |
| `CLAUDE.md` | Add WAL commands |

---

## Success Criteria

1. **Durability**: WAL entries survive process crash
2. **Consistency**: All critical operations sync to DB eventually
3. **Availability**: Trading continues during DB outage
4. **Idempotency**: Safe to replay entries (no duplicates)
5. **Observability**: WAL stats visible via CLI and monitoring
6. **Testability**: >90% test coverage on WAL module

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| WAL disk full | **Halt trading** (can't guarantee durability) |
| WAL file corruption | Checksum verification, skip corrupt entries |
| Long sync queue | Alert when pending > threshold |
| Memory growth | Bounded idempotency cache, periodic cleanup |

---

## User Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| WAL default state | **Enabled everywhere** | Consistent behavior, catches issues early |
| WAL write failure | **Halt trading** | No trade without durability guarantee |
| Branch strategy | **New feature branch** | `feature/wal-durability` for clean separation |

---

## Branch Strategy

```bash
# Create new branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/wal-durability
```

---

## Estimated Scope

- **New code**: ~800-1000 lines (WAL module)
- **Tests**: ~500-600 lines
- **Documentation**: ~300-400 lines
- **Modifications**: ~50-100 lines (trading engine, config)
