# Database Backup & Restore Policy

_Last updated: 14-Oct-2025_

This document defines how the **AI-Trading-Bot** project backs-up,
retains and restores its PostgreSQL database along with the recovery‐time
objectives (RTO) and recovery-point objectives (RPO).

---

## 1  Automated Backups

| Item | Value |
|------|-------|
| Tool | `atb db backup` (pg_dump → local backup) |
| Schedule | Every 1 h via Railway Scheduled Jobs |
| Destination | Local backup directory `${BACKUP_DIR}` |
| Format | `pg_dump` custom (`-Fc`) – compressed, schema-aware |
| Encryption | Local file system security |
| Retention | 7 days (configurable `BACKUP_RETENTION_DAYS`) |

Backups are stored under the path pattern:

```
backups/<db_name>/<YYYY>/<MM>/<DD>/backup-<timestamp>.dump
```

### 1.1 Environment variables

```
DATABASE_URL           # Postgres connection string
BACKUP_DIR      # Local backup directory
BACKUP_RETENTION_DAYS  # Optional, default 7
```

---

## 2  Restore Procedure

1. Download the desired backup file from backup directory:

   ```bash
   cp ./backups/<db>/<date>/backup-<ts>.dump /tmp/
   ```

2. (Optionally) create a fresh database to restore into:

   ```bash
   createdb mydb_restore
   ```

3. Restore using `pg_restore`:

   ```bash
   pg_restore -d mydb_restore --clean --create /tmp/backup-<ts>.dump
   ```

4. Run Alembic to ensure the schema is at the latest revision:

   ```bash
   alembic upgrade head
   ```

5. Point the application (e.g. in Railway) at the restored DB.

---

## 3  Monitoring & Heartbeats

• The heartbeat functionality is available via `atb tests heartbeat` and is run every 5 minutes by the
  scheduler. It inserts a `SystemEvent` row of type `TEST` with the
  message "Heartbeat".

• Alerts are triggered if **no** heartbeat is seen for > 15 minutes.
  (Configured in external monitoring platform.)

---

## 4  Recovery Objectives

| Term | Target |
|------|--------|
| **RPO** (data loss) | ≤ 15 minutes (hourly backups + WAL archiving) |
| **RTO** (downtime)  | ≤ 60 minutes (restore + redeploy) |

RPO is bounded by scheduled backups; if stricter RPO is required in the
future we will enable continuous PITR exports.

---

## 5  Testing Backups

A restore drill is performed **quarterly**:

1. A random backup from the past 90 days is restored into a sandbox DB.
2. Integration tests (`tests/run_tests.py all`) are executed against it.
3. Results are logged in the audit wiki.

---

## 6  Future Improvements

1. Use PostgreSQL logical replication to a warm-standby instance in a
different AWS region for sub-5-minute RPO.
2. Automatically verify backup integrity via `pg_restore --list`.
3. Add `wal-g` for continuous WAL shipping instead of hourly dumps.
