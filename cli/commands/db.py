from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

# Ensure project root and src are in sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

from alembic.config import Config  # type: ignore
from alembic.script import ScriptDirectory  # type: ignore
from sqlalchemy import create_engine, text  # type: ignore
from sqlalchemy import inspect as sa_inspect  # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore

from src.database.models import Base


# * Constants for Alembic/DB
MIN_POSTGRESQL_URL_PREFIX = "postgresql"
ALEMBIC_INI_PATH = str(PROJECT_ROOT / "alembic.ini")
MIGRATIONS_PATH = str(PROJECT_ROOT / "migrations")


def _print_header(title: str) -> None:
    print("\n" + title)
    print("=" * max(8, len(title)))


def _safe_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _resolve_database_url() -> str:
    from src.config.config_manager import get_config

    cfg = get_config()
    database_url: Optional[str] = cfg.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required but not set.")
    if not database_url.startswith(MIN_POSTGRESQL_URL_PREFIX):
        raise RuntimeError(
            f"Unsupported DATABASE_URL scheme. Expected '{MIN_POSTGRESQL_URL_PREFIX}://'."
        )
    return database_url


def _alembic_config(db_url: str) -> Config:
    if not os.path.exists(ALEMBIC_INI_PATH):
        raise RuntimeError(f"Missing alembic.ini at {ALEMBIC_INI_PATH}")
    cfg = Config(ALEMBIC_INI_PATH)
    cfg.set_main_option("script_location", MIGRATIONS_PATH)
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def _get_alembic_status(cfg: Config) -> dict[str, Any]:
    from alembic.runtime.migration import MigrationContext  # type: ignore

    script = ScriptDirectory.from_config(cfg)
    heads = list(script.get_heads())
    db_url = cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10, "application_name": "ai-trading-bot:alembic-status"},
    )
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            current = ctx.get_current_revision()
        start_point = current or "base"
        upgrade_path = list(script.get_upgrade_revs("head", start_point))  # type: ignore[arg-type]
        pending = [rev.revision for rev in reversed(upgrade_path)]
        if current is None:
            state = "unversioned"
        elif len(heads) > 1:
            state = "branching"
        elif pending:
            state = "outdated"
        else:
            state = "ok"
        return {"heads": heads, "current": current, "pending": pending, "state": state}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def _apply_migrations(cfg: Config) -> bool:
    try:
        from alembic import command  # type: ignore

        command.upgrade(cfg, "head")
        print("✅ Alembic migrations applied to head.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to apply migrations: {exc}")
        traceback.print_exc()
        return False


def _expected_schema_from_models() -> dict[str, Any]:
    expected_tables: list[str] = []
    expected_columns: dict[str, list[str]] = {}
    expected_pk: dict[str, list[str]] = {}
    expected_types: dict[str, dict[str, Any]] = {}

    for table in Base.metadata.sorted_tables:
        name = str(table.name)
        expected_tables.append(name)
        expected_columns[name] = [str(col.name) for col in table.columns]
        expected_pk[name] = [str(col.name) for col in table.primary_key.columns]
        expected_types[name] = {str(col.name): col.type for col in table.columns}

    return {
        "tables": expected_tables,
        "columns": expected_columns,
        "primary_keys": expected_pk,
        "types": expected_types,
    }


def _basic_integrity_checks(db_url: str) -> dict[str, Any]:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10, "application_name": "ai-trading-bot:deploy-check"},
    )

    expected = _expected_schema_from_models()
    expected_tables: list[str] = expected["tables"]

    results: dict[str, Any] = {
        "connectivity": False,
        "tables_exist": {},
        "row_counts": {},
        "alembic_version_present": False,
    }

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            results["connectivity"] = True
            insp = sa_inspect(conn)
            actual_tables = set(insp.get_table_names(schema="public"))
            for t in expected_tables:
                results["tables_exist"][t] = t in actual_tables
            results["alembic_version_present"] = "alembic_version" in actual_tables
            for t in expected_tables:
                if t in actual_tables:
                    try:
                        count_q = text(f"SELECT COUNT(*) FROM {t}")
                        results["row_counts"][t] = int(conn.execute(count_q).scalar() or 0)
                    except SQLAlchemyError:
                        results["row_counts"][t] = None
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Database connectivity/integrity check failed: {exc}")
        traceback.print_exc()
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    return results


def _verify_schema(db_url: str) -> dict[str, Any]:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10, "application_name": "ai-trading-bot:schema-verify"},
    )

    result: dict[str, Any] = {
        "ok": True,
        "missing_tables": [],
        "extra_tables": [],
        "missing_columns": {},
        "extra_columns": {},
        "primary_key_mismatches": {},
        "type_mismatches": {},
    }

    expected = _expected_schema_from_models()

    try:
        insp = sa_inspect(engine)
        actual_tables = set(insp.get_table_names(schema="public"))
        expected_tables = set(expected["tables"])
        missing_tables = sorted(expected_tables - actual_tables)
        extra_tables = sorted(actual_tables - expected_tables)
        if missing_tables or extra_tables:
            result["ok"] = False
            result["missing_tables"] = missing_tables
            result["extra_tables"] = extra_tables

        for table in sorted(expected["tables"]):
            if table not in actual_tables:
                continue
            expected_cols = set(expected["columns"][table])
            col_info = insp.get_columns(table, schema="public")
            actual_cols = {c["name"] for c in col_info}
            miss_cols = sorted(expected_cols - actual_cols)
            extra_cols = sorted(actual_cols - expected_cols)
            if miss_cols:
                result["ok"] = False
                result.setdefault("missing_columns", {})[table] = miss_cols
            if extra_cols:
                result.setdefault("extra_columns", {})[table] = extra_cols

            try:
                pk = insp.get_pk_constraint(table, schema="public") or {}
                actual_pk = sorted(pk.get("constrained_columns") or [])
            except Exception:
                actual_pk = []
            exp_pk = sorted(expected["primary_keys"].get(table, []))
            if actual_pk != exp_pk:
                result["ok"] = False
                result.setdefault("primary_key_mismatches", {})[table] = {
                    "expected": exp_pk,
                    "actual": actual_pk,
                }

            actual_types_map: dict[str, str] = {}
            for c in col_info:
                try:
                    compiled = c["type"].compile(dialect=engine.dialect)  # type: ignore[attr-defined]
                    actual_types_map[c["name"]] = str(compiled).lower().replace(" ", "")
                except Exception:
                    actual_types_map[c["name"]] = str(c["type"]).lower().replace(" ", "")

            for col_name, exp_type in expected["types"][table].items():
                if col_name not in actual_types_map:
                    continue
                try:
                    compiled_exp = exp_type.compile(dialect=engine.dialect)  # type: ignore[attr-defined]
                    exp_str = str(compiled_exp).lower().replace(" ", "")
                except Exception:
                    exp_str = str(exp_type).lower().replace(" ", "")
                act_str = actual_types_map.get(col_name, "")
                def _norm(s: str) -> str:
                    return (
                        s.replace("jsonb", "json")
                        .replace("timestampwithouttimezone", "timestamp")
                        .replace("doubleprecision", "float8")
                    )
                if _norm(exp_str) != _norm(act_str):
                    result["ok"] = False
                    result.setdefault("type_mismatches", {}).setdefault(table, {})[col_name] = {
                        "expected": exp_str,
                        "actual": act_str,
                    }
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    return result

def _migrate(ns: argparse.Namespace) -> int:
    """Run database migrations"""
    
    print("🔄 Running database migrations...")
    
    try:
        # Check if DATABASE_URL is available
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL environment variable not found")
            print("   Please ensure your Railway PostgreSQL service is properly configured")
            return 1
        
        print(f"✅ Database URL found: {database_url[:20]}...")
        
        if ns.check:
            # Just check migration status
            result = subprocess.run(
                ["alembic", "current"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Current migration status:")
                print(result.stdout)
                return 0
            else:
                print("❌ Failed to check migration status:")
                print(result.stderr)
                return 1
        
        # Run alembic upgrade
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Database migrations completed successfully")
            if result.stdout:
                print("📋 Migration output:")
                print(result.stdout)
            return 0
        else:
            print("❌ Database migrations failed")
            print("📋 Error output:")
            print(result.stderr)
            return 1
            
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return 1


def _verify(ns: argparse.Namespace) -> int:
    print("🔍 Database Integrity & Migration Check")
    print("=" * 50)
    try:
        db_url = _resolve_database_url()
        print(f"📦 Database URL resolved (postgres): {db_url}")

        print("\nConnectivity & Integrity")
        print("-----------------------")
        integrity = _basic_integrity_checks(db_url)
        if not integrity.get("connectivity"):
            print("❌ Cannot connect to database. Aborting further checks.")
            return 2
        print("✅ Connectivity OK")
        print("📊 Tables found:")
        for name, exists in sorted(integrity.get("tables_exist", {}).items()):
            print(f"  - {name}: {'present' if exists else 'missing'}")
        if integrity.get("row_counts"):
            print("📈 Row counts (approx):")
            for name, count in sorted(integrity["row_counts"].items()):
                if count is None:
                    print(f"  - {name}: n/a")
                else:
                    print(f"  - {name}: {count}")

        print("\nAlembic Migration Status")
        print("------------------------")
        cfg = _alembic_config(db_url)
        status = _get_alembic_status(cfg)
        current = status.get("current")
        heads = status.get("heads", [])
        pending = status.get("pending", [])
        state = status.get("state")
        print(f"Current revision: {current or 'none'}")
        print(f"Head(s): {', '.join(heads) if heads else 'none'}")
        print(f"Pending revisions: {len(pending)}")
        if pending:
            for rev in pending:
                print(f"  - pending: {rev}")
        if state == "unversioned":
            print("⚠️  Database is not versioned by Alembic (missing alembic_version).")
        elif state in {"outdated", "branching"}:
            print("⚠️  Database revision is outdated or divergent from head(s).")
        else:
            print("✅ Alembic state consistent.")

        if pending and ns.apply_migrations:
            print("\nApplying Migrations")
            print("-------------------")
            ok = _apply_migrations(cfg)
            if not ok:
                return 1
        elif pending and not ns.apply_migrations:
            print(
                "ℹ️  Pending migrations detected. To apply automatically, pass --apply-migrations or set ATB_AUTO_APPLY_MIGRATIONS=true."
            )

        print("\nSchema Verification Against Models")
        print("----------------------------------")
        verify = _verify_schema(db_url)
        if verify.get("ok"):
            print("✅ Schema matches SQLAlchemy models.")
            return 0
        else:
            print("❌ Schema deviations detected:")
            if verify.get("missing_tables"):
                print("  - Missing tables:")
                for t in verify["missing_tables"]:
                    print(f"    • {t}")
            if verify.get("extra_tables"):
                print("  - Extra tables (not defined in models):")
                for t in verify["extra_tables"]:
                    print(f"    • {t}")
            if verify.get("missing_columns"):
                print("  - Missing columns:")
                for tbl, cols in verify["missing_columns"].items():
                    print(f"    • {tbl}: {', '.join(cols)}")
            if verify.get("extra_columns"):
                print("  - Extra columns:")
                for tbl, cols in verify["extra_columns"].items():
                    print(f"    • {tbl}: {', '.join(cols)}")
            if verify.get("primary_key_mismatches"):
                print("  - Primary key mismatches:")
                for tbl, info in verify["primary_key_mismatches"].items():
                    print(f"    • {tbl}: expected={info['expected']} actual={info['actual']}")
            if verify.get("type_mismatches"):
                print("  - Column type mismatches:")
                for tbl, cols in verify["type_mismatches"].items():
                    for col, info in cols.items():
                        print(f"    • {tbl}.{col}: expected={info['expected']} actual={info['actual']}")
            return 1
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        traceback.print_exc()
        return 1


def _backup(ns: argparse.Namespace) -> int:
    backup_dir = ns.backup_dir
    retention_days = ns.retention
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set", file=sys.stderr)
        return 1

    parsed = urlparse(db_url)
    dbname = parsed.path.lstrip("/")
    password = parsed.password or ""

    timestamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = Path(backup_dir) / dbname / _dt.datetime.utcnow().strftime("%Y/%m/%d")
    dump_filename = f"backup-{timestamp}.dump"
    dump_path = backup_path / dump_filename

    backup_path.mkdir(parents=True, exist_ok=True)
    print(f"📦 Creating dump: {dump_path}")

    env = os.environ.copy()
    env["PGPASSWORD"] = password
    cmd = ["pg_dump", f"--dbname={db_url}", "-Fc", "-Z", "9", "-f", str(dump_path)]
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ pg_dump failed: {exc.stderr.decode()}", file=sys.stderr)
        return 1

    print("✅ Backup created successfully")
    if retention_days > 0:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=retention_days)
        print(f"🧹 Deleting backups older than {retention_days} days (before {cutoff.date()})")
        db_backup_dir = Path(backup_dir) / dbname
        if db_backup_dir.exists():
            for backup_file in db_backup_dir.rglob("backup-*.dump"):
                try:
                    file_time = _dt.datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff:
                        print(f"   • Removing {backup_file}")
                        backup_file.unlink()
                except OSError as e:
                    print(f"   • Warning: Could not check/remove {backup_file}: {e}")
    return 0


def _setup_railway(ns: argparse.Namespace) -> int:
    from src.config.config_manager import get_config
    from src.database.manager import DatabaseManager

    if ns.verify:
        # Verify Railway setup
        try:
            config = get_config()
            print("📊 Environment Check:")
            railway_project = config.get("RAILWAY_PROJECT_ID")
            database_url = config.get("DATABASE_URL")
            print(
                f"✅ Railway Project ID: {railway_project}"
                if railway_project
                else "⚠️  Not running on Railway (RAILWAY_PROJECT_ID not found)"
            )
            if not database_url:
                print("❌ DATABASE_URL not found")
                return 1
            if not database_url.startswith("postgresql"):
                print("❌ Database URL does not start with 'postgresql://'")
                return 1
            print(f"✅ Database URL: {database_url}")
            print("🔗 Testing PostgreSQL Database Connection...")
            db_manager = DatabaseManager()
            db_info = db_manager.get_database_info()
            print(f"  Database Type: {db_info['database_type']}")
            print(f"  Connection Pool Size: {db_info['connection_pool_size']}")
            if not db_manager.test_connection():
                print("❌ PostgreSQL database connection failed!")
                return 1
            print("✅ PostgreSQL database connection successful!")
            # Minimal ops validation
            session_id = db_manager.create_trading_session(
                strategy_name="VerificationTest",
                symbol="BTCUSDT",
                timeframe="1h",
                mode="PAPER",
                initial_balance=10000.0,
                session_name="railway_verification",
            )
            print(f"✅ Created test session #{session_id}")
            db_manager.end_trading_session(session_id, final_balance=10000.0)
            print(f"✅ Ended test session #{session_id}")
            print("✅ Railway PostgreSQL database setup verification completed successfully!")
            return 0
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            return 1

    if ns.check_local:
        try:
            config = get_config()
            database_url = config.get("DATABASE_URL")
            if not database_url:
                print("❌ DATABASE_URL not set")
                return 1
            if not database_url.startswith("postgresql"):
                print("❌ DATABASE_URL does not start with 'postgresql://'")
                return 1
            db_manager = DatabaseManager()
            if not db_manager.test_connection():
                print("❌ Local PostgreSQL connection failed")
                return 1
            print("✅ Local PostgreSQL connection successful")
            return 0
        except Exception as e:
            print(f"❌ Error checking local development setup: {e}")
            return 1

    # Default: print instructions
    print("🚀 Railway PostgreSQL Database Setup")
    print("=" * 60)
    print(
        "\n📋 Step-by-Step Instructions:\n... (see docs/RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md)"
    )
    return 0


def _reset_railway(ns: argparse.Namespace) -> int:
    import psycopg2
    from psycopg2 import sql

    env = ns.env
    db_url = os.environ.get(
        "RAILWAY_STAGING_DATABASE_URL" if env == "staging" else "RAILWAY_PRODUCTION_DATABASE_URL"
    )
    if not db_url:
        print(
            f"Database URL for {env} not set. Please set the {'RAILWAY_STAGING_DATABASE_URL' if env == 'staging' else 'RAILWAY_PRODUCTION_DATABASE_URL'} environment variable."
        )
        return 1
    if env == "production" and not ns.yes:
        confirm = input(
            "Are you sure you want to reset the PRODUCTION database? Type 'YES' to continue: "
        )
        if confirm != "YES":
            print("Aborted.")
            return 0
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = cur.fetchall()
            if not tables:
                print("No tables found.")
            table_names = [sql.Identifier(t[0]) for t in tables]
            cur.execute("SET session_replication_role = 'replica';")
            for t in table_names:
                cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(t))
            cur.execute("SET session_replication_role = 'origin';")
        conn.commit()
        print(f"All tables in {env} database have been reset (structure preserved).")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("db", help="Database utilities")
    sub = p.add_subparsers(dest="db_cmd", required=True)

    p_verify = sub.add_parser("verify", help="Verify database integrity and migrations")
    p_verify.add_argument(
        "--apply-migrations",
        action="store_true",
        help="Apply Alembic migrations to head if pending",
    )
    p_verify.set_defaults(func=_verify)

    p_migrate = sub.add_parser("migrate", help="Run database migrations")
    p_migrate.add_argument("--check", action="store_true", help="Check migration status only")
    p_migrate.set_defaults(func=_migrate)

    p_backup = sub.add_parser("backup", help="Backup database")
    p_backup.add_argument("--backup-dir", default=os.getenv("BACKUP_DIR", "./backups"))
    p_backup.add_argument(
        "--retention", type=int, default=int(os.getenv("BACKUP_RETENTION_DAYS", 7))
    )
    p_backup.set_defaults(func=_backup)

    p_reset = sub.add_parser("reset-railway", help="Reset Railway database")
    p_reset.add_argument("env", choices=["staging", "production"], help="Target environment")
    p_reset.add_argument("--yes", action="store_true", help="Skip confirmation (for production)")
    p_reset.set_defaults(func=_reset_railway)

    p_setup = sub.add_parser("setup-railway", help="Setup/verify Railway database")
    p_setup.add_argument("--verify", action="store_true")
    p_setup.add_argument("--check-local", action="store_true")
    p_setup.set_defaults(func=_setup_railway)
