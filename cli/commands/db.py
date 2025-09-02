from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure project root and src are in sys.path for absolute imports
from src.utils.project_paths import get_project_root

PROJECT_ROOT = get_project_root()
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
# * Allow environment overrides for packaged/production deployments
ALEMBIC_INI_PATH = os.getenv("ATB_ALEMBIC_INI", str(PROJECT_ROOT / "alembic.ini"))
MIGRATIONS_PATH = os.getenv("ATB_MIGRATIONS_PATH", str(PROJECT_ROOT / "migrations"))


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
    database_url: str | None = cfg.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required but not set.")
    if not database_url.startswith(MIN_POSTGRESQL_URL_PREFIX):
        raise RuntimeError(
            f"Unsupported DATABASE_URL scheme. Expected '{MIN_POSTGRESQL_URL_PREFIX}://'."
        )
    return database_url


def _alembic_config(db_url: str) -> Config:
    """Create an Alembic Config, tolerating missing alembic.ini in production.

    When ``alembic.ini`` is not available (e.g., installed package), fall back to
    a programmatic configuration using ``MIGRATIONS_PATH`` and the resolved DB URL.
    """
    if os.path.exists(ALEMBIC_INI_PATH):
        cfg = Config(ALEMBIC_INI_PATH)
    else:
        # * Programmatic config: no ini file present
        cfg = Config()
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
        # * Compute pending upgrade revisions using supported Alembic API
        upgrade_path = list(script.iterate_revisions("head", start_point))
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

        print("🔄 Applying migrations to head...")
        command.upgrade(cfg, "head")
        print("✅ Alembic migrations applied successfully to head.")

        # Verify migration status after applying
        from alembic.runtime.migration import MigrationContext  # type: ignore
        script = ScriptDirectory.from_config(cfg)
        db_url = cfg.get_main_option("sqlalchemy.url")
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10, "application_name": "ai-trading-bot:post-migration-verify"},
        )
        try:
            with engine.connect() as conn:
                ctx = MigrationContext.configure(conn)
                current = ctx.get_current_revision()
                heads = list(script.get_heads())
                if current in heads:
                    print("✅ Migration verification: database is at head revision")
                else:
                    print(f"⚠️  Migration verification: current={current}, expected one of {heads}")
        finally:
            engine.dispose()

        return True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to apply migrations: {exc}")
        print("\n💡 Troubleshooting tips:")
        print("   • Check if columns/tables already exist (duplicate column error)")
        print("   • Verify database permissions")
        print("   • Check database connectivity")
        print("   • Review recent schema changes")
        traceback.print_exc()
        return False


def _expected_schema_from_models() -> dict[str, Any]:
    expected_tables: list[str] = []
    expected_columns: dict[str, list[str]] = {}
    expected_pk: dict[str, list[str]] = {}
    expected_types: dict[str, dict[str, Any]] = {}
    expected_nullable: dict[str, dict[str, bool]] = {}
    expected_indexes: dict[str, list[dict[str, Any]]] = {}

    for table in Base.metadata.sorted_tables:
        name = str(table.name)
        expected_tables.append(name)
        expected_columns[name] = [str(col.name) for col in table.columns]
        expected_pk[name] = [str(col.name) for col in table.primary_key.columns]
        expected_types[name] = {str(col.name): col.type for col in table.columns}
        expected_nullable[name] = {str(col.name): bool(col.nullable) for col in table.columns}
        # Capture non-unique indexes defined in models
        model_indexes: list[dict[str, Any]] = []
        for idx in table.indexes:
            try:
                idx_name = str(idx.name) if idx.name else None
                cols = [str(c.name) for c in idx.expressions]
                unique = bool(getattr(idx, "unique", False))
                if idx_name and not unique and cols:
                    model_indexes.append({"name": idx_name, "columns": cols, "unique": unique})
            except Exception:
                continue
        expected_indexes[name] = model_indexes

    return {
        "tables": expected_tables,
        "columns": expected_columns,
        "primary_keys": expected_pk,
        "types": expected_types,
        "nullable": expected_nullable,
        "indexes": expected_indexes,
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
        # Ignore Alembic's internal versioning table in comparisons
        system_tables = {"alembic_version"}
        actual_tables_no_sys = actual_tables - system_tables
        expected_tables = set(expected["tables"])
        missing_tables = sorted(expected_tables - actual_tables_no_sys)
        extra_tables = sorted(actual_tables_no_sys - expected_tables)
        if missing_tables or extra_tables:
            result["ok"] = False
            result["missing_tables"] = missing_tables
            result["extra_tables"] = extra_tables

        for table in sorted(expected["tables"]):
            if table not in actual_tables_no_sys:
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
                        s
                        .replace("jsonb", "json")
                        .replace("timestampwithouttimezone", "timestamp")
                        # Normalize PostgreSQL float synonyms
                        .replace("doubleprecision", "float")
                        .replace("float8", "float")
                        .replace("float4", "float")
                        .replace("real", "float")
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


def _apply_safe_fixes(db_url: str, verify: dict[str, Any], expected: dict[str, Any]) -> None:
    """Apply safe schema fixes:
    - Create missing non-unique indexes from models
    - Convert JSON columns to JSONB on PostgreSQL
    - Add missing nullable columns without defaults
    """
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10, "application_name": "ai-trading-bot:safe-fixes"},
    )
    try:
        with engine.begin() as conn:
            insp = sa_inspect(conn)

            # Create missing non-unique indexes
            for table in expected["tables"]:
                if table not in insp.get_table_names(schema="public"):
                    continue
                actual_indexes = {i.get("name") for i in insp.get_indexes(table, schema="public")}
                for idx in expected.get("indexes", {}).get(table, []):
                    name = idx.get("name")
                    cols = idx.get("columns", [])
                    if not name or not cols:
                        continue
                    if name in actual_indexes:
                        continue
                    cols_sql = ", ".join(cols)
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols_sql})"))

            # Convert JSON to JSONB where expected type is JSONB
            for table in expected["tables"]:
                if table not in insp.get_table_names(schema="public"):
                    continue
                columns = insp.get_columns(table, schema="public")
                actual_types_map = {c["name"]: str(c["type"]).lower() for c in columns}
                for col_name, exp_type in expected["types"][table].items():
                    # Only attempt when actual exists and expected is JSON/JSONB
                    actual_t = actual_types_map.get(col_name, "")
                    try:
                        compiled_exp = exp_type.compile(dialect=engine.dialect)  # type: ignore[attr-defined]
                        exp_str = str(compiled_exp).lower()
                    except Exception:
                        exp_str = str(exp_type).lower()
                    if "jsonb" in exp_str and "json" in actual_t and "jsonb" not in actual_t:
                        conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {col_name} TYPE jsonb USING {col_name}::jsonb"))

            # Add missing nullable columns (without defaults) only
            for table in expected["tables"]:
                if table not in insp.get_table_names(schema="public"):
                    continue
                columns = insp.get_columns(table, schema="public")
                actual_cols = {c["name"] for c in columns}
                for col_name in expected["columns"][table]:
                    if col_name in actual_cols:
                        continue
                    nullable_map = expected.get("nullable", {}).get(table, {})
                    if not nullable_map.get(col_name, True):
                        # Skip non-nullable missing columns (unsafe)
                        continue
                    # Build a minimal type expression for safe add
                    exp_type = expected["types"][table][col_name]
                    try:
                        type_expr = exp_type.compile(dialect=engine.dialect)  # type: ignore[attr-defined]
                        type_sql = str(type_expr)
                    except Exception:
                        type_sql = str(exp_type)
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {type_sql}"))
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def _verify_enum_types(db_url: str) -> list[str]:
    """Verify that critical enum types exist and have required values.
    
    Returns:
        List of issues found (empty list if all checks pass)
    """
    issues = []
    
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10, "application_name": "ai-trading-bot:enum-verify"},
    )
    
    try:
        with engine.connect() as conn:
            # * Critical enum verification - orderstatus
            print("🔍 Checking orderstatus enum...")
            
            # Check if orderstatus enum exists
            result = conn.execute(text("""
                SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder) as labels
                FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                WHERE t.typname = 'orderstatus'
                GROUP BY t.typname
            """))
            
            enum_info = result.fetchall()
            if not enum_info:
                issues.append("orderstatus enum does not exist")
                return issues
            
            enum_name, labels = enum_info[0]
            required_values = ['PENDING', 'OPEN', 'FILLED', 'CANCELLED', 'FAILED']
            missing_values = [val for val in required_values if val not in labels]
            
            if missing_values:
                issues.append(f"orderstatus enum missing required values: {missing_values}")
                print(f"  ❌ Missing values: {missing_values}")
                print(f"  📋 Current values: {list(labels)}")
                print(f"  🔧 Fix: ALTER TYPE orderstatus ADD VALUE 'VALUE_NAME';")
            else:
                print(f"  ✅ All required values present: {list(labels)}")
            
            # Check which enum the positions table actually uses
            result = conn.execute(text("""
                SELECT column_name, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'positions' AND column_name = 'status'
            """))
            
            column_info = result.fetchall()
            if column_info:
                actual_enum = column_info[0][1]
                if actual_enum != 'orderstatus':
                    issues.append(f"positions.status uses '{actual_enum}' instead of 'orderstatus'")
                    print(f"  ❌ positions.status uses wrong enum type: {actual_enum}")
                else:
                    print(f"  ✅ positions.status correctly uses orderstatus enum")
            else:
                issues.append("positions.status column not found")
            
            # Test enum value acceptance
            print("🧪 Testing enum value acceptance...")
            test_values = ['PENDING', 'OPEN', 'FILLED', 'CANCELLED', 'FAILED']
            
            for test_val in test_values:
                try:
                    # Use string formatting for enum type casting (safe with known enum values)
                    result = conn.execute(text(f"SELECT '{test_val}'::orderstatus as test"))
                    converted = result.fetchone()[0]
                    print(f"  ✅ '{test_val}' -> {converted}")
                except Exception as e:
                    error_msg = str(e)
                    if "invalid input value for enum orderstatus" in error_msg:
                        issues.append(f"orderstatus enum rejects '{test_val}' value")
                        print(f"  ❌ '{test_val}' -> REJECTED: {error_msg[:100]}...")
                        if test_val == 'OPEN':
                            print(f"  🎯 CRITICAL: This explains the production error!")
                    else:
                        print(f"  ⚠️  '{test_val}' -> Unexpected error: {error_msg[:100]}...")
            
            # Check for conflicting enum types (use separate connection to avoid transaction issues)
            print("🔍 Checking for conflicting enum types...")
            try:
                with engine.connect() as conn2:
                    result = conn2.execute(text("""
                        SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder) as labels
                        FROM pg_type t
                        JOIN pg_enum e ON t.oid = e.enumtypid
                        WHERE t.typname LIKE '%orderstatus%'
                        GROUP BY t.typname
                        ORDER BY t.typname
                    """))
                    
                    all_enum_types = result.fetchall()
                    if len(all_enum_types) > 1:
                        print(f"  ⚠️  Multiple orderstatus-related enum types found:")
                        for enum_name, enum_labels in all_enum_types:
                            print(f"    - {enum_name}: {list(enum_labels)}")
                            if enum_name != 'orderstatus' and 'OPEN' not in enum_labels:
                                issues.append(f"Conflicting enum '{enum_name}' missing OPEN value")
                    else:
                        print(f"  ✅ Only one orderstatus enum type found")
            except Exception as e:
                print(f"  ⚠️  Could not check for conflicting enums: {e}")
                
    except Exception as e:
        issues.append(f"Enum verification failed: {e}")
        print(f"  ❌ Enum verification error: {e}")
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
    
    return issues


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

        # Check if database schema is actually out of sync before applying migrations
        schema_out_of_sync = False
        if pending:
            print("\n🔍 Checking if schema is actually out of sync...")
            try:
                # Quick check: see if we have missing tables or columns that migrations should create
                expected = _expected_schema_from_models()
                verify_result = _verify_schema(db_url)

                # Check for significant schema differences
                if (verify_result.get("missing_tables") or
                    verify_result.get("missing_columns") or
                    verify_result.get("type_mismatches")):
                    schema_out_of_sync = True
                    print("⚠️  Schema is out of sync with models - migrations needed")
                else:
                    print("✅ Schema appears consistent - migrations may be redundant")
                    print("   (This prevents duplicate column errors from re-running migrations)")

            except Exception as e:
                print(f"⚠️  Could not verify schema state: {e}")
                # If we can't verify, assume migrations are needed for safety
                schema_out_of_sync = True

        if pending and ns.apply_migrations and (schema_out_of_sync or ns.force_migrations):
            if ns.force_migrations and not schema_out_of_sync:
                print("⚠️  Forcing migrations despite schema appearing up-to-date")
            print("\nApplying Migrations")
            print("-------------------")
            ok = _apply_migrations(cfg)
            if not ok:
                return 1
        elif pending and ns.apply_migrations and not schema_out_of_sync and not ns.force_migrations:
            print("ℹ️  Skipping migrations - schema appears up to date")
            print("   (Use --force-migrations to override this check)")
        elif pending and not ns.apply_migrations:
            auto_apply = _safe_bool_env("ATB_AUTO_APPLY_MIGRATIONS", False)
            if auto_apply:
                print("ℹ️  Auto-applying migrations (ATB_AUTO_APPLY_MIGRATIONS=true)")
                if schema_out_of_sync:
                    ok = _apply_migrations(cfg)
                    if not ok:
                        return 1
                else:
                    print("ℹ️  Skipping migrations - schema appears up to date")
            else:
                print(
                    "ℹ️  Pending migrations detected. To apply automatically, pass --apply-migrations or set ATB_AUTO_APPLY_MIGRATIONS=true."
                )

        print("\nSchema Verification Against Models")
        print("----------------------------------")
        expected = _expected_schema_from_models()
        verify = _verify_schema(db_url)
        if verify.get("ok"):
            print("✅ Schema matches SQLAlchemy models.")
            
            # Add enum verification for successful schema checks too
            print("\nEnum Type Verification")
            print("----------------------")
            enum_issues = _verify_enum_types(db_url)
            if enum_issues:
                print("❌ Enum type issues detected:")
                for issue in enum_issues:
                    print(f"  • {issue}")
                return 1
            else:
                print("✅ Enum types verified successfully.")
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
            # Optionally apply safe fixes
            if ns.apply_fixes:
                print("\nApplying Safe Fixes")
                print("-------------------")
                try:
                    _apply_safe_fixes(db_url, verify, expected)
                    print("✅ Safe fixes applied.")
                    # Re-verify after fixes
                    print("\nRe-checking schema after fixes...")
                    verify2 = _verify_schema(db_url)
                    if verify2.get("ok"):
                        print("✅ Schema matches SQLAlchemy models after fixes.")
                        return 0
                    else:
                        print("⚠️  Remaining deviations after safe fixes. Review output above.")
                        return 1
                except Exception as e:
                    print(f"❌ Failed to apply safe fixes: {e}")
                    traceback.print_exc()
                    return 1
            else:
                # Add enum verification before returning failure
                print("\nEnum Type Verification")
                print("----------------------")
                enum_issues = _verify_enum_types(db_url)
                if enum_issues:
                    print("❌ Enum type issues detected:")
                    for issue in enum_issues:
                        print(f"  • {issue}")
                    return 1
                else:
                    print("✅ Enum types verified successfully.")
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

    # Import railway commands
    from cli.commands import railway

    p_verify = sub.add_parser("verify", help="Verify database integrity, migrations, and schema sync status")
    p_verify.add_argument(
        "--apply-migrations",
        action="store_true",
        help="Apply Alembic migrations to head if pending and schema is out of sync",
    )
    p_verify.add_argument(
        "--apply-fixes",
        action="store_true",
        help="Apply safe schema fixes (non-unique indexes, JSON→JSONB, nullable columns)",
    )
    p_verify.add_argument(
        "--force-migrations",
        action="store_true",
        help="Force apply migrations even if schema appears up-to-date (bypasses duplicate column prevention)",
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

    # Register railway subcommands
    railway.register(sub)
