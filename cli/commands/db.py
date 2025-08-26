from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root and src are in sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))


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


def _verify(_ns: argparse.Namespace) -> int:
    from src.config.config_manager import get_config
    from src.database.manager import DatabaseManager

    print("🔍 Verifying PostgreSQL Database Connection")
    print("=" * 50)
    try:
        db_manager = DatabaseManager()
        db_info = db_manager.get_database_info()
        print("📊 Database Information:")
        print(f"  - Database URL: {db_info['database_url']}")
        print(f"  - Database Type: {db_info['database_type']}")
        print(f"  - Connection Pool Size: {db_info['connection_pool_size']}")

        print("\n🔗 Testing Connection...")
        if db_manager.test_connection():
            print("✅ PostgreSQL database connection successful!")
        else:
            print("❌ PostgreSQL database connection failed!")
            return 1

        print("\n🎯 Testing Session Creation...")
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="PAPER",
            initial_balance=10000.0,
            session_name="verification_test",
        )
        print(f"✅ Created test session #{session_id}")

        print("\n📝 Testing Event Logging...")
        event_id = db_manager.log_event(
            event_type="TEST",
            message="PostgreSQL database verification test",
            severity="info",
            session_id=session_id,
        )
        print(f"✅ Logged test event #{event_id}")

        db_manager.end_trading_session(session_id, final_balance=10000.0)
        print(f"✅ Ended test session #{session_id}")

        print("\n⚙️  Configuration:")
        config = get_config()
        print(f"  - DATABASE_URL: {'Set' if config.get('DATABASE_URL') else 'Not set'}")
        print(f"  - Railway Environment: {'Yes' if config.get('RAILWAY_PROJECT_ID') else 'No'}")

        stats = db_manager.get_connection_stats()
        print("\n📊 Connection Pool Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        print("\n✅ PostgreSQL database verification completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        print(
            "\nTroubleshooting:\n  1. Ensure DATABASE_URL environment variable is set\n  2. Verify PostgreSQL database is running and accessible\n  3. Check that DATABASE_URL starts with 'postgresql://'\n  4. For Railway: ensure PostgreSQL service is deployed"
        )
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

    p_verify = sub.add_parser("verify", help="Verify database connection")
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
