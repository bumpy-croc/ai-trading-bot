from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import UTC
from pathlib import Path

# Ensure project root and src are in sys.path for absolute imports
from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))


def _print_header(title: str) -> None:
    print("\n" + title)
    print("=" * max(8, len(title)))


def _authenticate_railway() -> str | None:
    """Authenticate with Railway CLI and return the project ID."""
    try:
        # Check if already authenticated
        result = subprocess.run(["railway", "whoami"], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print("🔐 Railway CLI not authenticated. Please run 'railway login' first.")
            return None

        # Get project ID from environment
        project_id = os.getenv("RAILWAY_PROJECT_ID")
        if not project_id:
            print("❌ RAILWAY_PROJECT_ID environment variable not set.")
            return None

        # Verify project exists and is accessible
        result = subprocess.run(
            ["railway", "list", "--json"], capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            print("❌ Failed to list Railway projects.")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            print("   Make sure you're logged in with 'railway login' and have access to projects.")
            return None

        # Parse JSON output to find project
        try:
            import json

            projects = json.loads(result.stdout)
            project_found = False
            available_projects = []

            for project in projects:
                available_projects.append(f"{project['name']} (ID: {project['id']})")
                if project["id"] == project_id:
                    project_found = True
                    break

            if not project_found:
                print(f"❌ Project ID {project_id} not found in your Railway projects.")
                print("   Available projects:")
                for proj in available_projects:
                    print(f"     - {proj}")
                return None

        except json.JSONDecodeError:
            print("❌ Failed to parse Railway project list.")
            print(
                "   Raw output:",
                result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout,
            )
            return None

        print(f"✅ Authenticated with Railway project: {project_id}")
        return project_id

    except subprocess.TimeoutExpired:
        print("❌ Railway CLI authentication timed out.")
        return None
    except FileNotFoundError:
        print(
            "❌ Railway CLI not installed. Please install it first: https://docs.railway.app/develop/cli"
        )
        return None
    except Exception as e:
        print(f"❌ Railway authentication failed: {e}")
        return None


def _get_database_url(env: str) -> str | None:
    """Get the database URL for the specified environment."""
    env_var = f"RAILWAY_{env.upper()}_DATABASE_URL"
    db_url = os.getenv(env_var)

    if not db_url:
        print(f"❌ Environment variable {env_var} not set.")
        return None

    if not db_url.startswith("postgresql://"):
        print(f"❌ {env_var} does not appear to be a valid PostgreSQL URL.")
        return None

    return db_url


def _confirm_dangerous_action(action: str, env: str) -> bool:
    """Require double confirmation for dangerous actions."""
    print(f"\n🚨 DANGER: You are about to {action} the {env.upper()} database!")
    print("This action cannot be undone and may result in permanent data loss.")

    # First confirmation
    confirm1 = input(f"\nType 'I UNDERSTAND' to proceed with {action}: ").strip()
    if confirm1 != "I UNDERSTAND":
        print("❌ First confirmation failed. Aborting.")
        return False

    # Second confirmation
    confirm2 = input(f"Type '{env.upper()}' to confirm this is the correct environment: ").strip()
    if confirm2 != env.upper():
        print("❌ Second confirmation failed. Aborting.")
        return False

    return True


def _reset_database(env: str) -> int:
    """Reset the Railway database for the specified environment."""
    print(f"🔄 Resetting {env.upper()} database...")

    # Get database URL
    db_url = _get_database_url(env)
    if not db_url:
        return 1

    try:
        # Connect to database and reset all tables
        import psycopg2
        from psycopg2 import sql

        conn = psycopg2.connect(db_url)

        with conn.cursor() as cur:
            # Get all tables in public schema
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = cur.fetchall()

            if not tables:
                print("ℹ️  No tables found in database.")
                return 0

            table_names = [sql.Identifier(t[0]) for t in tables]
            print(f"📋 Found {len(table_names)} tables to reset: {[str(t) for t in table_names]}")

            # Disable referential integrity temporarily
            cur.execute("SET session_replication_role = 'replica';")

            # Truncate all tables with cascade to handle foreign keys
            for table in table_names:
                cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(table))
                print(f"✅ Reset table: {table}")

            # Re-enable referential integrity
            cur.execute("SET session_replication_role = 'origin';")

        conn.commit()
        print(f"✅ {env.upper()} database has been successfully reset!")
        return 0

    except ImportError:
        print("❌ psycopg2 not installed. Please install it: pip install psycopg2-binary")
        return 1
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return 1
    finally:
        try:
            if "conn" in locals():
                conn.close()
        except Exception:
            pass


def _backup_database(env: str) -> int:
    """Backup the Railway database for the specified environment."""
    print(f"📦 Backing up {env.upper()} database...")

    # Get database URL
    db_url = _get_database_url(env)
    if not db_url:
        return 1

    try:
        import datetime as dt
        from urllib.parse import urlparse

        parsed = urlparse(db_url)
        password = parsed.password or ""

        # Create temp directory for backup
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup_filename = f"railway_{env}_backup_{timestamp}.dump"
            backup_path = Path(temp_dir) / backup_filename

            print(f"📁 Creating backup in temporary directory: {backup_path}")

            # Set up environment for pg_dump
            env_vars = os.environ.copy()
            env_vars["PGPASSWORD"] = password

            # Run pg_dump
            cmd = [
                "pg_dump",
                f"--dbname={db_url}",
                "-Fc",  # Custom format
                "-Z",
                "9",  # Maximum compression
                "-f",
                str(backup_path),
            ]

            result = subprocess.run(
                cmd, env=env_vars, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ pg_dump failed: {result.stderr}")
                return 1

            # Check file size
            # Note: File size display removed due to unused variable
            print(f"📋 Backup saved to: {backup_path}")

            return 0

    except subprocess.TimeoutExpired:
        print("❌ Database backup timed out.")
        return 1
    except FileNotFoundError:
        print("❌ pg_dump not found. Please ensure PostgreSQL client tools are installed.")
        return 1
    except Exception as e:
        print(f"❌ Database backup failed: {e}")
        return 1


def _railway_reset(ns: argparse.Namespace) -> int:
    """Handle railway reset command."""
    env = ns.env

    _print_header(f"Railway Database Reset - {env.upper()}")

    # Authenticate with Railway
    project_id = _authenticate_railway()
    if not project_id:
        return 1

    # Double confirmation for dangerous action
    if not _confirm_dangerous_action("reset", env):
        return 1

    # Reset the database
    return _reset_database(env)


def _railway_backup(ns: argparse.Namespace) -> int:
    """Handle railway backup command."""
    env = ns.env

    _print_header(f"Railway Database Backup - {env.upper()}")

    # Authenticate with Railway
    project_id = _authenticate_railway()
    if not project_id:
        return 1

    # Note: Backup is not as dangerous as reset, but still require confirmation
    print(f"\n📦 You are about to backup the {env.upper()} database.")
    confirm = input("Type 'BACKUP' to proceed: ").strip()
    if confirm != "BACKUP":
        print("❌ Confirmation failed. Aborting.")
        return 1

    # Backup the database
    return _backup_database(env)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register railway commands under db subcommand."""
    p = subparsers.add_parser("railway", help="Railway database management commands")
    sub = p.add_subparsers(dest="railway_cmd", required=True)

    # Reset command
    p_reset = sub.add_parser("reset", help="Reset Railway database to latest schema")
    p_reset.add_argument(
        "--env",
        required=True,
        choices=["development", "staging", "production"],
        help="Target Railway environment",
    )
    p_reset.set_defaults(func=_railway_reset)

    # Backup command
    p_backup = sub.add_parser("backup", help="Download Railway database backup locally")
    p_backup.add_argument(
        "--env",
        required=True,
        choices=["development", "staging", "production"],
        help="Target Railway environment",
    )
    p_backup.set_defaults(func=_railway_backup)
