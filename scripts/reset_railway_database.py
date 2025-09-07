import os
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql

# Load environment variables from .env using python-dotenv
try:
    from dotenv import load_dotenv
except ImportError:
    print("Please install python-dotenv: pip install python-dotenv")
    sys.exit(1)

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

USAGE = """
Usage: python reset_railway_database.py [staging|production] [--complete-reset] [--aggressive]

Options:
  --complete-reset    Drop all types, tables, and reset schema completely (recommended)
                     Without this flag, only truncates data (preserves structure)
  --aggressive        Use aggressive enum dropping (for stubborn Railway deployments)
"""


def get_db_url(env):
    if env == "staging":
        return os.environ.get("RAILWAY_STAGING_DATABASE_URL")
    elif env == "production":
        return os.environ.get("RAILWAY_PRODUCTION_DATABASE_URL")
    else:
        return None


def confirm_production():
    confirm = input(
        "Are you sure you want to reset the PRODUCTION database? Type 'YES' to continue: "
    )
    return confirm == "YES"


def drop_all_types(conn, aggressive=False):
    """Drop all custom types/enums"""
    with conn.cursor() as cur:
        print("🔄 Dropping all custom types...")
        # Drop types in reverse dependency order
        types_to_drop = [
            'partialoperationtype',
            'eventtype',
            'tradesource',
            'orderstatus',
            'ordertype',
            'positionstatus',
            'positionside'
        ]

        if aggressive:
            print("  🚨 Using aggressive enum dropping...")
            # First try to drop any existing enums with PL/pgSQL
            try:
                cur.execute("""
                    DO $$
                    DECLARE
                        enum_name TEXT;
                    BEGIN
                        FOR enum_name IN
                            SELECT typname
                            FROM pg_type
                            WHERE typtype = 'e'
                            AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                        LOOP
                            EXECUTE 'DROP TYPE IF EXISTS ' || enum_name || ' CASCADE';
                            RAISE NOTICE 'Dropped enum: %', enum_name;
                        END LOOP;
                    END
                    $$;
                """)
                print("  ✓ Aggressive PL/pgSQL cleanup completed")
            except Exception as e:
                print(f"  ⚠️  PL/pgSQL cleanup failed: {e}")

        # Standard drop attempts
        for type_name in types_to_drop:
            try:
                cur.execute(sql.SQL("DROP TYPE IF EXISTS {} CASCADE").format(sql.Identifier(type_name)))
                print(f"  ✓ Dropped type: {type_name}")
            except Exception as e:
                print(f"  ⚠️  Could not drop type {type_name}: {e}")
    conn.commit()


def drop_all_tables(conn):
    """Drop all tables and reset schema"""
    with conn.cursor() as cur:
        print("🔄 Dropping all tables...")

        # Get all tables
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
        tables = cur.fetchall()

        if tables:
            table_names = [sql.Identifier(t[0]) for t in tables]
            # Drop tables in reverse dependency order (with CASCADE to handle FKs)
            for t in table_names:
                try:
                    cur.execute(sql.SQL("DROP TABLE {} CASCADE").format(t))
                    print(f"  ✓ Dropped table: {t}")
                except Exception as e:
                    print(f"  ⚠️  Could not drop table {t}: {e}")
        else:
            print("  No tables found.")

        # Reset alembic version table
        try:
            cur.execute("DROP TABLE IF EXISTS alembic_version CASCADE")
            print("  ✓ Dropped alembic_version table")
        except Exception as e:
            print(f"  ⚠️  Could not drop alembic_version: {e}")

    conn.commit()


def truncate_all_tables(conn):
    """Legacy truncate function - use drop_all_tables for complete reset"""
    print("⚠️  Using legacy truncate function. Consider using --complete-reset for full cleanup.")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tablename FROM pg_tables WHERE schemaname = 'public';
        """
        )
        tables = cur.fetchall()
        if not tables:
            print("No tables found.")
            return
        table_names = [sql.Identifier(t[0]) for t in tables]
        # Disable triggers (for foreign key constraints)
        cur.execute("SET session_replication_role = 'replica';")
        for t in table_names:
            cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(t))
        cur.execute("SET session_replication_role = 'origin';")
    conn.commit()


def main():
    # Parse command line arguments
    if len(sys.argv) < 2 or sys.argv[1] not in ("staging", "production"):
        print(USAGE)
        sys.exit(1)

    env = sys.argv[1]
    complete_reset = "--complete-reset" in sys.argv
    aggressive = "--aggressive" in sys.argv

    db_url = get_db_url(env)
    if not db_url:
        print(
            f"Database URL for {env} not set. Please set the {'RAILWAY_STAGING_DATABASE_URL' if env == 'staging' else 'RAILWAY_PRODUCTION_DATABASE_URL'} environment variable."
        )
        sys.exit(1)

    if env == "production" and not confirm_production():
        print("Aborted.")
        sys.exit(0)

    try:
        conn = psycopg2.connect(db_url)

        if complete_reset:
            mode = "AGGRESSIVE " if aggressive else ""
            print(f"🚨 Performing {mode}COMPLETE database reset (drops all schema objects)...")
            drop_all_types(conn, aggressive=aggressive)
            drop_all_tables(conn)
            print(f"✅ {mode}Complete schema reset successful for {env} database.")
            print("💡 You can now run fresh migrations.")
        else:
            print("⚠️  Performing PARTIAL reset (preserves schema, truncates data)...")
            truncate_all_tables(conn)
            print(f"✅ Data truncated for {env} database (structure preserved).")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
