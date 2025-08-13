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
Usage: python reset_railway_database.py [staging|production]
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


def truncate_all_tables(conn):
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
    if len(sys.argv) != 2 or sys.argv[1] not in ("staging", "production"):
        print(USAGE)
        sys.exit(1)
    env = sys.argv[1]
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
        truncate_all_tables(conn)
        print(f"All tables in {env} database have been reset (structure preserved).")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
