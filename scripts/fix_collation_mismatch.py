#!/usr/bin/env python3
"""
PostgreSQL Collation Version Mismatch Fix Script

This script provides a quick way to fix PostgreSQL collation version mismatches
that commonly occur in Railway deployments.

Usage:
    python scripts/fix_collation_mismatch.py --check
    python scripts/fix_collation_mismatch.py --fix
"""

import argparse
import os
import sys

import psycopg2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def get_database_url(env=None):
    """Get database URL from environment variables."""
    # If env is specified, use Railway-specific environment variables
    if env:
        env_var = f"RAILWAY_{env.upper()}_DATABASE_URL"
        db_url = os.getenv(env_var)

        if not db_url:
            print(f"❌ {env_var} environment variable not found")
            print(f"   Please set {env_var} to your {env} PostgreSQL connection string")
            print(f"   Example: {env_var}=postgresql://user:pass@host:port/database")
            sys.exit(1)

        if not db_url.startswith("postgresql://"):
            print(f"❌ {env_var} must start with 'postgresql://'")
            sys.exit(1)

        return db_url

    # Fallback to generic DATABASE_URL
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("❌ DATABASE_URL environment variable not found")
        print("   Please set DATABASE_URL to your PostgreSQL connection string")
        print("   Or use --env parameter to specify Railway environment:")
        print("   --env development (uses RAILWAY_DEVELOPMENT_DATABASE_URL)")
        print("   --env staging (uses RAILWAY_STAGING_DATABASE_URL)")
        print("   --env production (uses RAILWAY_PRODUCTION_DATABASE_URL)")
        sys.exit(1)

    if not db_url.startswith("postgresql://"):
        print("❌ DATABASE_URL must start with 'postgresql://'")
        sys.exit(1)

    return db_url


def check_collation_status(env=None):
    """Check the current collation status."""
    print("🔍 Checking PostgreSQL Collation Status")
    print("=" * 50)

    try:
        db_url = get_database_url(env)
        conn = psycopg2.connect(db_url)

        with conn.cursor() as cur:
            # Get database info
            cur.execute("""
                SELECT datname, datcollate, datctype
                FROM pg_database
                WHERE datname = current_database();
            """)
            db_info = cur.fetchone()

            print(f"📊 Database: {db_info[0]}")
            print(f"📊 LC_COLLATE: {db_info[1]}")
            print(f"📊 LC_CTYPE: {db_info[2]}")

            # Get PostgreSQL version
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"📊 PostgreSQL Version: {version.split(' ')[1]}")

            # Check collation version
            try:
                cur.execute("""
                    SELECT pg_collation_actual_version(oid) as actual_version
                    FROM pg_collation
                    WHERE collname = 'default'
                    LIMIT 1;
                """)
                actual_version = cur.fetchone()
                if actual_version and actual_version[0]:
                    print(f"📊 Collation Version: {actual_version[0]}")
                    print("✅ Collation version available")
                else:
                    print("⚠️  Collation version information unavailable")
            except Exception as e:
                print(f"⚠️  Could not get collation version: {e}")

        conn.close()

        print("\n💡 If you see collation warnings in your logs, run:")
        print("   python scripts/fix_collation_mismatch.py --fix")

    except psycopg2.Error as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


def fix_collation_mismatch(env=None):
    """Fix collation version mismatch."""
    print("🔧 Fixing PostgreSQL Collation Version Mismatch")
    print("=" * 55)

    try:
        db_url = get_database_url(env)
        conn = psycopg2.connect(db_url)
        conn.autocommit = True

        with conn.cursor() as cur:
            # Method 1: Try ALTER DATABASE REFRESH COLLATION VERSION (PostgreSQL 15+)
            print("🔄 Attempting ALTER DATABASE REFRESH COLLATION VERSION...")
            try:
                cur.execute("ALTER DATABASE current_database() REFRESH COLLATION VERSION;")
                print("✅ Successfully refreshed collation version!")
                conn.close()
                print("\n🎉 Collation version mismatch fixed!")
                return
            except psycopg2.Error as e:
                if "REFRESH COLLATION VERSION" in str(e):
                    print("⚠️  REFRESH COLLATION VERSION not supported (PostgreSQL < 15)")
                    print("   Falling back to manual object rebuild...")
                else:
                    print(f"❌ ALTER DATABASE failed: {e}")
                    conn.close()
                    sys.exit(1)

            # Method 2: Manual rebuild for older PostgreSQL versions
            print("\n🔄 Rebuilding database objects with explicit collation...")
            print("⚠️  This may take some time for large databases...")

            # Get all tables in public schema
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)

            tables = cur.fetchall()
            print(f"📋 Found {len(tables)} tables to process")

            tables_fixed = 0
            for (table_name,) in tables:
                print(f"🔧 Processing table: {table_name}")

                try:
                    # Get text/varchar columns for this table
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = %s
                        AND data_type IN ('text', 'varchar', 'character varying')
                        ORDER BY ordinal_position;
                    """, (table_name,))

                    text_columns = cur.fetchall()

                    if not text_columns:
                        print(f"   ℹ️  No text columns found in {table_name}")
                        continue

                    print(f"   📝 Found {len(text_columns)} text columns")

                    # Alter each text column to use explicit collation
                    for col_name, data_type, nullable, default in text_columns:
                        if data_type == 'text':
                            alter_sql = f'ALTER TABLE {table_name} ALTER COLUMN "{col_name}" TYPE TEXT COLLATE "en_US.UTF-8"'
                        else:
                            # For varchar/character varying, preserve length
                            cur.execute("""
                                SELECT character_maximum_length
                                FROM information_schema.columns
                                WHERE table_schema = 'public'
                                AND table_name = %s
                                AND column_name = %s;
                            """, (table_name, col_name))
                            max_len = cur.fetchone()[0]

                            if max_len:
                                alter_sql = f'ALTER TABLE {table_name} ALTER COLUMN "{col_name}" TYPE VARCHAR({max_len}) COLLATE "en_US.UTF-8"'
                            else:
                                alter_sql = f'ALTER TABLE {table_name} ALTER COLUMN "{col_name}" TYPE TEXT COLLATE "en_US.UTF-8"'

                        cur.execute(alter_sql)
                        print(f"   ✅ Updated column: {col_name}")

                    tables_fixed += 1
                    print(f"✅ Successfully processed table: {table_name}")

                except Exception as table_error:
                    print(f"   ❌ Failed to process table {table_name}: {table_error}")
                    continue

            if tables_fixed > 0:
                print(f"\n✅ Successfully fixed collation for {tables_fixed} tables")
            else:
                print("\nℹ️  No tables required collation fixes")

        conn.close()

        print("\n🎉 Collation version mismatch fix completed!")
        print("\n📝 Next Steps:")
        print("   1. Monitor your application logs for collation warnings")
        print("   2. Consider using explicit collations in future schema changes")
        print("   3. See docs/COLLATION_VERSION_MISMATCH_GUIDE.md for prevention strategies")

    except psycopg2.Error as e:
        print(f"❌ Database operation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fix PostgreSQL collation version mismatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using generic DATABASE_URL
  python scripts/fix_collation_mismatch.py --check
  python scripts/fix_collation_mismatch.py --fix

  # Using Railway environment-specific URLs
  python scripts/fix_collation_mismatch.py --env development --check
  python scripts/fix_collation_mismatch.py --env staging --fix
  python scripts/fix_collation_mismatch.py --env production --fix

Environment Variables:
  DATABASE_URL - Generic PostgreSQL connection string
  RAILWAY_DEVELOPMENT_DATABASE_URL - Development environment database URL
  RAILWAY_STAGING_DATABASE_URL - Staging environment database URL
  RAILWAY_PRODUCTION_DATABASE_URL - Production environment database URL
        """
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current collation status"
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix collation version mismatch"
    )

    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        help="Railway environment (uses environment-specific DATABASE_URL)"
    )

    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.print_help()
        sys.exit(1)

    # Show which environment/database URL we're using
    if args.env:
        env_var = f"RAILWAY_{args.env.upper()}_DATABASE_URL"
        print(f"🔧 Using Railway environment: {args.env}")
        print(f"🔧 Database URL from: {env_var}")
    else:
        print("🔧 Using generic DATABASE_URL")
    print()

    if args.check:
        check_collation_status(args.env)
    elif args.fix:
        # Safety confirmation for destructive operation
        if args.env:
            env_name = args.env.upper()
            print(f"⚠️  WARNING: You are about to modify the {env_name} database!")
        else:
            print("⚠️  WARNING: You are about to modify the database!")

        print("   This operation will update collation settings for text columns.")
        print("   Although safe, it's recommended to backup your database first.")
        print()

        confirm = input("Do you want to proceed? (type 'yes' to continue): ").strip().lower()
        if confirm != 'yes':
            print("❌ Operation cancelled.")
            sys.exit(0)

        fix_collation_mismatch(args.env)


if __name__ == "__main__":
    main()
