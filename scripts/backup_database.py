#!/usr/bin/env python3
"""Automated encrypted backup of the PostgreSQL database to S3.

This script is intended to be run by a scheduler/cron (e.g. Railway Scheduled Jobs)
Every execution performs the following steps:
1. Parse `DATABASE_URL` from environment/config (same resolver as the app).
2. Generate a compressed custom-format pg_dump archive.
3. Upload the dump to S3 under `s3://<BUCKET>/<db_name>/YYYY/mm/dd/backup-<timestamp>.dump`.
   • Server-side encryption (SSE-S3) is enabled by default.  Use `--kms-key` to switch to SSE-KMS.
4. Optionally prune backups older than the configured retention period.

Environment Variables (or CLI flags override):
• DATABASE_URL             – PostgreSQL URL
• AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
• S3_BACKUP_BUCKET         – Destination bucket name
• BACKUP_RETENTION_DAYS    – How long to keep backups (default 7)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PostgreSQL → S3 backup")
    parser.add_argument("--bucket", default=os.getenv("S3_BACKUP_BUCKET"), help="S3 bucket name")
    parser.add_argument("--retention", type=int, default=int(os.getenv("BACKUP_RETENTION_DAYS", 7)), help="Retention in days")
    parser.add_argument("--kms-key", help="KMS key ID for SSE-KMS (optional)")
    return parser.parse_args()


def _get_db_params(db_url: str):
    """Return (dbname, user, host, port, password)."""
    parsed = urlparse(db_url)
    return (
        parsed.path.lstrip("/"),  # dbname
        parsed.username,
        parsed.hostname,
        str(parsed.port or 5432),
        parsed.password,
    )


# ---------------------------------------------------------------------------
# Main backup routine
# ---------------------------------------------------------------------------


def perform_backup(bucket: str, retention_days: int, kms_key: str | None = None) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    dbname, user, host, port, password = _get_db_params(db_url)

    timestamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    s3_key_prefix = f"backups/{dbname}/{_dt.datetime.utcnow().strftime('%Y/%m/%d')}"
    dump_filename = f"backup-{timestamp}.dump"

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / dump_filename
        print(f"📦 Creating dump: {dump_path}")

        env = os.environ.copy()
        env["PGPASSWORD"] = password or ""  # pg_dump reads password from env

        # Run pg_dump in custom format, highest compression
        cmd = [
            "pg_dump",
            f"--dbname={db_url}",
            "-Fc",  # custom format (compressed, safe for restore)
            "-Z",
            "9",  # maximum compression
            "-f",
            str(dump_path),
        ]
        try:
            subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            print(f"❌ pg_dump failed: {exc.stderr.decode()}", file=sys.stderr)
            sys.exit(1)

        # Upload to S3
        s3 = boto3.client("s3")
        s3_key = f"{s3_key_prefix}/{dump_filename}"
        extra_args: dict[str, str] = {"ServerSideEncryption": "aws:kms" if kms_key else "AES256"}
        if kms_key:
            extra_args["SSEKMSKeyId"] = kms_key

        print(f"☁️  Uploading {dump_path} → s3://{bucket}/{s3_key}")
        try:
            s3.upload_file(str(dump_path), bucket, s3_key, ExtraArgs=extra_args)
        except ClientError as e:
            print(f"❌ Failed to upload to S3: {e}", file=sys.stderr)
            sys.exit(1)

        print("✅ Backup uploaded successfully")

    # Retention policy – delete objects older than retention_days
    if retention_days > 0:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=retention_days)
        print(f"🧹 Deleting backups older than {retention_days} days (before {cutoff.date()})")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"backups/{dbname}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                last_mod = obj["LastModified"]
                if last_mod.replace(tzinfo=None) < cutoff:
                    print(f"   • Removing {key}")
                    s3.delete_object(Bucket=bucket, Key=key)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args_ns = _parse_args()
    if not args_ns.bucket:
        print("❌ S3 bucket must be provided via --bucket or S3_BACKUP_BUCKET", file=sys.stderr)
        sys.exit(1)

    perform_backup(args_ns.bucket, args_ns.retention, args_ns.kms_key)