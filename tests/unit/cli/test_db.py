"""Tests for atb db commands."""

import argparse
from unittest.mock import Mock, patch

import pytest

from cli.commands.db import _verify, _migrate, _backup


class TestDbVerify:
    """Tests for the db verify command."""

    def test_verifies_database_successfully(self):
        """Test that database verification succeeds."""
        # Arrange
        args = argparse.Namespace(
            db_cmd="verify",
            apply_migrations=False,
            apply_fixes=False,
            force_migrations=False,
            env=None,
        )

        # Act
        with (
            patch("cli.commands.db._get_database_url_for_env") as mock_get_db_url,
            patch("cli.commands.db._basic_integrity_checks") as mock_integrity,
            patch("cli.commands.db._alembic_config") as mock_alembic_config,
            patch("cli.commands.db._get_alembic_status") as mock_get_status,
            patch("cli.commands.db._expected_schema_from_models") as mock_expected,
            patch("cli.commands.db._verify_schema") as mock_verify_schema,
        ):

            mock_get_db_url.return_value = "postgresql://user:pass@localhost/db"
            mock_integrity.return_value = {
                "connectivity": True,
                "tables_exist": {"trades": True},
                "row_counts": {"trades": 0},
            }
            mock_alembic_config.return_value = Mock()
            mock_get_status.return_value = {
                "state": "ok",
                "current": "abc123",
                "pending": [],
                "heads": ["abc123"],
            }
            mock_expected.return_value = {}
            mock_verify_schema.return_value = {"ok": True}

            result = _verify(args)

            # Assert
            assert result == 0

    def test_returns_error_when_database_url_missing(self):
        """Test that error is returned when DATABASE_URL is missing."""
        # Arrange
        args = argparse.Namespace(
            db_cmd="verify",
            apply_migrations=False,
            apply_fixes=False,
            force_migrations=False,
            env=None,
        )

        # Act
        with patch("cli.commands.db._get_database_url_for_env") as mock_get_db_url:
            mock_get_db_url.side_effect = RuntimeError("DATABASE_URL is required")

            result = _verify(args)

            # Assert
            assert result == 1


class TestDbMigrate:
    """Tests for the db migrate command."""

    def test_runs_migrations_successfully(self):
        """Test that migrations run successfully."""
        # Arrange
        args = argparse.Namespace(
            db_cmd="migrate",
            check=False,
            env=None,
        )

        # Act
        with (
            patch("cli.commands.db._get_database_url_for_env") as mock_get_db_url,
            patch("cli.commands.db.subprocess.run") as mock_subprocess,
        ):

            mock_get_db_url.return_value = "postgresql://user:pass@localhost/db"
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Running upgrade"
            mock_subprocess.return_value = mock_result

            result = _migrate(args)

            # Assert
            assert result == 0
            mock_subprocess.assert_called_once()

    def test_checks_migration_status_only(self):
        """Test that check flag only checks status."""
        # Arrange
        args = argparse.Namespace(
            db_cmd="migrate",
            check=True,
            env=None,
        )

        # Act
        with (
            patch("cli.commands.db._get_database_url_for_env") as mock_get_db_url,
            patch("cli.commands.db.subprocess.run") as mock_subprocess,
        ):

            mock_get_db_url.return_value = "postgresql://user:pass@localhost/db"
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "abc123 (head)"
            mock_subprocess.return_value = mock_result

            result = _migrate(args)

            # Assert
            assert result == 0
            # Should call subprocess with "alembic current"
            call_args = mock_subprocess.call_args[0][0]
            assert "alembic" in call_args
            assert "current" in call_args


class TestDbBackup:
    """Tests for the db backup command."""

    def test_creates_backup_successfully(self):
        """Test that backup is created successfully."""
        # Arrange
        args = argparse.Namespace(
            db_cmd="backup",
            backup_dir="./backups",
            retention=7,
            env=None,
        )

        # Act
        with (
            patch("cli.commands.db._get_database_url_for_env") as mock_get_db_url,
            patch("cli.commands.db.subprocess.run") as mock_run,
            patch("cli.commands.db.Path") as mock_path,
        ):

            mock_get_db_url.return_value = "postgresql://user:pass@localhost/db"
            mock_run.return_value = Mock(returncode=0)
            mock_path.return_value.mkdir.return_value = None

            result = _backup(args)

            # Assert
            assert result == 0
            mock_run.assert_called()

    def test_returns_error_when_backup_fails(self):
        """Test that error is returned when backup fails."""
        # Arrange
        import subprocess

        args = argparse.Namespace(
            db_cmd="backup",
            backup_dir="./backups",
            retention=7,
            env=None,
        )

        # Act
        with (
            patch("cli.commands.db._get_database_url_for_env") as mock_get_db_url,
            patch("cli.commands.db.subprocess.run") as mock_run,
            patch("cli.commands.db.Path") as mock_path,
        ):

            mock_get_db_url.return_value = "postgresql://user:pass@localhost/db"
            # subprocess.run is called with check=True, so it raises CalledProcessError on failure
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["pg_dump"], stderr=b"pg_dump failed"
            )
            mock_path.return_value.mkdir.return_value = None

            result = _backup(args)

            # Assert
            assert result == 1
