"""Integration tests for atb db commands."""

import argparse
from unittest.mock import Mock, patch

import pytest

from cli.commands.db import _verify


@pytest.mark.integration
class TestDbIntegration:
    """Integration tests for database commands."""

    def test_verify_connects_to_test_database(self):
        """Test that verify command connects to test database."""
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
            patch("cli.commands.db._get_database_url_for_env") as mock_get_url,
            patch("cli.commands.db._basic_integrity_checks") as mock_integrity,
            patch("cli.commands.db._alembic_config") as mock_alembic_config,
            patch("cli.commands.db._get_alembic_status") as mock_get_status,
            patch("cli.commands.db._expected_schema_from_models") as mock_expected_schema,
            patch("cli.commands.db._verify_schema") as mock_verify_schema,
        ):

            mock_get_url.return_value = "postgresql://user:pass@localhost/test_db"
            mock_integrity.return_value = {
                "connectivity": True,
                "tables_exist": {"trading_sessions": True, "trades": True},
                "row_counts": {"trading_sessions": 0, "trades": 0},
                "alembic_version_present": True,
            }
            mock_alembic_config.return_value = Mock()
            mock_get_status.return_value = {
                "state": "ok",
                "current": "abc123",
                "pending": [],
                "heads": ["abc123"],
            }
            mock_expected_schema.return_value = {
                "tables": ["trading_sessions", "trades"],
                "columns": {},
                "primary_keys": {},
                "types": {},
                "nullable": {},
                "indexes": {},
            }
            mock_verify_schema.return_value = {"ok": True}

            result = _verify(args)

            # Assert
            assert result == 0
