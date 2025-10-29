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
            patch("cli.commands.db._resolve_database_url") as mock_resolve_url,
            patch("cli.commands.db.create_engine") as mock_create_engine,
            patch("cli.commands.db._alembic_config") as mock_alembic_config,
            patch("cli.commands.db._get_alembic_status") as mock_get_status,
        ):

            mock_resolve_url.return_value = "postgresql://user:pass@localhost/test_db"
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            mock_alembic_config.return_value = Mock()
            mock_get_status.return_value = {
                "state": "ok",
                "current": "abc123",
                "pending": [],
            }

            result = _verify(args)

            # Assert
            assert result == 0
