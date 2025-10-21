"""Initialization tests for DatabaseManager."""

from unittest.mock import patch

import pytest

from src.database.manager import DatabaseManager

pytestmark = pytest.mark.unit


class TestInitialization:
    """Test DatabaseManager initialization"""

    def test_init_with_postgresql_url(self):
        """Test initialization with valid PostgreSQL URL"""
        postgresql_url = "postgresql://test:test@localhost:5432/test"

        with (
            patch("database.manager.create_engine"),
            patch("database.manager.sessionmaker"),
            patch("database.manager.Base.metadata.create_all"),
        ):
            db_manager = DatabaseManager(database_url=postgresql_url)
            assert db_manager.database_url == postgresql_url

    def test_init_from_environment(self):
        """Test initialization from DATABASE_URL environment variable"""
        postgresql_url = "postgresql://test:test@localhost:5432/test"

        with (
            patch("database.manager.get_config") as mock_config,
            patch("database.manager.create_engine"),
            patch("database.manager.sessionmaker"),
            patch("database.manager.Base.metadata.create_all"),
        ):
            mock_config.return_value.get.return_value = postgresql_url

            db_manager = DatabaseManager()
            assert db_manager.database_url == postgresql_url

    def test_init_fails_without_database_url(self):
        """Test initialization fails without DATABASE_URL"""
        with patch("database.manager.get_config") as mock_config:
            mock_config.return_value.get.return_value = None

            with pytest.raises(ValueError, match="DATABASE_URL environment variable is required"):
                DatabaseManager()

    def test_init_fails_with_non_postgresql_url(self):
        """Test initialization fails with non-PostgreSQL URL"""
        sqlite_url = "mysql://user:pass@localhost:3306/testdb"

        with pytest.raises(ValueError, match="Only PostgreSQL databases are supported"):
            DatabaseManager(database_url=sqlite_url)
