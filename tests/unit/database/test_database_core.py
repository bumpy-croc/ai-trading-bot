#!/usr/bin/env python3
"""
Unit tests for DatabaseManager core functionality.
Tests initialization, connection methods, and session management.
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src")
)

from src.database.manager import DatabaseManager
from src.database.models import OrderStatus, PositionSide, PositionStatus

pytestmark = pytest.mark.unit


class TestDatabaseManager:
    """Base test suite for DatabaseManager"""

    @pytest.fixture
    def mock_postgresql_db(self):
        """Create DatabaseManager with mocked PostgreSQL for testing"""
        postgresql_url = "postgresql://test_user:test_pass@localhost:5432/test_db"

        with (
            patch("database.manager.create_engine") as mock_create_engine,
            patch("database.manager.sessionmaker") as mock_sessionmaker,
            patch("database.manager.Base"),
        ):
            # Setup mocks
            mock_engine = Mock()
            mock_session_factory = Mock()
            mock_session = Mock()

            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory
            mock_session_factory.return_value = mock_session

            # Mock engine connection
            mock_connection = Mock()
            mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
            mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

            # Mock connection pool
            mock_engine.pool = Mock()
            mock_engine.pool.size = 5
            mock_engine.pool.checkedin = 2
            mock_engine.pool.checkedout = 3
            mock_engine.pool.overflow = 1
            mock_engine.pool.invalid = 0
            mock_engine.pool.status.return_value = "Pool status"
            mock_engine.pool.dispose = Mock()

            # Mock session behavior
            mock_session.execute.return_value = None
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            mock_session.rollback.return_value = None
            mock_session.close.return_value = None
            mock_session.query.return_value = Mock()
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)

            # Create database manager
            db_manager = DatabaseManager(database_url=postgresql_url)
            db_manager._mock_engine = mock_engine
            db_manager._mock_session = mock_session

            yield db_manager


class TestInitialization(TestDatabaseManager):
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


class TestConnectionMethods(TestDatabaseManager):
    """Test connection-related methods"""

    def test_test_connection_success(self, mock_postgresql_db):
        """Test successful database connection test"""
        with patch.object(mock_postgresql_db, "_mock_engine") as mock_engine:
            mock_engine.connect.return_value.__enter__.return_value = Mock()
            result = mock_postgresql_db.test_connection()
            assert result is True

    def test_test_connection_failure(self, mock_postgresql_db):
        """Test database connection test failure"""
        with patch.object(mock_postgresql_db, "_mock_engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("Connection failed")
            result = mock_postgresql_db.test_connection()
            assert result is False

    def test_get_connection_pool_status(self, mock_postgresql_db):
        """Test connection pool status retrieval"""
        status = mock_postgresql_db.get_connection_pool_status()

        expected_status = {
            "pool_size": 5,
            "checked_in": 2,
            "checked_out": 3,
            "overflow": 1,
            "invalid": 0,
        }

        assert status == expected_status

    def test_close_connection_pool(self, mock_postgresql_db):
        """Test connection pool closure"""
        mock_postgresql_db.close_connection_pool()
        mock_postgresql_db._mock_engine.pool.dispose.assert_called_once()


class TestSessionManagement(TestDatabaseManager):
    """Test session management methods"""

    def test_get_session_context_manager(self, mock_postgresql_db):
        """Test session context manager"""
        with mock_postgresql_db.get_session() as session:
            assert session is mock_postgresql_db._mock_session

    def test_get_session_exception_handling(self, mock_postgresql_db):
        """Test session exception handling and rollback"""
        mock_session = mock_postgresql_db._mock_session

        with pytest.raises(Exception):
            with mock_postgresql_db.get_session() as session:
                raise Exception("Test exception")

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_get_session_normal_commit(self, mock_postgresql_db):
        """Test session normal commit and close"""
        mock_session = mock_postgresql_db._mock_session

        with mock_postgresql_db.get_session() as session:
            pass  # Normal execution

        # Should commit and close on normal exit
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()