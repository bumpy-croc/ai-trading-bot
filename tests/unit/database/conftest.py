"""Shared fixtures for database unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

SRC_PATH = Path(__file__).resolve().parents[3] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.database.manager import DatabaseManager


@pytest.fixture
def mock_postgresql_db():
    """Create DatabaseManager with mocked PostgreSQL for testing."""
    postgresql_url = "postgresql://test_user:test_pass@localhost:5432/test_db"

    with (
        patch("database.manager.create_engine") as mock_create_engine,
        patch("database.manager.sessionmaker") as mock_sessionmaker,
        patch("database.manager.Base"),
    ):
        mock_engine = Mock()
        mock_session_factory = Mock()
        mock_session = Mock()

        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = mock_session_factory
        mock_session_factory.return_value = mock_session

        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        mock_engine.pool = Mock()
        # Pool methods should be callable (e.g., pool.size() not pool.size)
        mock_engine.pool.size.return_value = 5
        mock_engine.pool.checkedin.return_value = 2
        mock_engine.pool.checkedout.return_value = 3
        mock_engine.pool.overflow.return_value = 1
        mock_engine.pool.invalid.return_value = 0
        mock_engine.pool.status.return_value = "Pool status"
        mock_engine.pool.dispose = Mock()

        mock_session.execute.return_value = None
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.rollback.return_value = None
        mock_session.close.return_value = None
        mock_session.query.return_value = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)

        db_manager = DatabaseManager(database_url=postgresql_url)
        db_manager._mock_engine = mock_engine
        db_manager._mock_session = mock_session

        yield db_manager
