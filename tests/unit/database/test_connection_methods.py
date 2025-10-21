"""Connection-related tests for DatabaseManager."""

import pytest

pytestmark = pytest.mark.unit


class TestConnectionMethods:
    """Test connection-related methods"""

    def test_test_connection_success(self, mock_postgresql_db):
        """Test successful database connection test"""
        result = mock_postgresql_db.test_connection()
        assert result is True

    def test_test_connection_failure(self, mock_postgresql_db):
        """Test database connection test failure"""
        mock_postgresql_db._mock_session.execute.side_effect = Exception("Connection failed")

        result = mock_postgresql_db.test_connection()
        assert result is False

    def test_get_database_info(self, mock_postgresql_db):
        """Test get_database_info method"""
        info = mock_postgresql_db.get_database_info()

        assert isinstance(info, dict)
        assert "database_url" in info
        assert "database_type" in info
        assert "connection_pool_size" in info
        assert "checked_in_connections" in info
        assert "checked_out_connections" in info
        assert info["database_type"] == "postgresql"
        assert info["connection_pool_size"] == 5

    def test_get_connection_stats(self, mock_postgresql_db):
        """Test get_connection_stats method"""
        stats = mock_postgresql_db.get_connection_stats()

        assert isinstance(stats, dict)
        assert "pool_status" in stats
        assert "checked_in" in stats
        assert "checked_out" in stats
        assert "overflow" in stats
        assert "invalid" in stats

    def test_cleanup_connection_pool(self, mock_postgresql_db):
        """Test cleanup_connection_pool method"""
        mock_postgresql_db.cleanup_connection_pool()
        mock_postgresql_db._mock_engine.pool.dispose.assert_called_once()
