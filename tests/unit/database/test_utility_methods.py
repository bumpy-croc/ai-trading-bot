"""Utility method tests for DatabaseManager."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


class TestUtilityMethods:
    """Test utility methods"""

    def test_cleanup_old_data(self, mock_postgresql_db):
        """Test cleaning up old data"""
        mock_session = Mock()
        mock_session.end_time = datetime.utcnow() - timedelta(days=100)
        mock_session.is_active = False

        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_session]
        mock_postgresql_db._mock_session.query.return_value = mock_query

        mock_postgresql_db.cleanup_old_data(days_to_keep=90)

        mock_postgresql_db._mock_session.delete.assert_called()
        mock_postgresql_db._mock_session.commit.assert_called()

    def test_execute_query(self, mock_postgresql_db):
        """Test executing raw SQL queries"""
        mock_result = Mock()
        mock_row = {"id": 1, "name": "test"}
        mock_result.mappings.return_value = [mock_row]

        mock_connection = Mock()
        mock_connection.exec_driver_sql.return_value = mock_result
        mock_postgresql_db._mock_engine.connect.return_value.__enter__.return_value = (
            mock_connection
        )

        result = mock_postgresql_db.execute_query("SELECT * FROM test")

        assert isinstance(result, list)
